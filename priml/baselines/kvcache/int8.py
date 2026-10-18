"""INT8 per-token KV cache: the storage ``exp001`` swaps into the decode chain.

Every token's row of keys, and of values, is quantized symmetrically to
signed int8 against one scale computed from that row's absolute maximum.
The float32 scale's bits ride in the row's trailing columns, so ``k`` and
``v`` alone carry the whole cache and :func:`priml.baselines.kvcache.decode`
measured ``kv_bytes`` counts codes and scales together.

Attention never sees the codes: :meth:`Int8KVCache.update` returns the
dequantized valid slice in the incoming dtype, shaped exactly like the
float cache's answer, so the model's read path is unchanged.

The cache refuses :meth:`Int8KVCache.freeze`; a frozen view of the base
class would hand int8 codes and scale bits straight to attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

from torch import Tensor

import torch

from priml.model.attention.kvcache import KVCache, alloc_layer_cache


if TYPE_CHECKING:
    from priml.model.attention.kvcache import HasModules
    from priml.model.custom_types import DepthIndex


SCALE_COLUMNS: Final = 4
"""Trailing int8 columns per row holding one float32 scale, bit for bit."""

_CODE_MAX: Final = 127.0
"""Magnitude of the largest code: symmetric int8 range is -127..127."""


class Int8KVCache(KVCache):
    """KV cache storing int8 codes and one packed scale per token.

    ``k`` is int8 ``[..., H, max_seq, D + SCALE_COLUMNS]``: the first ``D``
    columns hold one token's key codes, the last ``SCALE_COLUMNS`` that
    row's float32 scale, bit for bit. ``v`` is the same over its own
    channel count. Rows stay whole under FIFO eviction, so codes and scales
    always shift together.
    """

    __slots__ = ()

    @override
    def freeze(self) -> KVCache:
        """Refuse a frozen view, which would serve raw codes to attention.

        Raises:
          NotImplementedError: Always; this cache has no float view to freeze.

        """
        raise NotImplementedError(
            "Int8KVCache cannot freeze: the shared-storage view would pass "
            "int8 codes and scale bits to attention unquantized.",
        )

    @override
    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize ``k`` and ``v`` into the cache, return the valid slice.

        Same contract as :meth:`KVCache.update`: the cache keeps the most
        recent ``max_seq`` tokens (FIFO when an update overflows), ``seen``
        counts every token ever written, and a rejected update leaves the
        cache as it was.

        Args:
          k: New keys ``[..., seq_len, channels_head]``, any float dtype.
          v: New values ``[..., seq_len, channels_v_head]``, likewise.

        Returns:
          k: Dequantized keys ``[..., length, channels_head]`` in ``k``'s dtype.
          v: Dequantized values ``[..., length, channels_v_head]`` in ``v``'s dtype.

        Raises:
          ValueError: ``k`` and ``v`` differ in length, or either cannot be
            written into the cache.

        """
        s = k.shape[-2]
        if v.shape[-2] != s:
            raise ValueError(
                f"Got {s} keys and {v.shape[-2]} values; keys and values pair "
                "up one per token.",
            )
        if s > self.max_seq:
            raise ValueError(
                f"Update length {s} exceeds cache capacity {self.max_seq}.",
            )
        for new, store in ((k, self.k), (v, self.v)):
            slot = store[..., :s, :-SCALE_COLUMNS].shape
            try:
                fits = torch.broadcast_shapes(new.shape, slot) == slot
            except RuntimeError:
                fits = False
            if not fits:
                raise ValueError(
                    f"Cannot write {tuple(new.shape)} into the cache's "
                    f"{tuple(slot)} slot.",
                )
        overflow = self.length + s - self.max_seq
        if overflow > 0:
            keep = self.length - overflow
            self.k[..., :keep, :] = self.k[..., overflow : self.length, :].clone()
            self.v[..., :keep, :] = self.v[..., overflow : self.length, :].clone()
            self.length = keep
        end = self.length + s
        if s:
            codes, scales = _quantize(k)
            self.k[..., self.length : end, :] = torch.cat((codes, scales), dim=-1)
            codes, scales = _quantize(v)
            self.v[..., self.length : end, :] = torch.cat((codes, scales), dim=-1)
        self.length = end
        self.seen += s
        return (
            _dequantized(self.k, end, k.dtype),
            _dequantized(self.v, end, v.dtype),
        )


def int8_layer_cache(
    model: HasModules,
    *,
    batch: int | tuple[int, ...],
    max_seq: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[DepthIndex, object]:
    """Allocate one :class:`Int8KVCache` per cached attention in ``model``.

    Each slot mirrors :func:`alloc_layer_cache` -- same keys, same shapes --
    widened by ``SCALE_COLUMNS`` so the scales live inside ``k`` and ``v``
    and the benchmark's byte count sees them.

    Args:
      model: Module tree containing cached attentions.
      batch: Batch size or batch shape.
      max_seq: Maximum sequence length.
      device: Device for allocated tensors.
      dtype: Accepted for the ``allocate_cache`` slot and ignored; storage
        is int8 by definition.

    Returns:
      cache: Mutable state keyed by each attention's full depth index.

    """
    del dtype
    result: dict[DepthIndex, object] = {}
    allocated = alloc_layer_cache(
        model,
        batch=batch,
        max_seq=max_seq,
        device=device,
        dtype=torch.int8,
    )
    for key, state in allocated.items():
        assert isinstance(state, KVCache)
        result[key] = Int8KVCache(
            k=torch.zeros(
                (*state.k.shape[:-1], state.k.shape[-1] + SCALE_COLUMNS),
                dtype=torch.int8,
                device=state.k.device,
            ),
            v=torch.zeros(
                (*state.v.shape[:-1], state.v.shape[-1] + SCALE_COLUMNS),
                dtype=torch.int8,
                device=state.v.device,
            ),
            length=0,
        )
    return result


def _quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    """Turn a float row into int8 codes plus its packed float32 scale.

    Args:
      x: Incoming keys or values ``[..., seq_len, channels]``.

    Returns:
      codes: ``[..., seq_len, channels]`` int8, ``x / scale`` rounded.
      scales: ``[..., seq_len, SCALE_COLUMNS]`` int8 holding the float32
        scale's bits; an all-zero row stores scale 1, so its codes are 0.

    """
    scale = x.float().abs().amax(dim=-1, keepdim=True).div(_CODE_MAX)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(x.float().div(scale)).clamp_(-_CODE_MAX, _CODE_MAX)
    codes = codes.to(torch.int8)
    scales = scale.view(torch.int32).view(torch.int8)
    return codes, scales


def _dequantized(store: Tensor, end: int, dtype: torch.dtype) -> Tensor:
    """Read the cache's first ``end`` rows back as floats in ``dtype``.

    Args:
      store: Packed int8 cache ``[..., max_seq, channels + SCALE_COLUMNS]``.
      end: Number of valid rows to return.
      dtype: Dtype the attention block reads in.

    Returns:
      values: ``[..., end, channels]`` dequantized rows in ``dtype``.

    """
    codes = store[..., :end, :-SCALE_COLUMNS]
    scale = store[..., :end, -SCALE_COLUMNS:].contiguous().view(torch.float32)
    return codes.to(dtype) * scale.to(dtype)
