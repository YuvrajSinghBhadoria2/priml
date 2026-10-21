"""FP4 (e2m1) per-token KV cache: the storage ``exp002`` swaps into the decode chain.

Every token's row of keys, and of values, is quantized to the 16-value e2m1
grid against one scale computed from that row's absolute maximum. Two codes
ride in one unsigned byte, and the float32 scale's bits occupy the row's
trailing columns, so ``k`` and ``v`` alone carry the whole cache and
:func:`priml.baselines.kvcache.decode` measured ``kv_bytes`` counts codes and
scales together.

Torch's native ``float4_e2m1fn_x2`` tensor still cannot be filled or
converted on every device this harness runs, so the codes stay in uint8 and
a 256-entry lookup table unpacks a byte's two nibbles on read in one gather. Attention never sees the codes:
:meth:`FP4KVCache.update` returns the dequantized valid slice in the
incoming dtype, shaped exactly like the float cache's answer, so the model's
read path is unchanged.

The cache refuses :meth:`FP4KVCache.freeze`; a frozen view of the base class
would hand packed codes and scale bits straight to attention.

The scale is rounded to float32 before encoding and clamped between its
smallest positive subnormal (2**-149) and largest finite value. Finite
float64 rows exceeding six times that maximum saturate; sufficiently small
values round to zero. Subnormal scales have coarser relative precision.
Decoded values saturate at the incoming dtype's finite limits. This is
the e2m1 value grid with float32 per-row scales, not the complete MX format;
midpoint ties select the lower magnitude and signed zero is preserved.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, override

from torch import Tensor

import torch

from priml.model.attention.attention import Attention
from priml.model.attention.kvcache import KVCache, alloc_layer_cache
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.custom_types import is_cached_attention


if TYPE_CHECKING:
    from priml.model.attention.kvcache import HasModules
    from priml.model.custom_types import DepthIndex


SCALE_COLUMNS: Final = 4
"""Trailing uint8 columns per row holding one float32 scale, bit for bit."""

_MAGNITUDE_MAX: Final = 6.0
"""Largest magnitude the e2m1 grid represents, so a row's peak maps there."""

_MIDPOINTS: Final = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
"""Gaps between the e2m1 magnitudes; a ratio snaps to the nearer side."""

_CODE_VALUES: Final = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
"""What each of the 16 e2m1 codes holds; the sign bit selects code + 8."""


class FP4KVCache(KVCache):
    """KV cache storing packed e2m1 codes and one packed scale per token.

    ``k`` is uint8 ``[..., H, max_seq, packed + SCALE_COLUMNS]`` where
    ``packed = ceil(D / 2)``: the first ``packed`` columns hold one token's
    key codes, two e2m1 values per byte (the even channel in the high
    nibble), the last ``SCALE_COLUMNS`` that row's float32 scale, bit for
    bit. ``v`` is the same over its own channel count. The served channel
    counts live on the cache because the packed width alone cannot say
    whether an odd ``D`` left its final low nibble as padding. Rows stay
    whole under FIFO eviction, so codes and scales always shift together.
    """

    __slots__ = ("channels_head", "channels_v_head")

    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        length: int | None = None,
        *,
        channels_head: int,
        channels_v_head: int | None = None,
    ) -> None:
        """Wrap packed rows, remembering the channel counts they encode.

        Args:
          k: Packed key rows ``[..., H, max_seq, packed + SCALE_COLUMNS]``.
          v: Packed value rows, likewise over its own channel count.
          length: Tokens already written; defaults to every allocated row.
          channels_head: Key channels the cache serves, padding excluded.
          channels_v_head: Value channels served; defaults to key channels.

        """
        super().__init__(k, v, length)
        self.channels_head = channels_head
        self.channels_v_head = (
            channels_head if channels_v_head is None else channels_v_head
        )

    @classmethod
    @override
    def alloc(
        cls,
        *,
        batch: int | tuple[int, ...],
        num_heads: int,
        max_seq: int,
        channels_head: int,
        channels_v_head: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate packed uint8 rows, including each row's float32 scale.

        Args:
          batch: Batch size or batch shape.
          num_heads: Number of cached heads.
          max_seq: Maximum sequence length.
          channels_head: Key channels per head, excluding the packed scale.
          channels_v_head: Value channels per head; defaults to key channels.
          device: Device for allocated tensors.
          dtype: Accepted for the base allocator contract; storage is uint8.

        Returns:
          cache: An empty packed cache with zero length and tokens seen.

        """
        del dtype
        v_dim = channels_head if channels_v_head is None else channels_v_head
        for name, channels in (
            ("channels_head", channels_head),
            ("channels_v_head", v_dim),
        ):
            # The packed width and the scale columns would hide a negative.
            if channels < 0:
                raise ValueError(f"{name} must be non-negative; got {channels}.")
        if isinstance(batch, int):
            batch = (batch,)
        return cls(
            k=torch.zeros(
                (*batch, num_heads, max_seq, _packed(channels_head) + SCALE_COLUMNS),
                device=device,
                dtype=torch.uint8,
            ),
            v=torch.zeros(
                (*batch, num_heads, max_seq, _packed(v_dim) + SCALE_COLUMNS),
                device=device,
                dtype=torch.uint8,
            ),
            length=0,
            channels_head=channels_head,
            channels_v_head=v_dim,
        )

    @override
    def freeze(self) -> KVCache:
        """Refuse a frozen view, which would serve raw codes to attention.

        Raises:
          NotImplementedError: Always; this cache has no float view to freeze.

        """
        raise NotImplementedError(
            "FP4KVCache cannot freeze: the shared-storage view would pass "
            "packed e2m1 codes and scale bits to attention unquantized.",
        )

    @override
    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize ``k`` and ``v`` into the cache, return the valid slice.

        Same contract as :meth:`KVCache.update`: the cache keeps the most
        recent ``max_seq`` tokens (FIFO when an update overflows), ``seen``
        counts every token ever written, and a rejected update leaves the
        cache as it was.

        Args:
          k: New keys ``[..., seq_len, channels_head]``; float inputs use the
            float32 scale range described in the module docstring.
          v: New values ``[..., seq_len, channels_v_head]``, likewise.

        Returns:
          k: Dequantized keys ``[..., length, channels_head]`` in ``k``'s dtype.
          v: Dequantized values ``[..., length, channels_v_head]`` in ``v``'s dtype.

        Raises:
          TypeError: Either input has a non-floating dtype.
          ValueError: ``k`` and ``v`` differ in length, or either cannot be
            written into the cache.

        """
        if not k.is_floating_point() or not v.is_floating_point():
            raise TypeError(
                "KV cache keys and values must have floating-point dtypes; "
                f"got {k.dtype} and {v.dtype}.",
            )
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
        packed: list[Tensor] = []
        for new, store, channels in (
            (k, self.k, self.channels_head),
            (v, self.v, self.channels_v_head),
        ):
            slot = (*store.shape[:-3], store.shape[-3], s, channels)
            try:
                fits = torch.broadcast_shapes(new.shape, slot) == slot
            except RuntimeError:
                fits = False
            if not fits:
                raise ValueError(
                    f"Cannot write {tuple(new.shape)} into the cache's "
                    f"{tuple(slot)} slot.",
                )
            # Moved here, before eviction, so a value that cannot reach the
            # cache's device fails with the cache untouched.
            packed.append(_quantize(new.expand(slot)).to(store.device))
        overflow = self.length + s - self.max_seq
        if overflow > 0:
            keep = self.length - overflow
            self.k[..., :keep, :] = self.k[..., overflow : self.length, :].clone()
            self.v[..., :keep, :] = self.v[..., overflow : self.length, :].clone()
            self.length = keep
        end = self.length + s
        self.k[..., self.length : end, :] = packed[0]
        self.v[..., self.length : end, :] = packed[1]
        self.length = end
        self.seen += s
        return self._served(k, v)

    def _served(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Return the valid slice this cache hands its attention."""
        return (
            _dequantized(
                self.k,
                end=self.length,
                channels=self.channels_head,
                dtype=k.dtype,
            ),
            _dequantized(
                self.v,
                end=self.length,
                channels=self.channels_v_head,
                dtype=v.dtype,
            ),
        )


def fp4_layer_cache(
    model: HasModules,
    *,
    batch: int | tuple[int, ...],
    max_seq: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[DepthIndex, object]:
    """Allocate one :class:`FP4KVCache` per cached attention in ``model``.

    Supports :class:`Attention` (including GQA) and
    :class:`MultiHeadLatentAttention`. Other cached attention families are
    rejected before allocation because they inspect raw storage geometry or
    require tuple/recurrent state.

    Each slot mirrors :func:`alloc_layer_cache` -- same keys, same logical shapes --
    narrowed to packed rows widened by ``SCALE_COLUMNS`` so the scales live
    inside ``k`` and ``v`` and the benchmark's byte count sees them.

    Args:
      model: Module tree containing cached attentions.
      batch: Batch size or batch shape.
      max_seq: Maximum sequence length.
      device: Device for allocated tensors.
      dtype: Accepted for the ``allocate_cache`` slot and ignored; storage
        is uint8 by definition.

    Returns:
      cache: Mutable state keyed by each attention's full depth index.

    Raises:
      TypeError: The model contains an unsupported cached attention family.

    """
    del dtype
    for module in model.modules():
        if is_cached_attention(module) and not isinstance(
            module,
            (Attention, MultiHeadLatentAttention),
        ):
            raise TypeError(
                f"Cached attention {type(module).__name__} is not supported by "
                "fp4_layer_cache; supported families are Attention (including "
                "GQA) and MultiHeadLatentAttention.",
            )
    result: dict[DepthIndex, object] = {}
    allocated = alloc_layer_cache(
        model,
        batch=batch,
        max_seq=0,
        device=device,
    )
    for key, state in allocated.items():
        assert isinstance(state, KVCache)
        result[key] = FP4KVCache.alloc(
            batch=tuple(state.k.shape[:-3]),
            num_heads=state.k.shape[-3],
            max_seq=max_seq,
            channels_head=state.k.shape[-1],
            channels_v_head=state.v.shape[-1],
            device=state.k.device,
        )
    return result


def _packed(channels: int) -> int:
    """Bytes holding ``channels`` e2m1 codes, two values per byte."""
    return (channels + 1) // 2


def _quantize(x: Tensor) -> Tensor:
    """Pack a float row as e2m1 nibbles followed by its float32 scale."""
    values = x.to(dtype=_working_dtype(x.dtype))
    # A zero-width row (MLA without a rotary key) has no peak to reduce.
    peak = (
        values.abs().amax(dim=-1, keepdim=True)
        if values.shape[-1]
        else values.new_zeros((*values.shape[:-1], 1))
    )
    limits = torch.finfo(torch.float32)
    scale = (
        peak.div(_MAGNITUDE_MAX)
        .clamp(
            min=limits.tiny * limits.eps,
            max=limits.max,
        )
        .float()
    )
    scale = torch.where(peak == 0, torch.ones_like(scale), scale)
    midpoints = _midpoints(scale.device, values.dtype)
    # The expand that broadcast the update leaves stride-0 axes; bucketize
    # asks for a contiguous view, and contiguous() is a no-op otherwise.
    ratios = values.abs().div(scale).contiguous()
    index = torch.bucketize(ratios, midpoints)
    codes = index.to(torch.uint8) | (values.signbit().to(torch.uint8) << 3)
    if codes.shape[-1] % 2:
        # An odd row parks its last code in the high nibble; read trims.
        codes = torch.cat((codes, codes.new_zeros((*codes.shape[:-1], 1))), dim=-1)
    packed = (codes[..., 0::2] << 4) | codes[..., 1::2]
    return torch.cat((packed, scale.view(torch.uint8)), dim=-1)


def _dequantized(
    store: Tensor,
    *,
    end: int,
    channels: int,
    dtype: torch.dtype,
) -> Tensor:
    """Read the cache's first ``end`` rows back as floats in ``dtype``."""
    packed = store[..., :end, :-SCALE_COLUMNS]
    # A plain clone preserves singleton strides; dtype views also require
    # divisible strides on those axes, even when no rows are valid yet.
    scale = (
        store[..., :end, -SCALE_COLUMNS:]
        .clone(memory_format=torch.contiguous_format)
        .view(torch.float32)
    )
    pairs = _pair_table(store.device, _working_dtype(dtype))
    values = pairs[packed.long()].flatten(-2)[..., :channels]
    limit = torch.finfo(dtype).max
    return (values * scale).clamp(min=-limit, max=limit).to(dtype)


@lru_cache(maxsize=8)
def _pair_table(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Map every packed byte to its (high, low) signed e2m1 values."""
    codes = torch.tensor(_CODE_VALUES, device=device, dtype=dtype)
    byte = torch.arange(256, device=device, dtype=torch.uint8)
    high = codes[(byte >> 4).long()]
    low = codes[(byte & 0x0F).long()]
    return torch.stack((high, low), dim=-1)


@lru_cache(maxsize=8)
def _midpoints(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build the snap midpoints, once per device."""
    return torch.tensor(_MIDPOINTS, device=device, dtype=dtype)


# Every dtype narrower than float32 promotes to it; float8 cannot be asked.
def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return ``dtype`` or float32, whichever is wider."""
    if dtype.itemsize < torch.float32.itemsize:
        return torch.float32
    return torch.promote_types(dtype, torch.float32)
