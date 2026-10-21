"""exp003: an attention kernel that reads packed fp4 cache rows directly.

:class:`~priml.baselines.kvcache.fp4.FP4KVCache` stores e2m1 codes cheaply but
hands attention a dequantized copy every step, so decode moves more bytes than
bf16's. This module closes that pass: :class:`FP4PackedKVCache` stores the same
rows and serves the packed slices themselves, and :class:`Fp4TritonAttention`
is a Triton decode kernel that loads a packed byte, looks its two nibbles up in
a 16-entry table, scales them in registers, and attends -- codes are never
materialized as a cache-sized bf16 tensor.

The kernel's arithmetic matches :func:`~priml.baselines.kvcache.fp4._dequantized`
per value (float32 table and scale, rounded to the incoming dtype before the
dot), so logits differ from the materialized path only by execution order.
It is an experiment kernel: dropout, windows, and explicit masks are refused,
and odd channel counts are rejected (the frozen decode ladder is even-width,
full-context, and maskless on decode).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, override

from configgle import Fig
from torch import Tensor, nn
from triton import language

import torch

from priml.baselines.kvcache.fp4 import (
    _CODE_VALUES,
    FP4KVCache,
)
from priml.kernel import jit_kernel, require_power_of_two
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.attention.kvcache import KVCache, alloc_layer_cache


if TYPE_CHECKING:
    import triton

    from priml.cost import Cost
    from priml.model.attention.kvcache import HasModules
    from priml.model.custom_types import DepthIndex
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")


__all__ = [
    "FP4PackedKVCache",
    "Fp4TritonAttention",
    "fp4_fused_layer_cache",
]


_SCALE_CODE_BYTES = 4
"""Bytes of float32 scale that trail a packed row; matches fp4.SCALE_COLUMNS."""


@jit_kernel
def _fp4_attention_forward(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    q_ptr: language.tensor,
    k_ptr: language.tensor,
    v_ptr: language.tensor,
    ks_ptr: language.tensor,
    vs_ptr: language.tensor,
    out_ptr: language.tensor,
    table_ptr: language.tensor,
    q_b: int,
    q_s: int,
    q_h: int,
    k_b: int,
    k_h: int,
    k_l: int,
    v_b: int,
    v_h: int,
    v_l: int,
    ks_b: int,
    ks_h: int,
    ks_l: int,
    vs_b: int,
    vs_h: int,
    vs_l: int,
    out_b: int,
    out_s: int,
    out_h: int,
    heads: language.constexpr,
    seq: language.constexpr,
    length: int,
    channels: language.constexpr,
    code_width: language.constexpr,
    block_l: language.constexpr,
    block_s: language.constexpr,
    causal: language.constexpr,
    logit_scale: language.constexpr,
) -> None:
    """One program per (batch, query head): stream keys, online softmax."""
    pid = language.program_id(0)
    b = pid // heads
    h = pid - b * heads
    s = language.arange(0, block_s)
    d = language.arange(0, channels)
    q = language.load(
        q_ptr + b * q_b + s[:, None] * q_s + h * q_h + d,
        mask=(s < seq)[:, None],
        other=0.0,
    )
    acc = language.zeros((block_s, channels), dtype=language.float32)
    m = language.full((block_s,), float("-inf"), dtype=language.float32)
    lse = language.zeros((block_s,), dtype=language.float32)
    lc = language.arange(0, block_l)
    wc = language.arange(0, code_width)
    for start in range(0, length, block_l):
        j = start + lc
        valid = j < length
        codes = language.load(
            k_ptr + b * k_b + h * k_h + j[:, None] * k_l + wc,
            mask=valid[:, None],
            other=0,
        )
        high = language.load(table_ptr + (codes.to(language.int32) >> 4))
        low = language.load(table_ptr + (codes.to(language.int32) & 15))
        key = language.reshape(language.join(high, low), block_l, channels)
        scale = language.load(
            ks_ptr + b * ks_b + h * ks_h + j * ks_l,
            mask=valid,
            other=1.0,
        )
        key = (key.to(language.float32) * scale[:, None]).to(language.bfloat16)
        logits = language.dot(q, language.trans(key)) * logit_scale
        if causal:
            query_pos = (length - seq) + s[:, None]
            logits = language.where(j[None, :] <= query_pos, logits, float("-inf"))
        logits = language.where(valid[None, :], logits, float("-inf"))
        m_new = language.maximum(m, language.max(logits, 1))
        alpha = language.exp(m - m_new)
        p = language.exp(logits - m_new[:, None])
        acc = acc * alpha[:, None]
        vcodes = language.load(
            v_ptr + b * v_b + h * v_h + j[:, None] * v_l + wc,
            mask=valid[:, None],
            other=0,
        )
        vhigh = language.load(table_ptr + (vcodes.to(language.int32) >> 4))
        vlow = language.load(table_ptr + (vcodes.to(language.int32) & 15))
        value = language.reshape(language.join(vhigh, vlow), block_l, channels)
        vscale = language.load(
            vs_ptr + b * vs_b + h * vs_h + j * vs_l,
            mask=valid,
            other=1.0,
        )
        value = (value.to(language.float32) * vscale[:, None]).to(language.bfloat16)
        acc += language.dot(p.to(language.bfloat16), value)
        lse = lse * alpha + language.sum(p, 1)
        m = m_new
    out = acc / lse[:, None]
    language.store(
        out_ptr + b * out_b + s[:, None] * out_s + h * out_h + d,
        out,
        mask=(s < seq)[:, None],
    )


@lru_cache(maxsize=8)
def _code_table(device: torch.device) -> Tensor:
    """Return the 16 signed e2m1 values as a bf16 lookup row, once per device."""
    return torch.tensor(_CODE_VALUES, device=device, dtype=torch.bfloat16)


class Fp4TritonAttention(nn.Module):
    """Attention over packed fp4 cache rows: the exp003 read path.

    The layout is every kernel's: ``[..., S, num_heads, channels_head]`` for
    queries and ``[..., S, heads, packed]`` for the packed keys and values
    :class:`FP4PackedKVCache` serves. The kernel refuses anything the frozen
    decode ladder does not use.
    """

    class Config(Fig["Fp4TritonAttention"]):
        channels_head: int = -1
        """Query/key width per head; must be even (two codes per byte)."""

        block_l: int = 128
        """Keys streamed per loop iteration; a power of two."""

        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the logical attention; the fusion hides no FLOPs.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Ignored; the dense accounting overreaches by policy.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V, or ``-1`` to use the key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation FLOPs and logical tensor bytes.

            """
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=-1,
                dropout_p=dropout_p,
                rows=rows,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_head % 2:
            raise ValueError(
                f"channels_head={config.channels_head} must be even; two e2m1 "
                "codes ride in one byte.",
            )
        require_power_of_two(block_l=config.block_l)
        self.channels_head = config.channels_head
        self.block_l = config.block_l

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        attn_mask: Tensor | None = None,
        window: int = -1,
        scale: float | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Attend ``q`` over packed ``k``/``v`` rows.

        Args:
          q: Queries ``[batch, seq, heads, channels]``, bfloat16.
          k: Packed keys ``[batch, length, heads, code_width + 4]``, uint8.
          v: Packed values, shaped like ``k``.
          dropout_p: Must be 0; this kernel trains nothing.
          is_causal: Bottom-right causal masking; the frozen ladder's calls.
          attn_mask: Must be None; decode is unmasked and prefill is causal.
          window: Must be -1; no windowed attention here.
          scale: Logit scale; None is ``channels ** -0.5``.
          **kwargs: The rest of the bus, unread.

        Returns:
          out: ``[batch, seq, heads, channels]``, bfloat16.

        Raises:
          ValueError: Dropout, a mask or a window was asked for, the dtypes
            or widths are not the packed contract, or the launch is empty.

        """
        del kwargs
        if dropout_p or attn_mask is not None or window != -1:
            raise ValueError(
                "Fp4TritonAttention supports the decode ladder only: no "
                "dropout, mask, or window.",
            )
        if (
            q.dtype != torch.bfloat16
            or k.dtype != torch.uint8
            or v.dtype != torch.uint8
        ):
            raise ValueError(
                f"Expected bf16 queries over uint8 packed rows, got "
                f"{q.dtype}, {k.dtype}, {v.dtype}.",
            )
        batch, seq, heads, channels = q.shape
        length = k.shape[1]
        if (
            channels != self.channels_head
            or k.shape[-1] != channels // 2 + _SCALE_CODE_BYTES
        ):
            raise ValueError(
                f"Queries carry {channels} channels over packed width "
                f"{k.shape[-1]}; this kernel was built for "
                f"{self.channels_head} + {_SCALE_CODE_BYTES}.",
            )
        if k.shape != v.shape or k.shape[:2] != (batch, length) or k.shape[2] != heads:
            raise ValueError(
                f"Packed keys {tuple(k.shape)} must match values and line up "
                f"with queries {(batch, length, heads)}.",
            )
        if not length or not seq:
            raise ValueError(
                f"Got an empty launch: batch={batch} seq={seq} length={length}."
            )
        k = k.movedim(-3, -2).contiguous()
        v = v.movedim(-3, -2).contiguous()
        ks = (
            k[..., channels // 2 :]
            .clone(memory_format=torch.contiguous_format)
            .view(
                torch.float32,
            )
        )
        vs = (
            v[..., channels // 2 :]
            .clone(memory_format=torch.contiguous_format)
            .view(
                torch.float32,
            )
        )
        out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        table = _code_table(q.device)
        block_s = max(16, 1 << (seq - 1).bit_length())
        _fp4_attention_forward[(batch * heads,)](
            q,
            k,
            v,
            ks,
            vs,
            out,
            table,
            q_b=q.stride(0),
            q_s=q.stride(1),
            q_h=q.stride(2),
            k_b=k.stride(0),
            k_h=k.stride(1),
            k_l=k.stride(2),
            v_b=v.stride(0),
            v_h=v.stride(1),
            v_l=v.stride(2),
            ks_b=ks.stride(0),
            ks_h=ks.stride(1),
            ks_l=ks.stride(2),
            vs_b=vs.stride(0),
            vs_h=vs.stride(1),
            vs_l=vs.stride(2),
            out_b=out.stride(0),
            out_s=out.stride(1),
            out_h=out.stride(2),
            heads=heads,
            seq=seq,
            length=length,
            channels=channels,
            code_width=channels // 2,
            block_l=self.block_l,
            block_s=block_s,
            causal=is_causal,
            logit_scale=channels**-0.5 if scale is None else scale,
        )
        return out.to(q.dtype)


class FP4PackedKVCache(FP4KVCache):
    """fp4 cache that serves packed rows instead of a dequantized copy."""

    @override
    def _served(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        del k, v
        return self.k[..., : self.length, :], self.v[..., : self.length, :]


def fp4_fused_layer_cache(
    model: HasModules,
    *,
    batch: int | tuple[int, ...],
    max_seq: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[DepthIndex, object]:
    """Allocate one :class:`FP4PackedKVCache` per cached attention.

    Same contract as :func:`fp4_layer_cache`; only the served slice differs.

    Args:
      model: Module tree containing cached attentions.
      batch: Batch size or batch shape.
      max_seq: Maximum sequence length.
      device: Device for allocated tensors.
      dtype: Accepted for the ``allocate_cache`` slot and ignored; storage
        is uint8 by definition.

    Returns:
      cache: Mutable state keyed by each attention's full depth index.

    """
    del dtype
    result: dict[DepthIndex, object] = {}
    allocated = alloc_layer_cache(
        model,
        batch=batch,
        max_seq=0,
        device=device,
    )
    for key, state in allocated.items():
        assert isinstance(state, KVCache)
        result[key] = FP4PackedKVCache.alloc(
            batch=tuple(state.k.shape[:-3]),
            num_heads=state.k.shape[-3],
            max_seq=max_seq,
            channels_head=state.k.shape[-1],
            channels_v_head=state.v.shape[-1],
            device=state.k.device,
        )
    return result
