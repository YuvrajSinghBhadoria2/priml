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
It is an experiment kernel: dropout, windows, and explicit masks are refused.
Head widths and key tiles must be powers of two of at least 16, as required
by the Triton dot products.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.baselines.kvcache.fp4 import (
    _CODE_VALUES,
    SCALE_COLUMNS,
    FP4KVCache,
)
from priml.kernel import jit_kernel, require_power_of_two
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.attention.kvcache import KVCache, alloc_layer_cache
from priml.model.custom_types import is_cached_attention


if TYPE_CHECKING:
    from triton import language

    import triton

    from priml.cost import Cost
    from priml.model.attention.kvcache import HasModules
    from priml.model.custom_types import DepthIndex
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")


__all__ = [
    "FP4PackedKVCache",
    "Fp4TritonAttention",
    "fp4_fused_layer_cache",
]


class Fp4TritonAttention(nn.Module):
    """Attention over packed fp4 cache rows: the exp003 read path.

    The layout is every kernel's: ``[..., S, num_heads, channels_head]`` for
    queries and ``[..., S, heads, packed]`` for the packed keys and values
    :class:`FP4PackedKVCache` serves. The kernel refuses anything the frozen
    decode ladder does not use.
    """

    class Config(Fig["Fp4TritonAttention"]):
        block_l: int = 128
        """Keys streamed per loop iteration; a power of two of at least 16."""

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
        require_power_of_two(block_l=config.block_l)
        if config.block_l < 16:
            raise ValueError(f"block_l must be at least 16; got {config.block_l}.")
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
        if channels < 16 or channels & (channels - 1):
            raise ValueError(
                f"Query channels must be a power of two of at least 16; got {channels}.",
            )
        if k.shape[-1] != channels // 2 + SCALE_COLUMNS:
            raise ValueError(
                f"Queries carry {channels} channels over packed width "
                f"{k.shape[-1]}; expected {channels // 2 + SCALE_COLUMNS}.",
            )
        if k.shape != v.shape or k.shape[:2] != (batch, length) or k.shape[2] != heads:
            raise ValueError(
                f"Packed keys {tuple(k.shape)} must match values and line up "
                f"with queries {(batch, length, heads)}.",
            )
        if not length or not seq:
            raise ValueError(
                f"Got an empty launch: batch={batch} seq={seq} length={length}.",
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
        _compiled_fp4_attention()[(batch * heads,)](
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
            q_d=q.stride(3),
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
            block_l=self.block_l,
            block_s=block_s,
            causal=is_causal,
            logit_scale=channels**-0.5 if scale is None else scale,
            value_limit=torch.finfo(q.dtype).max,
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

    Supports ordinary :class:`Attention`, including GQA. Other cached
    families inspect storage geometry, use recurrent state, or consume the
    latent with float tensor operations, so they cannot use these packed rows.

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
        if is_cached_attention(module) and not isinstance(module, Attention):
            raise TypeError(
                f"Cached attention {type(module).__name__} is not supported by "
                "fp4_fused_layer_cache; supported family is Attention (including GQA).",
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
        result[key] = FP4PackedKVCache.alloc(
            batch=tuple(state.k.shape[:-3]),
            num_heads=state.k.shape[-3],
            max_seq=max_seq,
            channels_head=state.k.shape[-1],
            channels_v_head=state.v.shape[-1],
            device=state.k.device,
        )
    return result


@lru_cache(maxsize=1)
def _compiled_fp4_attention() -> triton.JITFunction[..., object]:
    return jit_kernel(_fp4_attention_forward_triton)


def _fp4_attention_forward_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
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
    q_d: int,
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
    block_l: language.constexpr,
    block_s: language.constexpr,
    causal: language.constexpr,
    logit_scale: language.constexpr,
    value_limit: language.constexpr,
) -> None:
    """One program per (batch, query head): stream keys, online softmax."""
    pid = language.program_id(0)
    b = pid // heads
    h = pid - b * heads
    s = language.arange(0, block_s)
    d = language.arange(0, channels)
    q = language.load(
        q_ptr + b * q_b + s[:, None] * q_s + h * q_h + d * q_d,
        mask=(s < seq)[:, None],
        other=0.0,
    )
    acc = language.zeros((block_s, channels), dtype=language.float32)
    m = language.full((block_s,), float("-inf"), dtype=language.float32)
    lse = language.zeros((block_s,), dtype=language.float32)
    lc = language.arange(0, block_l)
    for start in range(0, length, block_l):
        j = start + lc
        valid = j < length
        codes = language.load(
            k_ptr + b * k_b + h * k_h + j[:, None] * k_l + d[None, :] // 2,
            mask=valid[:, None],
            other=0,
        )
        # Joining half-width rows gave incorrect 16-wide bf16 attention on RTX5090.
        # Channel indices avoid that layout transition before the dot product.
        index = language.where(
            (d[None, :] & 1) == 0,
            codes.to(language.int32) >> 4,
            codes.to(language.int32) & 15,
        )
        key = language.load(table_ptr + index)
        scale = language.load(
            ks_ptr + b * ks_b + h * ks_h + j * ks_l,
            mask=valid,
            other=1.0,
        )
        key = language.clamp(
            key.to(language.float32) * scale[:, None],
            -value_limit,
            value_limit,
        ).to(language.bfloat16)
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
            v_ptr + b * v_b + h * v_h + j[:, None] * v_l + d[None, :] // 2,
            mask=valid[:, None],
            other=0,
        )
        vindex = language.where(
            (d[None, :] & 1) == 0,
            vcodes.to(language.int32) >> 4,
            vcodes.to(language.int32) & 15,
        )
        value = language.load(table_ptr + vindex)
        vscale = language.load(
            vs_ptr + b * vs_b + h * vs_h + j * vs_l,
            mask=valid,
            other=1.0,
        )
        value = language.clamp(
            value.to(language.float32) * vscale[:, None],
            -value_limit,
            value_limit,
        ).to(language.bfloat16)
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
