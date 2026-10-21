"""Tests for exp003's fused fp4 attention kernel and its packed cache."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.kvcache.decode import _cache_bytes
from priml.baselines.kvcache.decode_test import _tiny
from priml.baselines.kvcache.experiments import exp003
from priml.baselines.kvcache.fp4 import (
    SCALE_COLUMNS,
    _dequantized,
    _packed,
    _quantize,
    fp4_layer_cache,
)
from priml.baselines.kvcache.fp4_fused import (
    FP4PackedKVCache,
    Fp4TritonAttention,
    fp4_fused_layer_cache,
)


_NUM_HEADS: Final = 3
_CHANNELS_HEAD: Final = 4


def _cuda_case(
    *,
    batch: int = 2,
    seq: int = 1,
    heads: int = 2,
    channels: int = 64,
    length: int = 300,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bf16 queries over randomly packed keys and values on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(batch, seq, heads, channels, device=device) * 0.5
    q = q.to(torch.bfloat16)
    k = _quantize(
        (torch.randn(batch, length, heads, channels, device=device) * 2).to(
            torch.bfloat16,
        ),
    )
    v = _quantize(
        (torch.randn(batch, length, heads, channels, device=device) * 2).to(
            torch.bfloat16,
        ),
    )
    return q, k, v


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Attend with SDPA over the same rows dequantized out of band."""
    _, length, _, width = k.shape
    channels = 2 * (width - SCALE_COLUMNS)
    key = _dequantized(
        k,
        end=length,
        channels=channels,
        dtype=torch.bfloat16,
    )
    value = _dequantized(
        v,
        end=length,
        channels=channels,
        dtype=torch.bfloat16,
    )
    key = key.movedim(-3, -2)
    value = value.movedim(-3, -2)
    attn_mask = None
    if causal:
        # Bottom-right causal: query row i (at position length - seq + i)
        # attends keys up to its own position; SDPA's is_causal is top-left
        # and disagrees whenever seq < length.
        seq = q.shape[1]
        rows = torch.arange(length - seq, length, device=q.device)
        attn_mask = torch.arange(length, device=q.device)[None, :] <= rows[:, None]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.movedim(-3, -2),
        key,
        value,
        attn_mask=attn_mask,
    )
    return out.movedim(-3, -2)


def test_packed_cache_serves_uint8_rows_in_the_incoming_dtype() -> None:
    cache = FP4PackedKVCache.alloc(
        batch=(2,),
        num_heads=_NUM_HEADS,
        max_seq=5,
        channels_head=_CHANNELS_HEAD,
        channels_v_head=_CHANNELS_HEAD,
        device=torch.device("cpu"),
    )
    k = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD)
    v = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD)

    k_all, v_all = cache.update(k, v)

    width = _packed(_CHANNELS_HEAD) + SCALE_COLUMNS
    assert k_all.dtype == torch.uint8
    assert v_all.dtype == torch.uint8
    assert k_all.shape == (2, _NUM_HEADS, 5, width)
    # The served rows are the storage itself, not a fresh dequantized copy.
    assert k_all.data_ptr() == cache.k[..., :5, :].data_ptr()


def test_fused_allocator_counts_the_same_bytes_as_the_fp4_allocator() -> None:
    model = _tiny().make().model
    max_seq = 64
    reference = fp4_layer_cache(model, batch=2, max_seq=max_seq, dtype=torch.float32)
    fused = fp4_fused_layer_cache(model, batch=2, max_seq=max_seq, dtype=torch.float32)
    assert set(fused) == set(reference)
    for state in fused.values():
        assert isinstance(state, FP4PackedKVCache)
    assert _cache_bytes(fused) == _cache_bytes(reference)


def test_forward_refuses_features_the_decode_ladder_never_asks_for() -> None:
    attention = Fp4TritonAttention(Fp4TritonAttention.Config(channels_head=8))
    q = torch.zeros(1, 1, 2, 8, dtype=torch.bfloat16)
    packed = torch.zeros(1, 8, 2, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="no dropout, mask, or window"):
        attention(q, packed, packed, dropout_p=0.1)
    with pytest.raises(ValueError, match="bf16"):
        attention(q.float(), packed, packed)
    with pytest.raises(ValueError, match="even"):
        Fp4TritonAttention(Fp4TritonAttention.Config(channels_head=3))


@pytest.mark.gpu_triton
@pytest.mark.parametrize(
    ("seq", "causal"),
    [(1, False), (1, True), (4, True)],
    ids=["decode", "decode-causal", "prefill"],
)
def test_kernel_matches_sdpa_over_dequantized_rows(seq: int, causal: bool) -> None:
    q, k, v = _cuda_case(seq=seq)
    attention = Fp4TritonAttention(Fp4TritonAttention.Config(channels_head=64))

    out = attention(q, k, v, is_causal=causal)
    expected = _reference(q, k, v, causal=causal)

    assert (out - expected).abs().max().item() < 0.02


@pytest.mark.gpu_triton
def test_exp003_decodes_with_the_fused_cache() -> None:
    # The real ladder config, shortened: the tiny model's 2-wide heads sit
    # under triton's K >= 16 dot floor.
    cfg = exp003()
    cfg.device = "cuda"
    cfg.num_warmup = 0
    cfg.num_repeats = 1
    cfg.num_decode_steps = 8

    report = cfg.make().measure()

    assert report.tokens.shape == (len(report.tokens), cfg.prompt_len + 9)
    assert report.kv_bytes_per_token == 1152


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
