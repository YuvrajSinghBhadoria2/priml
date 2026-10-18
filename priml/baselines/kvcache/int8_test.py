"""Tests for the int8 KV cache exp001 injects into the decode chain."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.kvcache.decode import _cache_bytes, decoder
from priml.baselines.kvcache.decode_test import (
    CHANNELS_HEAD,
    NUM_DECODE_STEPS,
    NUM_HEADS_KV,
    NUM_LAYERS,
    PROMPT_LEN,
    VOCAB,
    _prompt,
    _tiny,
)
from priml.baselines.kvcache.int8 import (
    SCALE_COLUMNS,
    Int8KVCache,
    int8_layer_cache,
)
from priml.model.attention.kvcache import KVCache, alloc_layer_cache


_NUM_HEADS: Final = 3
_CODE_MAX: Final = 127.0


def _cache(*, batch: int = 2, max_seq: int = 8) -> Int8KVCache:
    """Build a tiny int8 cache directly from the documented layout."""
    return Int8KVCache(
        k=torch.zeros(
            (batch, _NUM_HEADS, max_seq, CHANNELS_HEAD + SCALE_COLUMNS),
            dtype=torch.int8,
        ),
        v=torch.zeros(
            (batch, _NUM_HEADS, max_seq, CHANNELS_HEAD + SCALE_COLUMNS),
            dtype=torch.int8,
        ),
        length=0,
    )


def test_int8_cache_counts_its_scales_inside_the_cache_tensors() -> None:
    model = decoder().make()
    max_seq = 64
    default = alloc_layer_cache(
        model,
        batch=2,
        max_seq=max_seq,
        dtype=torch.float32,
    )
    packed = int8_layer_cache(
        model,
        batch=2,
        max_seq=max_seq,
        dtype=torch.float32,
    )
    float_bytes = _cache_bytes(default)
    int8_bytes = _cache_bytes(packed)

    first = next(iter(default.values()))
    assert isinstance(first, KVCache)
    channels = first.k.shape[-1]
    rows = len(default) * first.k.shape[0] * first.k.shape[1] * max_seq
    # exp000's storage at the model's dtype, spelled out so the identity
    # below does not depend on the model's dtype: two values, two bytes.
    bf16_bytes = rows * 2 * channels * torch.bfloat16.itemsize
    # Codes replace the two-byte halves; the float32 scales ride in-row and
    # are counted too, which is what keeps this from being half exactly.
    assert int8_bytes == bf16_bytes // 2 + rows * 2 * SCALE_COLUMNS
    assert int8_bytes < bf16_bytes
    assert int8_bytes < float_bytes


def test_update_stays_within_half_a_quantization_step() -> None:
    cache = _cache()
    k = torch.randn(2, _NUM_HEADS, 5, CHANNELS_HEAD) * 3.0
    v = torch.randn(2, _NUM_HEADS, 5, CHANNELS_HEAD) * 0.5

    k_all, v_all = cache.update(k, v)

    scale_k = k.abs().amax(dim=-1, keepdim=True) / _CODE_MAX
    scale_v = v.abs().amax(dim=-1, keepdim=True) / _CODE_MAX
    # Rounding costs at most half a step; the 0.51 leaves headroom for the
    # float32 multiply that reads the codes back.
    assert torch.all((k_all - k).abs() <= scale_k * 0.51)
    assert torch.all((v_all - v).abs() <= scale_v * 0.51)
    assert k_all.shape == k.shape
    assert cache.length == 5
    assert cache.seen == 5


def test_update_answers_in_the_incoming_dtype() -> None:
    cache = _cache()
    k = torch.randn(2, _NUM_HEADS, 3, CHANNELS_HEAD, dtype=torch.bfloat16)
    v = torch.randn(2, _NUM_HEADS, 3, CHANNELS_HEAD, dtype=torch.bfloat16)

    k_all, v_all = cache.update(k, v)

    assert k_all.dtype is torch.bfloat16
    assert v_all.dtype is torch.bfloat16


def test_fifo_shift_keeps_codes_and_scales_aligned() -> None:
    cache = _cache(max_seq=4)
    first = torch.randn(2, _NUM_HEADS, 4, CHANNELS_HEAD)
    cache.update(first, first)
    more = torch.randn(2, _NUM_HEADS, 2, CHANNELS_HEAD)

    k_all, _ = cache.update(more, more)

    expected = torch.cat((first[..., 2:, :], more), dim=-2)
    scale = expected.abs().amax(dim=-1, keepdim=True) / _CODE_MAX
    assert torch.all((k_all - expected).abs() <= scale * 0.51)
    assert cache.length == 4
    assert cache.seen == 6


def test_progress_matches_the_float_cache_on_the_same_writes() -> None:
    control = KVCache.alloc(
        batch=2,
        num_heads=_NUM_HEADS,
        max_seq=4,
        channels_head=CHANNELS_HEAD,
        dtype=torch.float32,
    )
    cache = _cache(max_seq=4)
    writes = [
        torch.randn(2, _NUM_HEADS, 3, CHANNELS_HEAD),
        torch.randn(2, _NUM_HEADS, 3, CHANNELS_HEAD),
        torch.randn(2, _NUM_HEADS, 2, CHANNELS_HEAD),
    ]
    for incoming in writes:
        control.update(incoming, incoming)
        cache.update(incoming, incoming)

    assert cache.length == control.length
    assert cache.seen == control.seen
    assert cache.max_seq == control.max_seq


def test_a_zero_row_dequantizes_to_zero() -> None:
    cache = _cache()
    zeros = torch.zeros(2, _NUM_HEADS, 2, CHANNELS_HEAD)

    k_all, v_all = cache.update(zeros, zeros)

    assert torch.isfinite(k_all).all()
    assert torch.isfinite(v_all).all()
    assert torch.equal(k_all, zeros)
    assert torch.equal(v_all, zeros)


def test_int8_layer_cache_widens_every_slot_the_model_allocates() -> None:
    model = decoder().make()
    default = alloc_layer_cache(
        model,
        batch=2,
        max_seq=16,
        dtype=torch.float32,
    )
    packed = int8_layer_cache(
        model,
        batch=2,
        max_seq=16,
        dtype=torch.float32,
    )

    assert packed.keys() == default.keys()
    for key, state in packed.items():
        assert isinstance(state, Int8KVCache)
        base = default[key]
        assert isinstance(base, KVCache)
        assert state.k.dtype is torch.int8
        assert state.v.dtype is torch.int8
        assert state.k.shape == (
            *base.k.shape[:-1],
            base.k.shape[-1] + SCALE_COLUMNS,
        )
        assert state.v.shape == (
            *base.v.shape[:-1],
            base.v.shape[-1] + SCALE_COLUMNS,
        )
        assert state.length == 0
        assert state.seen == 0


def test_update_refuses_a_width_the_cache_cannot_hold() -> None:
    cache = _cache()
    too_wide = torch.zeros(
        2,
        _NUM_HEADS,
        2,
        CHANNELS_HEAD + SCALE_COLUMNS + 1,
    )

    with pytest.raises(ValueError, match="Cannot write"):
        cache.update(too_wide, too_wide)


def test_freeze_is_refused_because_the_frozen_view_serves_raw_codes() -> None:
    with pytest.raises(NotImplementedError, match="freeze"):
        _cache().freeze()


def test_exp001_decodes_on_cpu_with_a_smaller_cache() -> None:
    cfg = _tiny()
    cfg.allocate_cache = int8_layer_cache
    first = cfg.make().measure()
    again = cfg.make().measure()
    control = _tiny().make().measure()

    assert torch.equal(first.tokens, again.tokens)
    assert first.tokens.shape == (len(first.tokens), PROMPT_LEN + NUM_DECODE_STEPS + 1)
    assert int(first.tokens.min()) >= 0
    assert int(first.tokens.max()) < VOCAB
    assert first.kv_bytes < control.kv_bytes
    assert first.kv_bytes_per_token == (
        NUM_LAYERS * 2 * NUM_HEADS_KV * (CHANNELS_HEAD + SCALE_COLUMNS)
    )


def test_the_prompt_the_end_to_end_run_feeds_is_still_exp000s() -> None:
    cfg = _tiny()
    cfg.allocate_cache = int8_layer_cache
    report = cfg.make().measure()

    assert torch.equal(report.tokens[:, :PROMPT_LEN], _prompt())
