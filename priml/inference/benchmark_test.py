"""Tests for the inference benchmark."""

from __future__ import annotations

import torch
from priml.inference.benchmark import (
    FROZEN_PROMPTS,
    InferenceResult,
    benchmark,
    build_model_config,
    compute_kv_bytes,
    exp000,
)


def test_build_model_config_finalizes() -> None:
    """Config finalizes without error and has correct geometry."""
    cfg = build_model_config(
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_heads_kv=2,
    )
    assert cfg.channels_in == 128
    assert cfg.channels_out == 256
    assert cfg.num_layers == 2


def test_compute_kv_bytes_scales_linearly_with_context() -> None:
    """KV bytes double when context doubles."""
    cfg = build_model_config(hidden_size=128, num_layers=2, num_heads=4, num_heads_kv=2)
    bytes_100 = compute_kv_bytes(cfg, context_len=100, dtype=torch.bfloat16)
    bytes_200 = compute_kv_bytes(cfg, context_len=200, dtype=torch.bfloat16)
    assert bytes_200 == 2 * bytes_100


def test_compute_kv_bytes_is_positive() -> None:
    """KV bytes are strictly positive."""
    cfg = build_model_config(hidden_size=128, num_layers=2)
    kv = compute_kv_bytes(cfg, context_len=64, dtype=torch.bfloat16)
    assert kv > 0


def test_compute_kv_bytes_scales_with_layers() -> None:
    """More layers means more KV bytes."""
    cfg_2 = build_model_config(hidden_size=128, num_layers=2)
    cfg_4 = build_model_config(hidden_size=128, num_layers=4)
    kv_2 = compute_kv_bytes(cfg_2, context_len=64, dtype=torch.bfloat16)
    kv_4 = compute_kv_bytes(cfg_4, context_len=64, dtype=torch.bfloat16)
    assert kv_4 == 2 * kv_2


def test_compute_kv_bytes_scales_with_batch() -> None:
    """Larger batch means more KV bytes."""
    cfg = build_model_config(hidden_size=128, num_layers=2)
    kv_1 = compute_kv_bytes(cfg, context_len=64, batch_size=1, dtype=torch.bfloat16)
    kv_4 = compute_kv_bytes(cfg, context_len=64, batch_size=4, dtype=torch.bfloat16)
    assert kv_4 == 4 * kv_1


def test_benchmark_returns_valid_result() -> None:
    """Benchmark returns a result with positive metrics."""
    cfg = build_model_config(hidden_size=64, num_layers=1, num_heads=2, num_heads_kv=1)
    prompts = torch.randint(0, 256, (2, 16), dtype=torch.long)
    result = benchmark(
        cfg,
        prompt_token_ids=prompts,
        max_new_tokens=4,
        dtype=torch.bfloat16,
        seed=42,
        device="cpu",
    )
    assert isinstance(result, InferenceResult)
    assert result.total_tokens > 0
    assert result.total_time_s > 0
    assert result.tokens_per_second > 0
    assert result.kv_bytes_per_token > 0
    assert result.kv_bytes_total > 0


def test_benchmark_is_deterministic() -> None:
    """Same seed produces same token count."""
    cfg = build_model_config(hidden_size=64, num_layers=1, num_heads=2, num_heads_kv=1)
    prompts = torch.randint(0, 256, (2, 16), dtype=torch.long)
    r1 = benchmark(
        cfg, prompt_token_ids=prompts, max_new_tokens=4, seed=42, device="cpu"
    )
    r2 = benchmark(
        cfg, prompt_token_ids=prompts, max_new_tokens=4, seed=42, device="cpu"
    )
    assert r1.total_tokens == r2.total_tokens


def test_exp000_runs() -> None:
    """exp000 runs and returns valid results."""
    result = exp000()
    assert isinstance(result, InferenceResult)
    assert result.tokens_per_second > 0
    assert result.kv_bytes_per_token > 0
    assert result.kv_bytes_total > result.kv_bytes_per_token


def test_frozen_prompts_are_fixed() -> None:
    """Frozen prompts never change."""
    assert len(FROZEN_PROMPTS) == 16
    assert FROZEN_PROMPTS[0] == "The quick brown fox jumps over the lazy dog."


def test_kv_bytes_reduced_by_gqa() -> None:
    """GQA (fewer KV heads) reduces KV bytes."""
    cfg_mha = build_model_config(
        hidden_size=128, num_layers=2, num_heads=4, num_heads_kv=4
    )
    cfg_gqa = build_model_config(
        hidden_size=128, num_layers=2, num_heads=4, num_heads_kv=2
    )
    kv_mha = compute_kv_bytes(cfg_mha, context_len=64, dtype=torch.bfloat16)
    kv_gqa = compute_kv_bytes(cfg_gqa, context_len=64, dtype=torch.bfloat16)
    assert kv_gqa < kv_mha
