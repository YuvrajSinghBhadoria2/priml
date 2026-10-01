"""Tests for the analytical KV-cache decode cost.

CPU only and pure arithmetic -- nothing here allocates a tensor or touches a
device, so the whole file runs in well under a millisecond and the CI GPU
question does not arise.

Shapes are the smallest that keep every axis distinct: no dimension is 1, and
dims that meet differ. A 1 broadcasts and hides a transposed axis, and a tie
between two dims hides a swapped one.
"""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.cost import peak
from priml.inference.kv_cache import KVCache


NUM_LAYERS: Final = 2
NUM_HEADS: Final = 3
NUM_HEADS_KV: Final = 2
CHANNELS_HEAD: Final = 4
ITEMSIZE_BF16: Final = 2
ITEMSIZE_INT8: Final = 1


def mha_cache(
    *,
    num_heads_kv: int = NUM_HEADS,
    weight_bytes: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> KVCache:
    """Build a plain multi-head cache: every query head has its own KV head."""
    return KVCache(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_heads_kv=num_heads_kv,
        channels_head=CHANNELS_HEAD,
        dtype=dtype,
        weight_bytes=weight_bytes,
    )


def test_bytes_per_token_counts_keys_and_values_across_layers() -> None:
    cache = mha_cache()
    assert (
        cache.bytes_per_token()
        == 2 * NUM_LAYERS * NUM_HEADS * CHANNELS_HEAD * ITEMSIZE_BF16
    )


def test_grouped_query_shrinks_the_cache_by_exactly_the_grouping_ratio() -> None:
    dense = mha_cache()
    grouped = mha_cache(num_heads_kv=NUM_HEADS_KV)
    assert (
        grouped.bytes_per_token() * NUM_HEADS == dense.bytes_per_token() * NUM_HEADS_KV
    )
    assert grouped.grouping_ratio == pytest.approx(NUM_HEADS / NUM_HEADS_KV)
    assert dense.grouping_ratio == 1.0


def test_finalize_mirrors_the_query_head_count_onto_the_kv_count() -> None:
    cache = KVCache(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        channels_head=CHANNELS_HEAD,
    )
    assert cache.finalize().num_heads_kv == NUM_HEADS


def test_finalize_leaves_an_explicit_kv_count_alone() -> None:
    # An explicitly grouped cache must not be widened back by finalize.
    cache = KVCache(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_heads_kv=NUM_HEADS_KV,
        channels_head=CHANNELS_HEAD,
    )
    assert cache.finalize().num_heads_kv == NUM_HEADS_KV


def test_a_narrower_dtype_halves_the_cache() -> None:
    wide = mha_cache()
    narrow = mha_cache(dtype=torch.int8)
    assert (
        narrow.bytes_per_token() * ITEMSIZE_BF16
        == wide.bytes_per_token() * ITEMSIZE_INT8
    )


def test_tokens_in_is_the_floor_of_the_budget_over_the_token_cost() -> None:
    cache = mha_cache()
    per_token = cache.bytes_per_token()
    assert cache.tokens_in(per_token * 5) == 5
    assert cache.tokens_in(per_token * 5 - 1) == 4


def test_tokens_in_reports_zero_rather_than_a_negative_budget() -> None:
    assert mha_cache().tokens_in(-1) == 0


def test_tokens_in_is_zero_for_an_unspecified_geometry() -> None:
    # Reporting 0 tells a caller sizing a budget that the shape is missing;
    # raising would only make them catch an exception to learn the same thing.
    assert KVCache().tokens_in(1 << 20) == 0


def test_kv_intensity_is_one_over_itemsize_at_every_context_length() -> None:
    # The load-bearing identity: one MAC per cached element, and the K/V factor
    # cancels against the MAC factor, so intensity is 1 / itemsize regardless
    # of context. Asserted across lengths so a drift cannot hide at one.
    cache = mha_cache()
    for context_len in (2, 8, 32, 128):
        cost = cache.decode_cost(batch_size=2, context_len=context_len)
        assert cost.intensity == pytest.approx(1 / ITEMSIZE_BF16)


def test_decode_is_memory_bound_at_every_context_length_on_every_device() -> None:
    # Decode never becomes compute-bound. Both terms sit far below every ridge
    # in the table, so there is no context at which the verdict flips.
    cache = mha_cache(weight_bytes=1 << 20)
    for device in ("a100", "h100", "b200", "rtx5090"):
        for context_len in (2, 64, 4096):
            cost = cache.decode_cost(
                batch_size=2,
                context_len=context_len,
                device=device,
            )
            assert cost.memory_bound, f"{device} at {context_len}"
            assert cost.fraction_of_ridge < 0.01


def test_traffic_scales_with_context_and_with_batch() -> None:
    # A weight read is paid once, so it is the context and batch terms that must
    # scale exactly. The default geometry reads no weights, which makes the
    # doubling exact rather than approximate.
    cache = mha_cache()
    base = cache.decode_cost(batch_size=2, context_len=8)
    longer = cache.decode_cost(batch_size=2, context_len=16)
    wider = cache.decode_cost(batch_size=4, context_len=8)
    assert longer.kv_bytes == base.kv_bytes * 2
    assert wider.kv_bytes == base.kv_bytes * 2
    assert longer.flops == base.flops * 2
    assert base.weight_bytes == 0
    assert base.weight_flops == 0


def test_the_weight_read_is_paid_once_per_step_whatever_the_context() -> None:
    # A small weight read so the cache overtakes it inside the tested range;
    # the geometry makes the cache 192 bytes per context token at batch 2, so
    # 1 KiB of weights is crossed almost immediately.
    weights = 1 << 10
    cache = mha_cache(weight_bytes=weights)
    short = cache.decode_cost(batch_size=2, context_len=8)
    longer = cache.decode_cost(batch_size=2, context_len=4096)
    # The weight read is a constant, so its share of the traffic -- and so its
    # pull on the intensity -- shrinks as the cache grows. The intensity falls
    # monotonically toward the cache-only limit rather than ever crossing it.
    assert short.weight_bytes == longer.weight_bytes == weights
    assert short.weight_flops == longer.weight_flops
    assert short.intensity > longer.intensity > 1 / ITEMSIZE_BF16
    cache_only = mha_cache().decode_cost(batch_size=2, context_len=4096)
    assert longer.intensity == pytest.approx(cache_only.intensity, rel=0.01)
    # And it is the same limit from either side: doubling the weights at a
    # context that already dominates them barely moves the intensity.
    heavier = mha_cache(weight_bytes=2 * weights).decode_cost(
        batch_size=2,
        context_len=4096,
    )
    assert heavier.intensity == pytest.approx(cache_only.intensity, rel=0.01)


def test_the_ridge_is_read_from_priml_cost_and_not_restated() -> None:
    # If this ever diverges from priml.cost the module is lying about the
    # device, and every memory-bound verdict in it becomes wrong.
    cost = mha_cache().decode_cost(batch_size=2, context_len=8, device="h100")
    assert cost.ridge == pytest.approx(
        peak()["h100"][torch.bfloat16, "intensity", "matmul"],
    )


def test_a_form_qualified_device_name_moves_the_ridge() -> None:
    # Ties the module to the form factors in the cost table: naming the board
    # selects the ridge, and SXM and PCIe genuinely disagree.
    sxm = mha_cache().decode_cost(batch_size=2, context_len=8, device="h100")
    pcie = mha_cache().decode_cost(batch_size=2, context_len=8, device="h100-pcie")
    assert pcie.ridge > sxm.ridge
    assert sxm.memory_bound
    assert pcie.memory_bound


def test_the_ridge_follows_the_cache_dtype_not_the_compute_dtype() -> None:
    # A low-bit cache experiment changes which ceiling it is measured against,
    # so asking for the ridge at another dtype has to actually do something.
    narrow = mha_cache(dtype=torch.float8_e4m3fn).decode_cost(
        batch_size=2,
        context_len=8,
        device="b200",
        dtype=torch.float8_e4m3fn,
    )
    wide = mha_cache(dtype=torch.bfloat16).decode_cost(
        batch_size=2,
        context_len=8,
        device="b200",
        dtype=torch.bfloat16,
    )
    assert narrow.ridge == pytest.approx(2 * wide.ridge)


def test_pprint_carries_the_geometry_so_a_run_is_reproducible_from_its_config() -> None:
    # A field that does not print cannot be overridden, forked, or diffed, and
    # the run stops being reproducible from the config that produced it.
    # Defaults are hidden by default, so the full tree is what has to carry
    # every field -- and a fork that SETS weight_bytes must surface it.
    printed = mha_cache().finalize().pformat(hide_default_values=False)
    for field in (
        "num_layers",
        "num_heads_kv",
        "channels_head",
        "weight_bytes",
        "dtype",
    ):
        assert field in printed
    forked = mha_cache(weight_bytes=1 << 10).finalize().pformat()
    assert "weight_bytes" in forked


def test_finalize_propagates_through_the_real_configgle_chain() -> None:
    # finalize() here delegates to super(), so what matters is that the
    # inference runs on the real Fig rather than only under a stand-in: the
    # sentinel must be resolved by the time the config is printed.
    printed = KVCache(num_layers=2, num_heads=3, channels_head=4).finalize().pformat()
    assert "num_heads_kv=3" in printed
