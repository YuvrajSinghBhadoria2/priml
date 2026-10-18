"""Tests for the KV-cache decode experiment configs.

``exp000`` is pinned whole by a pprint golden: its prompts, context length,
model, and dtype are the control every KV-cache fork is measured against.
``exp001`` is pinned the same way, plus a render-level check that its only
departure from ``exp000`` is the cache allocator.
"""

from __future__ import annotations

from priml.baselines.kvcache.experiments import exp000, exp001
from priml.testing.golden import assert_pprint_golden


def test_exp000_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp000_decodes_to_a_4096_token_context() -> None:
    cfg = exp000()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp001_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp001", config=exp001())


def test_exp001_decodes_to_a_4096_token_context() -> None:
    cfg = exp001()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp001_changes_only_the_cache_allocator() -> None:
    forked = (
        exp001()
        .pformat(
            finalize=True,
            mask_memory_addresses=True,
            hide_default_values=False,
            width=80,
        )
        .replace(
            "priml.baselines.kvcache.int8.int8_layer_cache",
            "priml.model.attention.kvcache.alloc_layer_cache",
        )
    )
    control = exp000().pformat(
        finalize=True,
        mask_memory_addresses=True,
        hide_default_values=False,
        width=80,
    )
    assert forked == control


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
