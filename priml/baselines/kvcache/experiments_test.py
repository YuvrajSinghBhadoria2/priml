"""Tests for the KV-cache decode experiment configs.

``exp000`` is pinned whole by a pprint golden: its prompts, context length,
model, and dtype are the control every KV-cache fork is measured against.
``exp001`` and ``exp002`` are pinned the same way. A config comparison checks
that only their cache allocator differs from ``exp000``.
"""

from __future__ import annotations

from priml.baselines.kvcache.experiments import exp000, exp001, exp002
from priml.baselines.kvcache.fp4 import fp4_layer_cache
from priml.baselines.kvcache.int8 import int8_layer_cache
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
    forked = exp001()
    control = exp000()
    assert forked.allocate_cache is int8_layer_cache
    forked.allocate_cache = control.allocate_cache
    assert forked.udiff(control) == ""


def test_exp002_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp002", config=exp002())


def test_exp002_decodes_to_a_4096_token_context() -> None:
    cfg = exp002()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp002_changes_only_the_cache_allocator() -> None:
    forked = exp002()
    control = exp000()
    assert forked.allocate_cache is fp4_layer_cache
    forked.allocate_cache = control.allocate_cache
    assert forked.udiff(control) == ""


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
