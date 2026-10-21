"""Tests for the KV-cache decode experiment configs.

``exp000`` is pinned whole by a pprint golden: its prompts, context length,
model, and dtype are the control every KV-cache fork is measured against.
Config comparisons pin each fork's cache allocator and, for exp003,
the attention kernel that reads the packed cache.
"""

from __future__ import annotations

from pathlib import Path

import os
import subprocess
import sys

from priml.baselines.kvcache.experiments import exp000, exp001, exp002, exp003
from priml.baselines.kvcache.fp4 import fp4_layer_cache
from priml.baselines.kvcache.fp4_fused import Fp4TritonAttention, fp4_fused_layer_cache
from priml.baselines.kvcache.int8 import int8_layer_cache
from priml.model.attention.attention import Attention
from priml.model.transformer.block import TransformerBlock
from priml.testing.golden import assert_pprint_golden


def test_exp000_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp000_decodes_to_a_4096_token_context() -> None:
    cfg = exp000()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp001_decodes_to_a_4096_token_context() -> None:
    cfg = exp001()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp001_changes_only_the_cache_allocator() -> None:
    forked = exp001()
    control = exp000()
    assert forked.allocate_cache is int8_layer_cache
    forked.allocate_cache = control.allocate_cache
    assert forked.udiff(control) == ""


def test_exp002_decodes_to_a_4096_token_context() -> None:
    cfg = exp002()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp002_changes_only_the_cache_allocator() -> None:
    forked = exp002()
    control = exp000()
    assert forked.allocate_cache is fp4_layer_cache
    forked.allocate_cache = control.allocate_cache
    assert forked.udiff(control) == ""


def test_experiments_import_without_triton(tmp_path: Path) -> None:
    probe = tmp_path / "import_without_triton.py"
    probe.write_text(
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        "class BlockTriton(MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'triton' or fullname.startswith('triton.'):\n"
        "            raise ModuleNotFoundError('Triton import blocked')\n"
        "sys.meta_path.insert(0, BlockTriton())\n"
        "from priml.baselines.kvcache.experiments import exp000, exp001, exp002, exp003\n"
        "for factory in (exp000, exp001, exp002, exp003):\n"
        "    factory().copy_tree().finalize()\n"
        "assert 'triton' not in sys.modules\n",
    )
    result = subprocess.run(
        [sys.executable, "-B", "import_without_triton.py"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_exp003_decodes_to_a_4096_token_context() -> None:
    cfg = exp003()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


def test_exp003_changes_only_the_cache_and_the_kernel_reading_it() -> None:
    forked = exp003().copy_tree().finalize()
    control = exp000().copy_tree().finalize()
    assert forked.allocate_cache is fp4_fused_layer_cache
    assert isinstance(forked.model.block, list)
    assert isinstance(control.model.block, list)
    for forked_block in forked.model.block:
        assert isinstance(forked_block, TransformerBlock.Config)
        assert isinstance(forked_block.attn, Attention.Config)
        assert isinstance(forked_block.attn.attn_kernel, Fp4TritonAttention.Config)
    forked.allocate_cache = control.allocate_cache
    for forked_block, control_block in zip(
        forked.model.block,
        control.model.block,
        strict=True,
    ):
        assert isinstance(forked_block, TransformerBlock.Config)
        assert isinstance(control_block, TransformerBlock.Config)
        assert isinstance(forked_block.attn, Attention.Config)
        assert isinstance(control_block.attn, Attention.Config)
        forked_block.attn.attn_kernel = control_block.attn.attn_kernel
    assert forked.udiff(control) == ""


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
