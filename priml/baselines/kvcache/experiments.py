"""KV-cache decode experiments.

``exp000`` times greedy decode of a byte-level GQA decoder over a cache that
grows to a 4096-token context, with the cache stored at the model's own
dtype. Every later experiment forks a named parent and changes ONE thing:
``allocate_cache``, the storage the cache uses.

    exp000  bf16 weights, bf16 KV cache, 4096-token context
      +-- exp001  int8 codes, one float32 scale per token, packed in k/v

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.kvcache.experiments.exp000

Objective:
  Raise decode tokens per second, or shrink cache bytes at equal tokens per
  second, by changing how the cache stores keys and values.

Score:
  Decode tokens per second over every timed step, prefill excluded, with the
  p99 step beside it: a scheme that stalls some steps shows there first.

A fork may change:
  ``allocate_cache``: the cache's dtype, layout, quantization, and the
  arithmetic that reads it back.

A fork may not change:
  The model, its weights' dtype, the prompts, the context length, or the
  harness and its timing.
"""

from __future__ import annotations

from priml.baselines.kvcache.decode import DecodeBenchmark
from priml.baselines.kvcache.int8 import int8_layer_cache


def exp000() -> DecodeBenchmark.Config:
    """Time bf16 greedy decode to a 4096-token context on a CUDA device.

    The control every KV-cache experiment forks. Frozen: improvements belong
    in a fork, never in an edit here.

    Hypothesis:
      At a 4096-token context a decode step of this model reads more cache
      than weights, so decode tokens per second is bound by cache bytes, and
      a cache stored in fewer bytes per value decodes faster at equal model
      quality.

    References:
      https://arxiv.org/abs/2305.13245
        Ainslie et al. 2023. GQA: Training generalized multi-query
        transformer models from multi-head checkpoints.
      https://arxiv.org/abs/2402.02750
        Liu et al. 2024. KIVI: A tuning-free asymmetric 2bit quantization for
        KV cache.

    Results:
      TBD.

    Returns:
      config: The frozen decode benchmark.

    """
    cfg = DecodeBenchmark.Config()
    cfg.prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that",
        "The mitochondria is the powerhouse of the cell",
        "The speed of light is approximately 299792458 meters per second",
        "Photosynthesis is the process by which plants convert",
        "Quantum entanglement is a phenomenon where two particles",
        "To be or not to be, that is the question",
        "Water boils at 100 degrees Celsius at sea level",
    ]
    cfg.num_decode_steps = 4096 - cfg.prompt_len
    cfg.device = "cuda"
    return cfg


def exp001() -> DecodeBenchmark.Config:
    """Store int8 KV rows in the frozen exp000 decode benchmark.

    ``allocate_cache`` becomes :func:`int8_layer_cache`: each token's key
    row and value row quantize symmetrically to signed int8 against one
    scale, and the float32 scale's bits pack into the row's trailing
    columns, so measured cache bytes count codes and scales together.
    Attention reads the dequantized slice in the model's dtype, so the
    model, its dtype, the prompts, the context length, and the harness are
    exp000's.

    Hypothesis:
      At a 4096-token context a decode step reads more cache than weights,
      so decode tokens per second scale with the bytes each cached value
      occupies. int8 codes with one scale per token move 68/128 of exp000's
      bf16 bytes per value, so decode runs faster while the greedy tokens
      stay within the quantization error of the control.

    References:
      https://arxiv.org/abs/2402.02750
        Liu et al. 2024. KIVI: A tuning-free asymmetric 2bit quantization
        for KV cache. KIVI's per-token value granularity motivates this
        fork's scale; KIVI quantizes keys per channel at 2 bits, while this
        fork keeps both halves per-token at 8 bits as the first rung.

    Results:
      TBD.

    Returns:
      config: exp000 with the int8 cache allocator.

    """
    cfg = exp000()
    cfg.allocate_cache = int8_layer_cache
    return cfg
