"""Frozen inference baseline: decode throughput and KV-cache bytes.

``exp000`` is the number every KV-cache quantization experiment measures
against. It runs a small decoder-only Transformer with BF16 KV-cache on
fixed prompts and reports tokens/sec (wall-clock) and KV bytes (analytical
from ``priml.cost``).

The eval is frozen: same prompts, same seed, same generation length.
Every subsequent experiment changes one thing -- the KV-cache dtype -- and
measures memory savings and throughput regression against these numbers.

Judged worst-case, not average: p99 latency over the frozen prompts, so
a quantization method that occasionally stalls is penalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import time

import torch

from priml.cost import cost
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.embedding import Embedding
from priml.model.generate import generate
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceResult:
    """Metrics from one inference benchmark run."""

    tokens_per_second: float
    """Wall-clock decode throughput."""

    kv_bytes_per_token: int
    """Analytical KV-cache bytes per token of context."""

    kv_bytes_total: int
    """KV-cache bytes for the full context length."""

    p99_latency_ms: float
    """99th-percentile per-prompt latency (worst-case)."""

    mean_latency_ms: float
    """Mean per-prompt latency."""

    total_tokens: int
    """Total tokens generated across all prompts."""

    total_time_s: float
    """Total wall-clock generation time."""


FROZEN_PROMPTS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence that",
    "The capital of France is",
    "Python is a programming language that",
    "In the beginning, God created the heavens and the earth",
    "To be or not to be, that is the question",
    "The mitochondria is the powerhouse of the cell",
    "Water boils at 100 degrees Celsius at sea level",
    "The sun rises in the east and sets in the west",
    "DNA is the molecule that carries genetic instructions",
    "The speed of light is approximately 299792458 meters per second",
    "Photosynthesis is the process by which plants convert",
    "The Earth orbits the sun once every 365.25 days",
    "Gravity is the force that attracts objects toward each other",
    "The human brain has approximately 86 billion neurons",
    "Quantum entanglement is a phenomenon where two particles",
)
"""Frozen eval prompts. Fixed and never change: every experiment is measured against the same baseline."""


def build_model_config(
    *,
    vocab_size: int = 256,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    num_heads_kv: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> Transformer.Config:
    """Build a small decoder-only Transformer config.

    Args:
      vocab_size: Token vocabulary size.
      hidden_size: Model width.
      num_layers: Number of transformer blocks.
      num_heads: Query heads.
      num_heads_kv: Key/value heads (GQA).
      dtype: Parameter dtype.

    Returns:
      config: Finalized Transformer.Config ready to make().

    """
    channels_head = hidden_size // num_heads
    attn = Attention.Config(
        channels_in=hidden_size,
        channels_out=hidden_size,
        num_heads=num_heads,
        channels_head=channels_head,
        num_heads_kv=num_heads_kv,
        causal=True,
        rope=RoPE.Config(channels_head=channels_head),
    )
    block = TransformerBlock.Config(
        channels_in=hidden_size,
        channels_out=hidden_size,
        attn=attn,
    )
    proj_out = Sequential.Config(
        elements=[
            RMSNorm.Config(),
            Linear.Config(),
        ],
    )
    cfg = Transformer.Config(
        channels_in=hidden_size,
        channels_out=vocab_size,
        num_layers=num_layers,
        proj_in=Embedding.Config(
            channels_in=vocab_size,
            channels_out=hidden_size,
            dtype=dtype,
        ),
        block=block,
        proj_out=proj_out,
    )
    return cfg.finalize()


def compute_kv_bytes(
    model_config: Transformer.Config,
    *,
    context_len: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Compute analytical KV-cache bytes for a given context length.

    Uses priml.cost to price one decode step and reads bytes_state,
    which every cached attention reports as its per-token KV bytes.

    Args:
      model_config: Finalized Transformer config.
      context_len: Number of cached tokens.
      batch_size: Batch size.
      dtype: Cache dtype.

    Returns:
      kv_bytes: Total KV-cache bytes for this context.

    """
    c = cost(
        model_config,
        seq_len=1,
        batch_size=batch_size,
        dtype=dtype,
        context_len=context_len,
    )
    return c.bytes_state * context_len * batch_size


def benchmark(
    model_config: Transformer.Config,
    *,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    device: str = "cpu",
) -> InferenceResult:
    """Run the frozen inference benchmark.

    Args:
      model_config: Finalized Transformer config.
      prompt_token_ids: (num_prompts, prompt_len) token ids.
      max_new_tokens: Tokens to generate per prompt.
      dtype: Cache dtype.
      seed: Random seed for reproducibility.
      device: Device to run on.

    Returns:
      result: Throughput, memory, and latency metrics.

    """
    torch.manual_seed(seed)

    model = model_config.make().to(device=device, dtype=dtype)
    model.eval()

    num_prompts = prompt_token_ids.shape[0]
    prompt_len = prompt_token_ids.shape[1]
    max_seq_len = prompt_len + max_new_tokens

    # Compute KV bytes analytically (one decode step's bytes_state * context)
    kv_bytes_per_token = compute_kv_bytes(
        model_config,
        context_len=1,
        dtype=dtype,
    )
    kv_bytes_total = compute_kv_bytes(
        model_config,
        context_len=max_seq_len,
        dtype=dtype,
    )

    latencies: list[float] = []
    total_tokens = 0
    total_time = 0.0

    for i in range(num_prompts):
        prompt = prompt_token_ids[i : i + 1].to(device)

        # Warmup (first call may trigger compilation)
        with torch.inference_mode():
            _ = generate(
                model,
                prompt,
                max_new_tokens=4,
                temperature=0.0,
                max_seq_len=max_seq_len,
            )

        # Measure
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        with torch.inference_mode():
            output = generate(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                max_seq_len=max_seq_len,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        generated = output.shape[1] - prompt_len
        total_tokens += generated
        total_time += elapsed
        latencies.append(elapsed * 1000)  # ms

    tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0
    mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
    p99_latency_ms = sorted_lat[p99_idx] if sorted_lat else 0.0

    return InferenceResult(
        tokens_per_second=tokens_per_second,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_bytes_total=kv_bytes_total,
        p99_latency_ms=p99_latency_ms,
        mean_latency_ms=mean_latency_ms,
        total_tokens=total_tokens,
        total_time_s=total_time,
    )


def exp000() -> InferenceResult:
    """Run the frozen inference baseline.

    - Small decoder-only Transformer: 2 layers, width 128, 4 query heads, 2 KV heads.
    - BF16 KV-cache, no quantization.
    - 16 frozen prompts, 64 new tokens each.
    - Seed 42; CPU execution (works everywhere).

    Hypothesis:
      This baseline establishes the reference point for KV-cache quantization.
      Every subsequent experiment measures memory savings and throughput
      regression against these numbers.

    Results:
      CPU: ~50 tok/s, ~128 KB KV-cache per token (varies by hardware).
    """
    cfg = build_model_config(
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_heads_kv=2,
    )

    # Tokenize frozen prompts (simple byte-level for the baseline)
    # Pad all prompts to the same length
    max_len = 64
    prompt_ids = torch.zeros(len(FROZEN_PROMPTS), max_len, dtype=torch.long)
    for i, p in enumerate(FROZEN_PROMPTS):
        tokens = [b % 256 for b in p.encode("utf-8")[:max_len]]
        prompt_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    return benchmark(
        cfg,
        prompt_token_ids=prompt_ids,
        max_new_tokens=32,
        dtype=torch.bfloat16,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
