"""Inference-time utilities.

``exp000`` is the frozen inference baseline: a small decoder-only model
run with BF16 KV-cache, scored on tokens/sec and KV bytes. Every quantization
experiment measures against this number.

The baseline is judged worst-case, not average: p99 latency over frozen
eval prompts, so a quantization method that occasionally stalls is penalized.
"""

from priml.inference.benchmark import InferenceResult, exp000

__all__ = ["InferenceResult", "exp000"]
