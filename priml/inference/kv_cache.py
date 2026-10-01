"""Analytical cost of one decode step, read from a KV-cache geometry.

A decode step's intensity is not a function of context length. Reading the KV
cache costs one MAC per cached element, and both the elements and the
arithmetic over them grow linearly in context, so the two cancel. Each cached
element is a K and a V of ``channels_head * num_heads_kv`` at ``itemsize``, and
each costs one MAC:

    intensity = flops / bytes = 1 / itemsize

which is 0.5 FLOP/byte at bfloat16, about 590x below an H100's ridge. That
ratio, not the context length, is the reason decode is slow while prefill gets
faster -- and it is the reason the whole enterprise of paged attention, KV
quantization and fused decode kernels exists. Nobody makes decode faster by
making the matmul faster; they make it faster by moving fewer bytes.

The other term in a decode step, reading the weights once, has intensity 2.0
(the same MAC, with no second tensor to read) and does not grow with context.
So it dominates the mix only while the cache is comparable to the weights, and
dilutes as the cache grows. Either way a step is memory-bound at every context
length on every device in ``priml.cost``'s table: the KV term is ~590x under
the ridge and the weight term ~150x. The crossover is not a cliff; there is no
context at which decode becomes compute-bound.

``weight_bytes`` is a slot rather than a field derived from a model config
because the module does not own a model. A caller that has one reads its
parameter count; a caller isolating KV traffic passes 0, which is the honest
limit and is what makes the ``2 / itemsize`` identity exact.

``priml.cost`` owns the device table and defines the ridge; this owns the
geometry and reuses that ridge rather than restating it. Activation traffic and
the prefill step are out of scope: neither depends on the KV cache, and a
caller costing a whole forward already has them from ``priml.cost``.

The grouping assumption is Grouped-Query Attention, where several query heads
read one key and value head. ``num_heads_kv`` equal to ``num_heads`` is the
multi-head case and needs no separate path: the saving is entirely
``num_heads / num_heads_kv``, which is exactly what ``grouping_ratio`` reports
and what a grouped-query model spends its memory reduction on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import math

from configgle import Fig

import torch

from priml.cost import Device, peak, resolve_dtype


__all__ = ["DecodeCost", "KVCache"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeCost:
    """One decode step's KV traffic and where it lands against a device's ridge.

    ``intensity`` is KV flops over KV bytes and ``ridge`` is the device's own
    intensity ceiling, both from ``priml.cost``, so the two are directly
    comparable and a reader does not have to re-derive either.
    """

    flops: int
    """Attention arithmetic for the step, counted over the whole context."""

    kv_bytes: int
    """Cache traffic the step must read: the whole cache, per sequence.

    Activation traffic is excluded, as it is for any hand-written flop count:
    it does not scale with context and does not decide which side of the ridge
    a decode step lands on."""

    weight_bytes: int
    """Parameter bytes read once, independent of context.

    Reported apart from ``kv_bytes`` because the two mix differently: the cache
    grows with context and the weights do not, so a step's intensity is a
    weighted mean of the two and the weights' share falls as the cache grows."""

    weight_flops: int
    """Arithmetic over the weight read, one MAC per parameter."""

    context_len: int
    """Context length the step was priced at."""

    batch_size: int
    """Sequences decoded in lockstep."""

    intensity: float
    """``flops / kv_bytes``: FLOP per byte, for this step alone."""

    ridge: float
    """The device's intensity ceiling, from ``priml.cost``."""

    @property
    def memory_bound(self) -> bool:
        """Whether the step sits below the ridge and is waiting on the bus.

        The consequence is the point: below the ridge, a faster matmul cannot
        help and a smaller cache can.
        """
        return self.intensity < self.ridge

    @property
    def fraction_of_ridge(self) -> float:
        """``intensity / ridge``; 1.0 is exactly at the crossover."""
        if self.ridge == 0 or math.isinf(self.ridge):
            return 0.0
        return self.intensity / self.ridge


class KVCache(Fig["KVCache"]):
    """The geometry of a KV cache, and what one decode step costs against it.

    Every field is geometry rather than a tunable. A field meaningful only when
    another field holds a particular value belongs on the piece that reads it,
    not here as a scalar: there is no ``memory_bound`` flag, because
    :attr:`DecodeCost.memory_bound` is derived and a derived field a caller can
    disagree with is a field that can be wrong.
    """

    num_layers: int = -1
    """Attention layer count. Each layer keeps its own keys and values."""

    channels_head: int = -1
    """Per-head width: one key and one value of this width per KV head, per token."""

    num_heads_kv: int = -1
    """Key/value head count (-1 to mirror the query head count).

    Equal to the query head count is multi-head attention. Smaller is
    Grouped-Query Attention, where several query heads share one key and value
    head, and that ratio is where the memory saving comes from."""

    num_heads: int = -1
    """Query head count, used only to resolve ``num_heads_kv`` (-1 to infer)."""

    dtype: torch.dtype | None = None
    """Storage dtype of the cache entries, which is not the compute dtype.

    A cache held in int8 while compute runs in bfloat16 is a quantization
    choice, so this is the field a low-bit cache experiment changes."""

    weight_bytes: int = 0
    """Parameter bytes one decode step reads, once, independent of context.

    Zero isolates KV traffic, which is the honest limit for a module that owns
    no model. It is also what makes the ``2 / itemsize`` intensity identity
    exact, so a caller reading a real parameter count should say so rather than
    leave this at 0 and believe the mix it reported.
    """

    @override
    def finalize(self) -> Self:
        if self.num_heads_kv == -1 and self.num_heads != -1:
            self.num_heads_kv = self.num_heads
        return super().finalize()

    @property
    def dtype_resolved(self) -> torch.dtype:
        """The cache dtype, defaulting to torch's as ``priml.cost`` does."""
        return resolve_dtype(self.dtype)

    @property
    def grouping_ratio(self) -> float:
        """Query heads per key/value head; 1.0 for plain multi-head attention."""
        if self.num_heads <= 0 or self.num_heads_kv <= 0:
            return 1.0
        return self.num_heads / self.num_heads_kv

    def bytes_per_token(self) -> int:
        """Bytes one token of context costs in the cache, across all layers.

        Two tensors per layer -- keys and values -- each ``num_heads_kv`` wide by
        ``channels_head``. Grouped-query attention is already priced here by
        ``num_heads_kv`` being the smaller number, so there is no separate
        discount to apply on top of it.
        """
        return (
            2
            * self.num_layers
            * self.num_heads_kv
            * self.channels_head
            * self.dtype_resolved.itemsize
        )

    def tokens_in(self, num_bytes: int) -> int:
        """How many tokens of context fit in ``num_bytes`` of cache.

        An under-specified geometry reads as 0 rather than raising, because the
        caller is sizing a budget and wants a number it can compare, not an
        exception it has to catch to discover the same thing.
        """
        per_token = self.bytes_per_token()
        if per_token <= 0:
            return 0
        return max(0, num_bytes // per_token)

    def decode_cost(
        self,
        *,
        batch_size: int = 1,
        context_len: int = 1,
        device: Device | str = "h100",
        dtype: torch.dtype | None = None,
    ) -> DecodeCost:
        """Price one decode step at a context length, against a device's ridge.

        ``ridge`` is read from ``priml.cost`` at ``dtype``, so naming a
        low-precision cache prices the ridge that dtype can actually reach
        rather than the compute dtype's.

        Args:
          batch_size: Sequences decoded in lockstep.
          context_len: Tokens of context each sequence holds at this step.
          device: Device name or a form-qualified name such as ``"h100-pcie"``.
          dtype: Dtype to read the ridge at; defaults to the cache's own.

        Returns:
          cost: The step's KV traffic, its intensity, and the ridge it is
            measured against.

        """
        flops, kv_bytes, weight_flops = self._step_flops_and_bytes(
            batch_size=batch_size,
            context_len=context_len,
        )
        ridge_dtype = resolve_dtype(self.dtype if dtype is None else dtype)
        ridge = peak()[device, ridge_dtype, "intensity", "matmul"]
        total_bytes = kv_bytes + self.weight_bytes
        return DecodeCost(
            flops=flops,
            kv_bytes=kv_bytes,
            weight_bytes=self.weight_bytes,
            weight_flops=weight_flops,
            context_len=context_len,
            batch_size=batch_size,
            intensity=flops / total_bytes if total_bytes else math.inf,
            ridge=ridge,
        )

    def _step_flops_and_bytes(
        self,
        *,
        batch_size: int,
        context_len: int,
    ) -> tuple[int, int, int]:
        """Return the step's ``(flops, kv_bytes, weight_flops)``.

        The weight term is one MAC per parameter over a single token, so it
        contributes ``2 * weight_bytes / itemsize`` FLOP and ``weight_bytes``
        bytes exactly once, whatever the context. That is why it is reported
        separately from the cache rather than folded into a per-token rate.
        """
        itemsize = self.dtype_resolved.itemsize
        kv_bytes = self.bytes_per_token() * batch_size * context_len
        kv_flops = (
            2
            * self.num_layers
            * batch_size
            * context_len
            * self.num_heads_kv
            * self.channels_head
        )
        weight_flops = 2 * self.weight_bytes // itemsize
        return kv_flops + weight_flops, kv_bytes, weight_flops
