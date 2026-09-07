"""Deterministic, local, approximate token counting (Phase 3).

There is no real tokenizer here -- no `transformers`, no model vocabulary,
no network access. Every count produced by this module is an
*approximation*, and callers must treat it as such. The default estimator
uses a character-length heuristic (roughly 4 characters per token for
common English BPE tokenizers); it deliberately does **not** equate word
count with token count, since word length varies a lot and that would be
a much cruder (and differently biased) approximation.
"""

from __future__ import annotations

import math
from typing import Protocol

__all__ = [
    "CharacterRatioTokenEstimator",
    "DEFAULT_TOKEN_ESTIMATOR",
    "TokenEstimator",
]


class TokenEstimator(Protocol):
    """A local, deterministic, approximate token counter.

    Implementations must be pure functions of their input text: the same
    text must always produce the same estimate, with no network access and
    no real tokenizer vocabulary involved.
    """

    def estimate(self, text: str) -> int:
        """Return an approximate, non-negative token count for `text`."""
        ...

    def describe(self) -> str:
        """Return a short, stable identifier for this estimator's method+config.

        Used in `GeneratorConfiguration` so a serialized workload records
        *how* its token counts were approximated, without needing to
        serialize estimator internals.
        """
        ...


class CharacterRatioTokenEstimator:
    """Approximates token count as ``ceil(len(text) / chars_per_token)``.

    This is a widely used rough heuristic for English text tokenized by
    common BPE tokenizers (roughly 4 characters per token), not a real
    tokenizer count. It is intentionally character-based rather than
    word-based: naively counting words would silently treat token count
    and word count as equivalent, which is a materially different (and
    less accurate) approximation that this module avoids.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {chars_per_token}")
        self._chars_per_token = chars_per_token

    @property
    def chars_per_token(self) -> float:
        return self._chars_per_token

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self._chars_per_token)

    def describe(self) -> str:
        return f"character_ratio(chars_per_token={self._chars_per_token})"


DEFAULT_TOKEN_ESTIMATOR: TokenEstimator = CharacterRatioTokenEstimator()
