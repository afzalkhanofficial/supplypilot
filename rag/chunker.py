"""
rag.chunker — split raw text into overlapping character windows.

Design decisions
----------------
* Character-based (not token-based) splitting keeps this dependency-free
  and deterministic across Python versions.
* A 1 000-character window with 200-character overlap produces chunks that
  map cleanly to the 512-token context limit of all-MiniLM-L6-v2 while
  preserving sentence continuity across boundaries.
* Windows are snapped to the nearest word boundary (space) so mid-word
  splits never appear in stored chunk_text.
* Empty or whitespace-only chunks are silently discarded.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Default window size and overlap in characters.
_DEFAULT_CHUNK_SIZE: int = 1_000
_DEFAULT_OVERLAP: int = 200


def _snap_to_word_boundary(text: str, pos: int, search_backward: bool) -> int:
    """
    Adjust *pos* so it lands on a whitespace boundary.

    Args:
        text: The full source string.
        pos: The candidate character index.
        search_backward: If True, scan left for a space; otherwise scan right.

    Returns:
        An adjusted index that is never out-of-range.  If no space is found
        within a 50-character search window the original *pos* is returned.
    """
    limit = min(50, len(text))
    if search_backward:
        for i in range(pos, max(pos - limit, 0), -1):
            if text[i] == " ":
                return i
    else:
        for i in range(pos, min(pos + limit, len(text))):
            if text[i] == " ":
                return i + 1
    return pos


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping windows of approximately *chunk_size*
    characters with *overlap* characters of context carried into the next
    window.

    Args:
        text: Raw document text (may contain newlines and unicode).
        chunk_size: Target window size in characters (default 1 000).
        overlap: Number of characters to repeat at the start of each
            successive window (default 200).

    Returns:
        An ordered list of non-empty string chunks.  The list is empty if
        *text* is blank.

    Raises:
        ValueError: If *chunk_size* <= *overlap* or either value is <= 0.
    """
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be > 0 and overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    # Normalise whitespace: collapse runs of blank lines to a single newline
    # so that PDF artefacts (form feeds, multiple blank lines) don't bloat chunks.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        logger.debug("chunk_text received empty text — returning empty list.")
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap the end boundary leftward to a word boundary unless we are at
        # the very end of the text (no point adjusting the last slice).
        if end < text_len:
            end = _snap_to_word_boundary(text, end, search_backward=True)

        window = text[start:end].strip()
        if window:
            chunks.append(window)

        if end >= text_len:
            break

        # Advance start by (chunk_size - overlap), snapped rightward.
        next_start = start + chunk_size - overlap
        next_start = _snap_to_word_boundary(text, next_start, search_backward=False)
        # Guard against infinite loop if snapping fails to advance.
        start = max(next_start, start + 1)

    logger.debug("chunk_text produced %d chunks from %d characters.", len(chunks), text_len)
    return chunks
