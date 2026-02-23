"""
HuggingFace Corpus Loader for NSCK Distributional Semantics
============================================================
Attempts to download a small slice of a HuggingFace dataset to augment the
built-in corpus used for distributional HV training.

Usage
-----
    from python.core.language.hf_corpus_loader import load_hf_corpus

    sentences = load_hf_corpus("fka/awesome-chatgpt-prompts", max_sentences=1000)
    # Returns List[List[str]] — each inner list is a tokenised sentence.
    # Falls back to [] silently when offline or when the datasets library is absent.

Supported datasets (tried in order when dataset_name is "auto"):
  1. fka/awesome-chatgpt-prompts  — ~165 prompts, text column "prompt"
  2. yahma/alpaca-cleaned         — 52K rows, text in "instruction"+"output"

All network errors are caught; callers always receive a valid (possibly empty)
list and should fall back to BUILTIN_CORPUS.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hf_corpus(
    dataset_name: str = "auto",
    split: str = "train",
    max_sentences: int = 2000,
    text_columns: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Download *dataset_name* from HuggingFace and return tokenised sentences.

    Parameters
    ----------
    dataset_name  : HuggingFace dataset id, or ``"auto"`` to try defaults.
    split         : dataset split to use (default ``"train"``).
    max_sentences : cap on how many tokenised sentences to return.
    text_columns  : list of column names to extract text from.
                    Auto-detected if ``None``.

    Returns
    -------
    List[List[str]]
        Tokenised sentences.  Empty list on any failure.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        logger.debug("[HFCorpus] 'datasets' not installed; skipping HF corpus.")
        return []

    if dataset_name == "auto":
        candidates = [
            ("fka/awesome-chatgpt-prompts", ["prompt"]),
            ("yahma/alpaca-cleaned",        ["instruction", "output"]),
        ]
    else:
        candidates = [(dataset_name, text_columns or [])]

    for ds_name, cols in candidates:
        sentences = _try_load(ds_name, split, max_sentences, cols or None, load_dataset)
        if sentences:
            logger.info("[HFCorpus] Loaded %d sentences from %s", len(sentences), ds_name)
            return sentences

    logger.debug("[HFCorpus] All HuggingFace downloads failed; caller should use BUILTIN_CORPUS.")
    return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_load(
    dataset_name: str,
    split: str,
    max_sentences: int,
    text_columns: Optional[List[str]],
    load_dataset,
) -> List[List[str]]:
    """Single attempt to download one dataset.  Returns [] on any error."""
    try:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=False)
    except Exception as exc:
        logger.debug("[HFCorpus] Could not load %s: %s", dataset_name, exc)
        return []

    # Auto-detect text columns
    if not text_columns:
        text_columns = _detect_text_columns(ds.column_names)
    if not text_columns:
        return []

    sentences: List[List[str]] = []
    for row in ds:
        if len(sentences) >= max_sentences:
            break
        for col in text_columns:
            raw = row.get(col, "")
            if not isinstance(raw, str):
                continue
            for sent in _split_to_sentences(raw):
                tokens = _tokenize(sent)
                if len(tokens) >= 3:  # skip trivial fragments
                    sentences.append(tokens)
                if len(sentences) >= max_sentences:
                    break
    return sentences


_TEXT_COLUMN_NAMES = {
    "text", "prompt", "instruction", "output", "response",
    "question", "answer", "context", "content", "body",
}

def _detect_text_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if c.lower() in _TEXT_COLUMN_NAMES]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_to_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _tokenize(sentence: str) -> List[str]:
    return re.findall(r"\b[a-z]{2,}\b", sentence.lower())
