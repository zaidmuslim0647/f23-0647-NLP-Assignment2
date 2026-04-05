from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .data_utils import tokenize_text

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_token_counter(documents: Iterable[str], lowercase: bool = False) -> Counter[str]:
    """Count token frequencies over a collection of documents."""
    counter: Counter[str] = Counter()
    for doc in documents:
        tokens = tokenize_text(doc, lowercase=lowercase)
        counter.update(tokens)
    return counter


def build_vocab_from_counter(
    counter: Counter[str],
    max_vocab_size: int = 10000,
    include_pad: bool = True,
    include_unk: bool = True,
) -> dict[str, int]:
    """
    Build token->index mapping from frequency counter.

    Ordering:
    1) Special tokens first
    2) Remaining tokens by descending frequency
    3) Alphabetical tie-break for deterministic output
    """
    if max_vocab_size <= 0:
        raise ValueError("max_vocab_size must be > 0")

    specials: list[str] = []
    if include_pad:
        specials.append(PAD_TOKEN)
    if include_unk:
        specials.append(UNK_TOKEN)

    token_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    limit = max(0, max_vocab_size - len(specials))
    selected_tokens = [token for token, _ in token_items[:limit]]

    word2idx: dict[str, int] = {}
    for token in specials + selected_tokens:
        if token not in word2idx:
            word2idx[token] = len(word2idx)
    return word2idx


def build_vocab_from_documents(
    documents: Iterable[str],
    max_vocab_size: int = 10000,
    lowercase: bool = False,
    include_pad: bool = True,
    include_unk: bool = True,
) -> tuple[dict[str, int], Counter[str]]:
    """Build vocabulary directly from raw text documents."""
    counter = build_token_counter(documents, lowercase=lowercase)
    word2idx = build_vocab_from_counter(
        counter=counter,
        max_vocab_size=max_vocab_size,
        include_pad=include_pad,
        include_unk=include_unk,
    )
    return word2idx, counter


def numericalize_tokens(tokens: list[str], word2idx: dict[str, int]) -> list[int]:
    """Map tokens to ids using <UNK> fallback."""
    unk_idx = word2idx.get(UNK_TOKEN)
    if unk_idx is None:
        raise KeyError("Vocabulary must contain <UNK> token")
    return [word2idx.get(token, unk_idx) for token in tokens]


def save_word2idx(word2idx: dict[str, int], file_path: str | Path) -> None:
    """Persist vocabulary mapping as JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)


def load_word2idx(file_path: str | Path) -> dict[str, int]:
    """Load vocabulary mapping from JSON."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}
