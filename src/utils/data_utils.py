from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
# Covers Urdu/Arabic punctuation marks and common sentence delimiters.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\u061f\u06d4]+")
_TOKEN_RE = re.compile(r"[\u0600-\u06FF\w]+", flags=re.UNICODE)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_corpus_lines(file_path: str | Path, drop_empty: bool = True) -> list[str]:
    """Load corpus text file as stripped lines (each line treated as one document)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = _normalize_whitespace(raw_line)
            if drop_empty and not line:
                continue
            lines.append(line)
    return lines


def sentence_split(text: str) -> list[str]:
    """Split a text into sentences using basic Urdu-friendly punctuation rules."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_text(text: str, lowercase: bool = False) -> list[str]:
    """Tokenize text into Urdu/word tokens while dropping punctuation-only tokens."""
    norm = _normalize_whitespace(text)
    if lowercase:
        norm = norm.lower()
    return _TOKEN_RE.findall(norm)


def load_metadata(file_path: str | Path) -> Any:
    """Load metadata JSON. Supports list or dict top-level structures."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """Write JSON to disk with UTF-8 encoding."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
