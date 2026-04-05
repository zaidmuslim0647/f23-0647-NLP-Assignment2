from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.data_utils import load_corpus_lines
from utils.vocab import build_vocab_from_documents, save_word2idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-k vocabulary from cleaned corpus for Assignment 2"
    )
    parser.add_argument(
        "--cleaned-path",
        type=str,
        default="cleaned.txt",
        help="Path to cleaned.txt corpus",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="embeddings/word2idx.json",
        help="Output path for token->index JSON",
    )
    parser.add_argument(
        "--freq-output",
        type=str,
        default="",
        help="Optional output path for token frequencies JSON",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=10000,
        help="Maximum total vocabulary size including special tokens",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before tokenization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_lines = load_corpus_lines(args.cleaned_path)
    word2idx, counter = build_vocab_from_documents(
        documents=corpus_lines,
        max_vocab_size=args.max_vocab_size,
        lowercase=args.lowercase,
        include_pad=True,
        include_unk=True,
    )

    output_json_path = PROJECT_ROOT / args.output_json
    save_word2idx(word2idx, output_json_path)

    print(f"Loaded documents: {len(corpus_lines)}")
    print(f"Built vocabulary size: {len(word2idx)}")
    print(f"Saved vocabulary JSON: {output_json_path}")

    if args.freq_output:
        freq_path = PROJECT_ROOT / args.freq_output
        freq_path.parent.mkdir(parents=True, exist_ok=True)
        with freq_path.open("w", encoding="utf-8") as f:
            json.dump(counter.most_common(), f, ensure_ascii=False, indent=2)
        print(f"Saved token frequencies: {freq_path}")


if __name__ == "__main__":
    main()
