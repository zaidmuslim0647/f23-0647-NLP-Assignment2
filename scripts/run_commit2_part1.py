from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.data_utils import load_corpus_lines, load_metadata
from utils.embeddings import (
    build_cooccurrence_matrix,
    build_term_document_matrix,
    compute_ppmi,
    compute_tfidf,
    infer_topics_from_metadata,
    plot_tsne_for_top_tokens,
    tokenize_documents,
    top_discriminative_words_per_topic,
    top_k_neighbors,
)
from utils.vocab import build_token_counter, build_vocab_from_documents, load_word2idx, save_word2idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Commit 2 Part 1 pipeline")
    parser.add_argument("--cleaned-path", type=str, default="cleaned.txt")
    parser.add_argument("--metadata-path", type=str, default="Metadata.json")
    parser.add_argument("--word2idx-path", type=str, default="embeddings/word2idx.json")
    parser.add_argument("--max-vocab-size", type=int, default=10000)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--lowercase", action="store_true")

    parser.add_argument("--tfidf-output", type=str, default="embeddings/tfidf_matrix.npy")
    parser.add_argument("--ppmi-output", type=str, default="embeddings/ppmi_matrix.npy")
    parser.add_argument(
        "--discriminative-output",
        type=str,
        default="embeddings/top_discriminative_words_by_topic.json",
    )
    parser.add_argument(
        "--neighbors-output",
        type=str,
        default="embeddings/ppmi_nearest_neighbors.json",
    )
    parser.add_argument("--tsne-output", type=str, default="embeddings/ppmi_tsne_top200.png")
    return parser.parse_args()


def _save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    docs = load_corpus_lines(PROJECT_ROOT / args.cleaned_path)
    print(f"Loaded cleaned corpus documents: {len(docs)}")

    word2idx_path = PROJECT_ROOT / args.word2idx_path
    if word2idx_path.exists():
        word2idx = load_word2idx(word2idx_path)
        print(f"Loaded existing vocabulary: {len(word2idx)} tokens")
    else:
        word2idx, _ = build_vocab_from_documents(
            docs,
            max_vocab_size=args.max_vocab_size,
            lowercase=args.lowercase,
            include_pad=True,
            include_unk=True,
        )
        save_word2idx(word2idx, word2idx_path)
        print(f"Built and saved vocabulary: {len(word2idx)} tokens")

    tokenized_docs = tokenize_documents(docs, lowercase=args.lowercase)

    term_doc = build_term_document_matrix(tokenized_docs, word2idx)
    tfidf = compute_tfidf(term_doc)
    tfidf_out = PROJECT_ROOT / args.tfidf_output
    tfidf_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(tfidf_out, tfidf)
    print(f"Saved TF-IDF matrix: {tfidf_out} shape={tfidf.shape}")

    metadata_path = PROJECT_ROOT / args.metadata_path
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
        topics = infer_topics_from_metadata(metadata)
        if len(topics) == len(docs):
            top_words = top_discriminative_words_per_topic(tfidf, topics, word2idx, top_k=10)
            discr_out = PROJECT_ROOT / args.discriminative_output
            _save_json(top_words, discr_out)
            print(f"Saved top discriminative words by topic: {discr_out}")
        else:
            print(
                "Skipped top discriminative words: inferred topic count does not match documents "
                f"({len(topics)} vs {len(docs)})"
            )
    else:
        print("Metadata.json not found, skipping topic-discriminative terms export")

    cooc = build_cooccurrence_matrix(
        tokenized_docs, word2idx, window_size=args.window_size, symmetric=True
    )
    ppmi = compute_ppmi(cooc)
    ppmi_out = PROJECT_ROOT / args.ppmi_output
    ppmi_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(ppmi_out, ppmi)
    print(f"Saved PPMI matrix: {ppmi_out} shape={ppmi.shape}")

    default_queries = [
        "Pakistan",
        "Hukumat",
        "Adalat",
        "Maeeshat",
        "Fauj",
        "Sehat",
        "Taleem",
        "Aabadi",
        "حکومت",
        "پاکستان",
    ]
    neighbors = top_k_neighbors(ppmi, word2idx, default_queries, top_k=5)
    neigh_out = PROJECT_ROOT / args.neighbors_output
    _save_json(neighbors, neigh_out)
    print(f"Saved nearest neighbors JSON: {neigh_out}")

    token_counter = build_token_counter(docs, lowercase=args.lowercase)
    tsne_out = PROJECT_ROOT / args.tsne_output
    plot_tsne_for_top_tokens(ppmi, word2idx, token_counter, tsne_out, top_n=200)
    print(f"Saved t-SNE plot: {tsne_out}")


if __name__ == "__main__":
    main()
