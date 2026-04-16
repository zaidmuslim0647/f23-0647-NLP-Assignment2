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

from utils.data_utils import load_corpus_lines
from utils.embeddings import (
    build_cooccurrence_matrix,
    compute_ppmi,
    tokenize_documents,
)
from utils.vocab import build_vocab_from_documents, load_word2idx, save_word2idx
from utils.word2vec import (
    analogy_predictions,
    compute_mrr,
    docs_to_token_ids,
    nearest_neighbors,
    plot_loss_curves,
    train_skipgram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Commit 3 skip-gram pipeline")
    parser.add_argument("--cleaned-path", type=str, default="cleaned.txt")
    parser.add_argument("--raw-path", type=str, default="raw.txt")
    parser.add_argument("--word2idx-path", type=str, default="embeddings/word2idx.json")
    parser.add_argument("--ppmi-path", type=str, default="embeddings/ppmi_matrix.npy")

    parser.add_argument("--max-vocab-size", type=int, default=10000)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-negatives", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lowercase", action="store_true")

    parser.add_argument(
        "--embeddings-output", type=str, default="embeddings/embeddings_w2v.npy"
    )
    parser.add_argument(
        "--neighbors-output", type=str, default="embeddings/w2v_nearest_neighbors.json"
    )
    parser.add_argument(
        "--analogy-output", type=str, default="embeddings/w2v_analogies.json"
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default="embeddings/w2v_four_condition_comparison.json",
    )
    parser.add_argument("--loss-plot", type=str, default="embeddings/w2v_loss_curve.png")
    return parser.parse_args()


def _save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _default_analogies() -> list[tuple[str, str, str]]:
    return [
        ("Mard", "Aurat", "Larka"),
        ("Islamabad", "Pakistan", "Delhi"),
        ("Captain", "Team", "Wazir"),
        ("Doctor", "Hospital", "Teacher"),
        ("Cricket", "Player", "Hockey"),
        ("Parliament", "Hukumat", "Adalat"),
        ("Punjab", "Lahore", "Sindh"),
        ("Bank", "Maeeshat", "School"),
        ("Fauj", "Sarhad", "Police"),
        ("Sehat", "Hospital", "Taleem"),
    ]


def _default_mrr_pairs() -> list[tuple[str, str]]:
    return [
        ("Pakistan", "Islamabad"),
        ("Hukumat", "Parliament"),
        ("Adalat", "Qanoon"),
        ("Maeeshat", "Bank"),
        ("Fauj", "Sarhad"),
        ("Sehat", "Hospital"),
        ("Taleem", "School"),
        ("Aabadi", "Shehar"),
        ("Cricket", "Match"),
        ("Team", "Player"),
        ("Election", "Vote"),
        ("Wazir", "Hukumat"),
        ("Karachi", "Sindh"),
        ("Lahore", "Punjab"),
        ("Doctor", "Sehat"),
        ("Budget", "Maeeshat"),
        ("Trade", "Bank"),
        ("Court", "Judge"),
        ("Police", "Qanoon"),
        ("Parliament", "Siasat"),
    ]


def _build_ppmi_baseline(
    cleaned_docs: list[str],
    word2idx: dict[str, int],
    lowercase: bool,
    window_size: int,
    ppmi_path: Path,
) -> np.ndarray:
    if ppmi_path.exists():
        return np.load(ppmi_path)

    tokenized = tokenize_documents(cleaned_docs, lowercase=lowercase)
    cooc = build_cooccurrence_matrix(tokenized, word2idx, window_size=window_size, symmetric=True)
    ppmi = compute_ppmi(cooc)
    ppmi_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ppmi_path, ppmi)
    return ppmi


def _train_condition(
    docs: list[str],
    word2idx: dict[str, int],
    embedding_dim: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float]]:
    token_ids = docs_to_token_ids(docs, word2idx, lowercase=args.lowercase)
    model, losses = train_skipgram(
        token_ids=token_ids,
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    return model.averaged_embeddings(), losses


def main() -> None:
    args = parse_args()

    cleaned_docs = load_corpus_lines(PROJECT_ROOT / args.cleaned_path)
    print(f"Loaded cleaned corpus documents: {len(cleaned_docs)}")

    word2idx_path = PROJECT_ROOT / args.word2idx_path
    if word2idx_path.exists():
        word2idx = load_word2idx(word2idx_path)
        print(f"Loaded existing vocabulary with {len(word2idx)} tokens")
    else:
        word2idx, _ = build_vocab_from_documents(
            documents=cleaned_docs,
            max_vocab_size=args.max_vocab_size,
            lowercase=args.lowercase,
            include_pad=True,
            include_unk=True,
        )
        save_word2idx(word2idx, word2idx_path)
        print(f"Built vocabulary with {len(word2idx)} tokens")

    raw_path = PROJECT_ROOT / args.raw_path
    raw_docs = load_corpus_lines(raw_path) if raw_path.exists() else []
    if raw_docs:
        print(f"Loaded raw corpus documents: {len(raw_docs)}")
    else:
        print("raw.txt not found; condition C2 will be skipped")

    # C3 required model (cleaned, d=100)
    emb_c3, loss_c3 = _train_condition(cleaned_docs, word2idx, embedding_dim=100, args=args)

    emb_out = PROJECT_ROOT / args.embeddings_output
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_out, emb_c3.astype(np.float32))
    print(f"Saved C3 embeddings to: {emb_out} shape={emb_c3.shape}")

    query_words_required = [
        "Pakistan",
        "Hukumat",
        "Adalat",
        "Maeeshat",
        "Fauj",
        "Sehat",
        "Taleem",
        "Aabadi",
    ]
    nn_required = nearest_neighbors(emb_c3, word2idx, query_words_required, top_k=10)
    _save_json(nn_required, PROJECT_ROOT / args.neighbors_output)
    print(f"Saved nearest-neighbor evaluation: {PROJECT_ROOT / args.neighbors_output}")

    analogies = _default_analogies()
    analogy_results = analogy_predictions(emb_c3, word2idx, analogies, top_k=3)
    _save_json(analogy_results, PROJECT_ROOT / args.analogy_output)
    print(f"Saved analogy evaluation: {PROJECT_ROOT / args.analogy_output}")

    ppmi = _build_ppmi_baseline(
        cleaned_docs=cleaned_docs,
        word2idx=word2idx,
        lowercase=args.lowercase,
        window_size=args.window_size,
        ppmi_path=PROJECT_ROOT / args.ppmi_path,
    )

    emb_c2 = None
    loss_c2: list[float] = []
    if raw_docs:
        emb_c2, loss_c2 = _train_condition(raw_docs, word2idx, embedding_dim=100, args=args)

    emb_c4, loss_c4 = _train_condition(cleaned_docs, word2idx, embedding_dim=200, args=args)

    comparison_queries = ["Pakistan", "Hukumat", "Adalat", "Maeeshat", "Fauj"]
    labeled_pairs = _default_mrr_pairs()

    conditions: dict[str, np.ndarray] = {
        "C1_ppmi_baseline": ppmi,
        "C3_skipgram_cleaned_d100": emb_c3,
        "C4_skipgram_cleaned_d200": emb_c4,
    }
    if emb_c2 is not None:
        conditions["C2_skipgram_raw_d100"] = emb_c2

    comparison: dict[str, object] = {}
    for name, matrix in conditions.items():
        comparison[name] = {
            "top5_neighbors": nearest_neighbors(matrix, word2idx, comparison_queries, top_k=5),
            "mrr": compute_mrr(matrix, word2idx, labeled_pairs),
        }

    _save_json(comparison, PROJECT_ROOT / args.comparison_output)
    print(f"Saved four-condition comparison: {PROJECT_ROOT / args.comparison_output}")

    loss_by_condition = {
        "C3 cleaned d=100": loss_c3,
        "C4 cleaned d=200": loss_c4,
    }
    if loss_c2:
        loss_by_condition["C2 raw d=100"] = loss_c2
    plot_loss_curves(loss_by_condition, PROJECT_ROOT / args.loss_plot)
    print(f"Saved training loss curve: {PROJECT_ROOT / args.loss_plot}")


if __name__ == "__main__":
    main()
