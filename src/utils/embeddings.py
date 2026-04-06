from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from .data_utils import tokenize_text
from .vocab import PAD_TOKEN, UNK_TOKEN


def tokenize_documents(documents: list[str], lowercase: bool = False) -> list[list[str]]:
    """Tokenize each document into a token sequence."""
    return [tokenize_text(doc, lowercase=lowercase) for doc in documents]


def apply_vocab(tokens: list[str], word2idx: dict[str, int]) -> list[str]:
    """Replace OOV tokens by <UNK> token string."""
    if UNK_TOKEN not in word2idx:
        raise KeyError("word2idx must contain <UNK>")
    return [tok if tok in word2idx else UNK_TOKEN for tok in tokens]


def build_term_document_matrix(
    tokenized_docs: list[list[str]], word2idx: dict[str, int]
) -> np.ndarray:
    """Build dense term-document count matrix of shape [num_docs, vocab_size]."""
    num_docs = len(tokenized_docs)
    vocab_size = len(word2idx)
    matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)

    for d_idx, tokens in enumerate(tokenized_docs):
        mapped = apply_vocab(tokens, word2idx)
        counts = Counter(mapped)
        for token, cnt in counts.items():
            matrix[d_idx, word2idx[token]] = float(cnt)
    return matrix


def compute_tfidf(term_doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute TF-IDF as specified in assignment:
    TF-IDF(w, d) = TF(w, d) * log(N / (1 + df(w))).
    """
    if term_doc_matrix.ndim != 2:
        raise ValueError("term_doc_matrix must be 2D")

    tf = term_doc_matrix
    num_docs = tf.shape[0]
    df = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log(float(num_docs) / (1.0 + df)).astype(np.float32)
    return tf * idf[None, :]


def build_cooccurrence_matrix(
    tokenized_docs: list[list[str]],
    word2idx: dict[str, int],
    window_size: int = 5,
    symmetric: bool = True,
) -> np.ndarray:
    """Build dense word-word co-occurrence matrix with sliding context window."""
    vocab_size = len(word2idx)
    cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for tokens in tokenized_docs:
        mapped = apply_vocab(tokens, word2idx)
        ids = [word2idx[t] for t in mapped]

        for i, center_id in enumerate(ids):
            left = max(0, i - window_size)
            right = min(len(ids), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context_id = ids[j]
                cooc[center_id, context_id] += 1.0
                if symmetric:
                    cooc[context_id, center_id] += 1.0
    return cooc


def compute_ppmi(cooc_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Positive PMI matrix from co-occurrence matrix."""
    total = float(cooc_matrix.sum())
    if total <= 0:
        raise ValueError("cooc_matrix has zero total count")

    p_wc = cooc_matrix / total
    p_w = p_wc.sum(axis=1, keepdims=True)
    p_c = p_wc.sum(axis=0, keepdims=True)

    denom = p_w @ p_c
    pmi = np.log2((p_wc + eps) / (denom + eps))
    ppmi = np.maximum(0.0, pmi)
    ppmi[cooc_matrix <= 0] = 0.0
    return ppmi.astype(np.float32)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / norms
    return normalized @ normalized.T


def top_k_neighbors(
    matrix: np.ndarray,
    word2idx: dict[str, int],
    query_words: list[str],
    top_k: int = 5,
) -> dict[str, list[dict[str, float]]]:
    """Return top-k nearest neighbors (cosine) for each query word."""
    idx2word = {i: w for w, i in word2idx.items()}
    sim = cosine_similarity_matrix(matrix)

    results: dict[str, list[dict[str, float]]] = {}
    for query in query_words:
        if query not in word2idx:
            results[query] = []
            continue

        q_idx = word2idx[query]
        scores = sim[q_idx].copy()
        scores[q_idx] = -1.0
        nn_idx = np.argsort(-scores)[:top_k]
        results[query] = [
            {"word": idx2word[int(i)], "cosine": float(scores[int(i)])} for i in nn_idx
        ]
    return results


def _semantic_group(token: str) -> str:
    """Assign token to a coarse semantic group for t-SNE color coding."""
    politics = {
        "Pakistan",
        "Hukumat",
        "Wazir",
        "Parliament",
        "Election",
        "سیاست",
        "حکومت",
        "وزیر",
        "پارلیمنٹ",
        "انتخابات",
    }
    sports = {
        "Cricket",
        "Match",
        "Team",
        "Player",
        "Score",
        "کھیل",
        "کرکٹ",
        "میچ",
        "ٹیم",
        "کھلاڑی",
    }
    geography = {
        "Lahore",
        "Karachi",
        "Islamabad",
        "Punjab",
        "Sindh",
        "پاکستان",
        "لاہور",
        "کراچی",
        "اسلام آباد",
        "پنجاب",
        "سندھ",
    }

    if token in politics:
        return "politics"
    if token in sports:
        return "sports"
    if token in geography:
        return "geography"
    return "other"


def plot_tsne_for_top_tokens(
    matrix: np.ndarray,
    word2idx: dict[str, int],
    token_counter: Counter[str],
    output_path: str | Path,
    top_n: int = 200,
    random_state: int = 42,
) -> None:
    """Create and save t-SNE plot for most frequent tokens."""
    idx2word = {i: w for w, i in word2idx.items()}

    top_tokens = [
        tok
        for tok, _ in token_counter.most_common()
        if tok in word2idx and tok not in {PAD_TOKEN, UNK_TOKEN}
    ][:top_n]

    if len(top_tokens) < 5:
        raise ValueError("Not enough tokens for t-SNE visualization")

    token_ids = np.array([word2idx[t] for t in top_tokens], dtype=np.int64)
    vectors = matrix[token_ids]

    perplexity = min(30, len(top_tokens) - 1)
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coords = tsne.fit_transform(vectors)

    group_to_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for i, token in enumerate(top_tokens):
        group_to_points[_semantic_group(token)].append((coords[i, 0], coords[i, 1]))

    plt.figure(figsize=(12, 9))
    for group, points in group_to_points.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, label=group, alpha=0.7, s=35)

    # Label a subset to keep the plot readable while preserving token context.
    label_count = min(40, len(top_tokens))
    for i in range(label_count):
        plt.text(coords[i, 0], coords[i, 1], top_tokens[i], fontsize=8)

    plt.title("t-SNE of Top Frequent Tokens (PPMI Vectors)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Semantic Category")
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    plt.close()


def infer_topics_from_metadata(metadata: object) -> list[str]:
    """
    Extract topic/category labels from metadata in a schema-tolerant way.
    Returns an empty list if no labels can be inferred.
    """
    labels: list[str] = []

    def pick_label(item: dict) -> str | None:
        for key in ["topic", "category", "label", "section", "genre"]:
            if key in item and isinstance(item[key], str):
                return item[key]
        return None

    if isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, dict):
                label = pick_label(item)
                if label is not None:
                    labels.append(label)
    elif isinstance(metadata, dict):
        if "articles" in metadata and isinstance(metadata["articles"], list):
            for item in metadata["articles"]:
                if isinstance(item, dict):
                    label = pick_label(item)
                    if label is not None:
                        labels.append(label)
        else:
            # dict of id -> metadata dict
            for value in metadata.values():
                if isinstance(value, dict):
                    label = pick_label(value)
                    if label is not None:
                        labels.append(label)

    return labels


def top_discriminative_words_per_topic(
    tfidf_matrix: np.ndarray,
    topics: list[str],
    word2idx: dict[str, int],
    top_k: int = 10,
) -> dict[str, list[str]]:
    """
    Compute top-k discriminative words per topic by mean TF-IDF within each topic.
    Expects len(topics) == number of documents.
    """
    if len(topics) != tfidf_matrix.shape[0]:
        raise ValueError("Number of topics must match number of documents")

    idx2word = {i: w for w, i in word2idx.items()}
    topic_to_rows: dict[str, list[int]] = defaultdict(list)
    for row_idx, topic in enumerate(topics):
        topic_to_rows[topic].append(row_idx)

    results: dict[str, list[str]] = {}
    for topic, rows in topic_to_rows.items():
        mean_scores = tfidf_matrix[rows].mean(axis=0)
        best = np.argsort(-mean_scores)[:top_k]
        words = [idx2word[int(i)] for i in best if idx2word[int(i)] not in {PAD_TOKEN, UNK_TOKEN}]
        results[topic] = words[:top_k]
    return results
