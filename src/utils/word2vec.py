from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .data_utils import tokenize_text
from .vocab import UNK_TOKEN


def docs_to_token_ids(
    documents: list[str],
    word2idx: dict[str, int],
    lowercase: bool = False,
) -> list[int]:
    """Tokenize and map the corpus into a flat token-id stream."""
    if UNK_TOKEN not in word2idx:
        raise KeyError("word2idx must contain <UNK>")

    unk_idx = word2idx[UNK_TOKEN]
    ids: list[int] = []
    for doc in documents:
        for tok in tokenize_text(doc, lowercase=lowercase):
            ids.append(word2idx.get(tok, unk_idx))
    return ids


def generate_skipgram_pairs(token_ids: list[int], window_size: int = 5) -> np.ndarray:
    """Build positive (center, context) pairs using a symmetric context window."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    pairs: list[tuple[int, int]] = []
    for i, center_id in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(len(token_ids), i + window_size + 1)
        for j in range(left, right):
            if i == j:
                continue
            pairs.append((center_id, token_ids[j]))

    if not pairs:
        raise ValueError("No skip-gram pairs were generated. Check corpus size/window.")
    return np.asarray(pairs, dtype=np.int64)


def build_noise_distribution(
    token_ids: list[int],
    vocab_size: int,
    power: float = 0.75,
) -> np.ndarray:
    """Build the negative-sampling distribution Pn(w) proportional to f(w)^power."""
    counts = np.bincount(np.asarray(token_ids, dtype=np.int64), minlength=vocab_size).astype(
        np.float64
    )
    adjusted = np.power(np.maximum(counts, 0.0), power)
    total = adjusted.sum()
    if total <= 0:
        raise ValueError("Noise distribution has zero mass")
    return (adjusted / total).astype(np.float64)


class SkipGramNegSampling(nn.Module):
    """Skip-gram Word2Vec with separate center/context embedding matrices."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        init_bound = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_bound, init_bound)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        pos_context_ids: torch.Tensor,
        neg_context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative-sampling objective:
        -log sigma(u_o^T v_c) - sum_k log sigma(-u_k^T v_c)
        """
        center_vec = self.center_embeddings(center_ids)  # [B, D]
        pos_vec = self.context_embeddings(pos_context_ids)  # [B, D]
        neg_vec = self.context_embeddings(neg_context_ids)  # [B, K, D]

        pos_score = torch.sum(center_vec * pos_vec, dim=1)  # [B]
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-12)

        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-12), dim=1)

        return (pos_loss + neg_loss).mean()

    def averaged_embeddings(self) -> np.ndarray:
        """Return final embeddings as 0.5 * (V + U)."""
        center = self.center_embeddings.weight.detach().cpu().numpy()
        context = self.context_embeddings.weight.detach().cpu().numpy()
        return 0.5 * (center + context)


def train_skipgram(
    token_ids: list[int],
    vocab_size: int,
    embedding_dim: int = 100,
    window_size: int = 5,
    num_negatives: int = 10,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> tuple[SkipGramNegSampling, list[float]]:
    """Train skip-gram with negative sampling and return model + loss history."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SkipGramNegSampling(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pairs = generate_skipgram_pairs(token_ids, window_size=window_size)
    noise_dist = build_noise_distribution(token_ids, vocab_size=vocab_size, power=0.75)

    loss_history: list[float] = []
    pair_count = pairs.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(pair_count)
        running_loss = 0.0
        seen = 0

        pbar = tqdm(
            range(0, pair_count, batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        for start in pbar:
            batch_idx = indices[start : start + batch_size]
            batch = pairs[batch_idx]

            centers = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
            positives = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
            negatives_np = np.random.choice(
                vocab_size,
                size=(batch.shape[0], num_negatives),
                replace=True,
                p=noise_dist,
            )
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(centers, positives, negatives)
            loss.backward()
            optimizer.step()

            batch_n = batch.shape[0]
            running_loss += float(loss.item()) * batch_n
            seen += batch_n
            pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

        epoch_loss = running_loss / max(1, seen)
        loss_history.append(epoch_loss)

    return model, loss_history


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def nearest_neighbors(
    embedding_matrix: np.ndarray,
    word2idx: dict[str, int],
    query_words: list[str],
    top_k: int = 10,
) -> dict[str, list[dict[str, float]]]:
    """Get top-k nearest neighbors by cosine similarity for each query."""
    idx2word = {i: w for w, i in word2idx.items()}
    emb = _normalize_rows(embedding_matrix.astype(np.float32))

    results: dict[str, list[dict[str, float]]] = {}
    for query in query_words:
        if query not in word2idx:
            results[query] = []
            continue

        q_idx = word2idx[query]
        scores = emb @ emb[q_idx]
        scores[q_idx] = -math.inf
        best = np.argsort(-scores)[:top_k]
        results[query] = [
            {"word": idx2word[int(i)], "cosine": float(scores[int(i)])} for i in best
        ]
    return results


def analogy_predictions(
    embedding_matrix: np.ndarray,
    word2idx: dict[str, int],
    analogies: list[tuple[str, str, str]],
    top_k: int = 3,
) -> list[dict[str, object]]:
    """Solve analogy vectors: v(b) - v(a) + v(c)."""
    idx2word = {i: w for w, i in word2idx.items()}
    emb = _normalize_rows(embedding_matrix.astype(np.float32))

    outputs: list[dict[str, object]] = []
    for a, b, c in analogies:
        if a not in word2idx or b not in word2idx or c not in word2idx:
            outputs.append(
                {
                    "analogy": [a, b, c],
                    "predictions": [],
                    "skipped": True,
                    "reason": "one or more tokens missing from vocabulary",
                }
            )
            continue

        target = emb[word2idx[b]] - emb[word2idx[a]] + emb[word2idx[c]]
        target_norm = target / (np.linalg.norm(target) + 1e-12)
        sims = emb @ target_norm

        for token in (a, b, c):
            sims[word2idx[token]] = -math.inf

        best = np.argsort(-sims)[:top_k]
        preds = [{"word": idx2word[int(i)], "cosine": float(sims[int(i)])} for i in best]
        outputs.append({"analogy": [a, b, c], "predictions": preds, "skipped": False})

    return outputs


def compute_mrr(
    embedding_matrix: np.ndarray,
    word2idx: dict[str, int],
    labeled_pairs: list[tuple[str, str]],
) -> float:
    """Compute MRR over query-target synonym/relatedness pairs."""
    emb = _normalize_rows(embedding_matrix.astype(np.float32))
    reciprocal_ranks: list[float] = []

    for query, target in labeled_pairs:
        if query not in word2idx or target not in word2idx:
            continue

        q_idx = word2idx[query]
        t_idx = word2idx[target]

        scores = emb @ emb[q_idx]
        scores[q_idx] = -math.inf
        ranking = np.argsort(-scores)

        rank_positions = np.where(ranking == t_idx)[0]
        if rank_positions.size > 0:
            reciprocal_ranks.append(1.0 / float(rank_positions[0] + 1))

    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))


def plot_loss_curves(loss_by_condition: dict[str, list[float]], output_path: str | Path) -> None:
    """Plot one or more training loss curves."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, losses in loss_by_condition.items():
        if not losses:
            continue
        xs = np.arange(1, len(losses) + 1)
        plt.plot(xs, losses, marker="o", label=label)

    plt.title("Skip-gram Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
