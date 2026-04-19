from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import Dataset

from .data_utils import tokenize_text


TOPIC_TO_ID = {
    "Politics": 0,
    "Sports": 1,
    "Economy": 2,
    "International": 3,
    "Health & Society": 4,
}

ID_TO_TOPIC = {v: k for k, v in TOPIC_TO_ID.items()}

TOPIC_KEYWORDS = {
    "Politics": {"election", "government", "minister", "parliament", "vote", "assembly", "hukumat"},
    "Sports": {"cricket", "match", "team", "player", "score", "tournament", "psl"},
    "Economy": {"inflation", "trade", "bank", "gdp", "budget", "dollar", "market", "maeeshat"},
    "International": {"un", "treaty", "foreign", "bilateral", "conflict", "diplomacy", "border"},
    "Health & Society": {"hospital", "disease", "vaccine", "flood", "education", "school", "sehat"},
}


@dataclass
class TopicExample:
    input_ids: list[int]
    attention_mask: list[int]
    label_id: int
    tokens: list[str]


class TopicDataset(Dataset):
    def __init__(self, examples: list[TopicExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TopicExample:
        return self.examples[idx]


def infer_topic(text: str, metadata_label: str | None = None) -> str:
    """Infer one of the five required categories using metadata and keyword fallback."""
    if metadata_label:
        label_l = metadata_label.lower()
        if any(k in label_l for k in ["polit", "gov", "minister", "parliament", "election"]):
            return "Politics"
        if any(k in label_l for k in ["sport", "cricket", "match"]):
            return "Sports"
        if any(k in label_l for k in ["econom", "trade", "bank", "budget", "business"]):
            return "Economy"
        if any(k in label_l for k in ["intern", "foreign", "global", "world", "diploma"]):
            return "International"
        if any(k in label_l for k in ["health", "society", "education", "flood"]):
            return "Health & Society"

    tokens = [t.lower() for t in tokenize_text(text, lowercase=True)]
    counts = {topic: 0 for topic in TOPIC_KEYWORDS}
    for tok in tokens:
        for topic, kws in TOPIC_KEYWORDS.items():
            if tok in kws:
                counts[topic] += 1

    best_topic = max(counts.items(), key=lambda x: x[1])[0]
    if counts[best_topic] == 0:
        return "Politics"
    return best_topic


def metadata_labels(metadata: object, n_docs: int) -> list[str | None]:
    """Schema-tolerant extraction of metadata labels aligned with documents."""

    def pick(item: dict) -> str | None:
        for key in ["topic", "category", "label", "section", "genre"]:
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    labels: list[str | None] = []
    if isinstance(metadata, list):
        for item in metadata:
            labels.append(pick(item) if isinstance(item, dict) else None)
    elif isinstance(metadata, dict):
        if isinstance(metadata.get("articles"), list):
            for item in metadata["articles"]:
                labels.append(pick(item) if isinstance(item, dict) else None)
        else:
            for value in metadata.values():
                labels.append(pick(value) if isinstance(value, dict) else None)

    if len(labels) < n_docs:
        labels.extend([None] * (n_docs - len(labels)))
    return labels[:n_docs]


def stratified_indices(labels: list[int], seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """Create a 70/15/15 stratified split over integer labels."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_label[y].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label_indices in by_label.values():
        ids = label_indices[:]
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_test = n - n_train - n_val

        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            while n_train + n_val + n_test > n:
                n_train = max(1, n_train - 1)

        train_idx.extend(ids[:n_train])
        val_idx.extend(ids[n_train : n_train + n_val])
        test_idx.extend(ids[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def encode_documents(
    docs: list[str],
    labels: list[int],
    word2idx: dict[str, int],
    max_len: int = 256,
) -> list[TopicExample]:
    unk = word2idx.get("<UNK>", 1)
    examples: list[TopicExample] = []

    for text, label in zip(docs, labels):
        tokens = tokenize_text(text, lowercase=False)
        ids = [word2idx.get(tok, unk) for tok in tokens[:max_len]]
        mask = [1] * len(ids)

        if len(ids) < max_len:
            pad = max_len - len(ids)
            ids.extend([0] * pad)
            mask.extend([0] * pad)

        examples.append(TopicExample(input_ids=ids, attention_mask=mask, label_id=label, tokens=tokens[:max_len]))

    return examples


def collate_topic_batch(batch: list[TopicExample]) -> dict[str, torch.Tensor | list[list[str]]]:
    input_ids = torch.tensor([ex.input_ids for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex.attention_mask for ex in batch], dtype=torch.bool)
    labels = torch.tensor([ex.label_id for ex in batch], dtype=torch.long)
    tokens = [ex.tokens for ex in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tokens": tokens,
    }


def topic_distribution(label_ids: list[int]) -> dict[str, int]:
    counter = Counter(label_ids)
    return {ID_TO_TOPIC[k]: int(v) for k, v in sorted(counter.items())}


def cls_metrics(true_y: list[int], pred_y: list[int]) -> dict[str, object]:
    acc = float(np.mean(np.asarray(true_y) == np.asarray(pred_y))) if true_y else 0.0
    macro = f1_score(true_y, pred_y, labels=sorted(ID_TO_TOPIC.keys()), average="macro", zero_division=0)
    cm = confusion_matrix(true_y, pred_y, labels=sorted(ID_TO_TOPIC.keys()))

    return {
        "accuracy": acc,
        "macro_f1": float(macro),
        "labels": [ID_TO_TOPIC[i] for i in sorted(ID_TO_TOPIC.keys())],
        "confusion_matrix": cm.tolist(),
    }


def plot_train_val_curves(
    train_vals: list[float],
    val_vals: list[float],
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = np.arange(1, len(train_vals) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, train_vals, marker="o", label="Train")
    plt.plot(xs, val_vals, marker="o", label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_confusion(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Oranges")
    plt.title("Transformer Topic Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    ids = np.arange(len(labels))
    plt.xticks(ids, labels, rotation=35, ha="right")
    plt.yticks(ids, labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    token_labels: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(token_labels), attn_matrix.shape[0], attn_matrix.shape[1], 24)
    labels = token_labels[:n]
    attn = attn_matrix[:n, :n]

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    ids = np.arange(n)
    plt.xticks(ids, labels, rotation=60, ha="right", fontsize=7)
    plt.yticks(ids, labels, fontsize=7)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
