from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def load_conll(path: str | Path) -> list[tuple[list[str], list[str]]]:
    """Load CoNLL data with one token/tag per line and blank-line sentence boundaries."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CoNLL file not found: {p}")

    data: list[tuple[list[str], list[str]]] = []
    tokens: list[str] = []
    tags: list[str] = []

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            if tokens:
                data.append((tokens, tags))
                tokens, tags = [], []
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        token = parts[0]
        tag = parts[-1]
        tokens.append(token)
        tags.append(tag)

    if tokens:
        data.append((tokens, tags))
    return data


def build_word_vocab(datasets: list[list[tuple[list[str], list[str]]]], min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for ds in datasets:
        for tokens, _ in ds:
            counter.update(tok.lower() for tok in tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq < min_freq:
            continue
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def build_tag_vocab(dataset: list[tuple[list[str], list[str]]], pad_token: str = PAD_TOKEN) -> dict[str, int]:
    tags = sorted({tag for _, tag_seq in dataset for tag in tag_seq})
    tag2idx = {pad_token: 0}
    for tag in tags:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
    return tag2idx


@dataclass
class SequenceExample:
    input_ids: list[int]
    tag_ids: list[int]
    tokens: list[str]


class SequenceDataset(Dataset):
    def __init__(
        self,
        data: list[tuple[list[str], list[str]]],
        word2idx: dict[str, int],
        tag2idx: dict[str, int],
    ) -> None:
        self.examples: list[SequenceExample] = []
        unk_id = word2idx[UNK_TOKEN]

        for tokens, tags in data:
            input_ids = [word2idx.get(tok.lower(), unk_id) for tok in tokens]
            tag_ids = [tag2idx[tag] for tag in tags]
            self.examples.append(SequenceExample(input_ids=input_ids, tag_ids=tag_ids, tokens=tokens))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SequenceExample:
        return self.examples[idx]


def collate_sequence_batch(
    batch: list[SequenceExample],
    pad_token_id: int = 0,
    pad_tag_id: int = 0,
) -> dict[str, torch.Tensor | list[list[str]]]:
    lengths = [len(ex.input_ids) for ex in batch]
    max_len = max(lengths)

    input_ids = []
    tag_ids = []
    mask = []
    tokens = []

    for ex in batch:
        pad_len = max_len - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_token_id] * pad_len)
        tag_ids.append(ex.tag_ids + [pad_tag_id] * pad_len)
        mask.append([1] * len(ex.input_ids) + [0] * pad_len)
        tokens.append(ex.tokens)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "tokens": tokens,
    }


def flatten_masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_flat = y_true[mask]
    pred_flat = y_pred[mask]
    return true_flat, pred_flat


def pos_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    pad_tag_idx: int,
) -> dict[str, object]:
    true_flat, pred_flat = flatten_masked(y_true, y_pred, mask)

    if true_flat.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "confusion_matrix": []}

    accuracy = float((true_flat == pred_flat).mean())

    labels = sorted(set(int(x) for x in true_flat if int(x) != pad_tag_idx))
    macro = f1_score(true_flat, pred_flat, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(true_flat, pred_flat, labels=labels)

    return {
        "accuracy": accuracy,
        "macro_f1": float(macro),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    tick_idx = np.arange(len(labels))
    plt.xticks(tick_idx, labels, rotation=45, ha="right")
    plt.yticks(tick_idx, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_losses(train_losses: list[float], val_losses: list[float], output_path: str | Path, title: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, val_losses, marker="o", label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def bio_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            ent_type = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == f"I-{ent_type}":
                i += 1
            spans.add((ent_type, start, i - 1))
            continue
        i += 1
    return spans


def ner_entity_metrics(true_tags: list[list[str]], pred_tags: list[list[str]]) -> dict[str, object]:
    by_type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gold_seq, pred_seq in zip(true_tags, pred_tags):
        gold_spans = bio_spans(gold_seq)
        pred_spans = bio_spans(pred_seq)

        for span in pred_spans:
            ent_type = span[0]
            if span in gold_spans:
                by_type_counts[ent_type]["tp"] += 1
            else:
                by_type_counts[ent_type]["fp"] += 1

        for span in gold_spans:
            ent_type = span[0]
            if span not in pred_spans:
                by_type_counts[ent_type]["fn"] += 1

    metrics: dict[str, object] = {}
    all_tp = all_fp = all_fn = 0

    for ent_type, counts in sorted(by_type_counts.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        all_tp += tp
        all_fp += fp
        all_fn += fn

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        metrics[ent_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    overall_p = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    overall_r = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if overall_p + overall_r else 0.0

    metrics["overall"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
    }
    return metrics
