from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.sequence_models import NERTaggerCRF, NERTaggerSoftmax, POSTagger
from utils.sequence_labeling import (
    PAD_TOKEN,
    build_tag_vocab,
    build_word_vocab,
    collate_sequence_batch,
    load_conll,
    ner_entity_metrics,
    plot_confusion_matrix,
    plot_losses,
    pos_metrics,
    SequenceDataset,
)
from utils.vocab import load_word2idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Commit 5 sequence labeling models")

    parser.add_argument("--pos-train", type=str, default="data/pos_train.conll")
    parser.add_argument("--pos-val", type=str, default="data/pos_val.conll")
    parser.add_argument("--pos-test", type=str, default="data/pos_test.conll")

    parser.add_argument("--ner-train", type=str, default="data/ner_train.conll")
    parser.add_argument("--ner-val", type=str, default="data/ner_val.conll")
    parser.add_argument("--ner-test", type=str, default="data/ner_test.conll")

    parser.add_argument("--word2idx-path", type=str, default="embeddings/word2idx.json")
    parser.add_argument("--pretrained-embeddings", type=str, default="embeddings/embeddings_w2v.npy")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pos-model-out", type=str, default="models/bilstm_pos.pt")
    parser.add_argument("--ner-model-out", type=str, default="models/bilstm_ner.pt")
    parser.add_argument("--pos-metrics-out", type=str, default="models/pos_metrics.json")
    parser.add_argument("--ner-metrics-out", type=str, default="models/ner_metrics.json")
    parser.add_argument("--ablation-out", type=str, default="models/ablation_results.json")

    parser.add_argument(
        "--pos-confusion-out",
        type=str,
        default="models/pos_confusion_matrix.png",
    )
    parser.add_argument(
        "--pos-loss-plot-out",
        type=str,
        default="models/pos_loss_curve.png",
    )
    parser.add_argument(
        "--ner-loss-plot-out",
        type=str,
        default="models/ner_loss_curve.png",
    )
    return parser.parse_args()


@dataclass
class TrainOutput:
    model_state: dict[str, torch.Tensor]
    train_losses: list[float]
    val_losses: list[float]
    best_val_score: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_embedding_matrix(
    vocab: dict[str, int],
    repo_word2idx_path: Path,
    repo_embedding_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor | None, int]:
    if not repo_word2idx_path.exists() or not repo_embedding_path.exists():
        return None, 100

    ext_word2idx = load_word2idx(repo_word2idx_path)
    ext_emb = np.load(repo_embedding_path)
    emb_dim = int(ext_emb.shape[1])

    matrix = np.random.normal(0.0, 0.02, size=(len(vocab), emb_dim)).astype(np.float32)
    shared = 0
    for tok, idx in vocab.items():
        ext_idx = ext_word2idx.get(tok)
        if ext_idx is None:
            continue
        if ext_idx >= ext_emb.shape[0]:
            continue
        matrix[idx] = ext_emb[ext_idx]
        shared += 1

    print(f"Loaded pretrained embeddings for {shared}/{len(vocab)} vocabulary tokens")
    return torch.tensor(matrix, dtype=torch.float32, device=device), emb_dim


def train_pos(
    model: POSTagger,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tag_pad_idx: int,
    idx2tag: dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> TrainOutput:
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)

    best_state = deepcopy(model.state_dict())
    best_score = -1.0
    no_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc=f"POS epoch {epoch + 1}/{args.epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tag_ids.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_count = int(mask.sum().item())
            running += float(loss.item()) * batch_count
            seen += batch_count

        train_loss = running / max(1, seen)
        train_losses.append(train_loss)

        val_loss, val_macro_f1 = eval_pos_loss_and_f1(model, val_loader, tag_pad_idx, device)
        val_losses.append(val_loss)

        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    return TrainOutput(
        model_state=best_state,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_score=best_score,
    )


def eval_pos_loss_and_f1(
    model: POSTagger,
    loader: DataLoader,
    tag_pad_idx: int,
    device: torch.device,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tag_ids.view(-1))

            token_count = int(mask.sum().item())
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count

            preds = logits.argmax(dim=-1)
            all_true.append(tag_ids.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    mask = np.concatenate(all_mask, axis=0).astype(bool)

    metrics = pos_metrics(y_true, y_pred, mask, pad_tag_idx=tag_pad_idx)
    return total_loss / max(1, total_tokens), float(metrics["macro_f1"])


def evaluate_pos_model(
    model: POSTagger,
    loader: DataLoader,
    tag_pad_idx: int,
    idx2tag: dict[int, str],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            logits = model(input_ids, lengths)
            preds = logits.argmax(dim=-1)

            all_true.append(tag_ids.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    mask = np.concatenate(all_mask, axis=0).astype(bool)
    metrics = pos_metrics(y_true, y_pred, mask, pad_tag_idx=tag_pad_idx)

    labels_idx = metrics.get("labels", [])
    labels_text = [idx2tag[i] for i in labels_idx]
    metrics["labels_text"] = labels_text
    return metrics


def train_ner_crf(
    model: NERTaggerCRF,
    train_loader: DataLoader,
    val_loader: DataLoader,
    idx2tag: dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> TrainOutput:
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = deepcopy(model.state_dict())
    best_score = -1.0
    no_improve = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc=f"NER-CRF epoch {epoch + 1}/{args.epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            loss = model.loss(input_ids, lengths, tag_ids, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_count = int(mask.sum().item())
            running += float(loss.item()) * batch_count
            seen += batch_count

        train_loss = running / max(1, seen)
        train_losses.append(train_loss)

        val_loss, val_f1 = eval_ner_crf_loss_and_f1(model, val_loader, idx2tag, device)
        val_losses.append(val_loss)

        if val_f1 > best_score:
            best_score = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    return TrainOutput(
        model_state=best_state,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_score=best_score,
    )


def eval_ner_crf_loss_and_f1(
    model: NERTaggerCRF,
    loader: DataLoader,
    idx2tag: dict[int, str],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    true_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            loss = model.loss(input_ids, lengths, tag_ids, mask)
            total_loss += float(loss.item()) * int(mask.sum().item())
            total_tokens += int(mask.sum().item())

            pred_ids = model.decode(input_ids, lengths, mask)
            for b_idx, pred_seq in enumerate(pred_ids):
                seq_len = int(lengths[b_idx].item())
                gold = tag_ids[b_idx, :seq_len].tolist()
                true_tags.append([idx2tag[t] for t in gold])
                pred_tags.append([idx2tag[t] for t in pred_seq])

    metrics = ner_entity_metrics(true_tags, pred_tags)
    return total_loss / max(1, total_tokens), float(metrics["overall"]["f1"])


def evaluate_ner_crf(
    model: NERTaggerCRF,
    loader: DataLoader,
    idx2tag: dict[int, str],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    true_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            pred_ids = model.decode(input_ids, lengths, mask)
            for b_idx, pred_seq in enumerate(pred_ids):
                seq_len = int(lengths[b_idx].item())
                gold = tag_ids[b_idx, :seq_len].tolist()
                true_tags.append([idx2tag[t] for t in gold])
                pred_tags.append([idx2tag[t] for t in pred_seq])

    return ner_entity_metrics(true_tags, pred_tags)


def train_and_eval_ner_softmax(
    model: NERTaggerSoftmax,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    tag_pad_idx: int,
    idx2tag: dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)

    best_state = deepcopy(model.state_dict())
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tag_ids.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_metrics = eval_ner_softmax(model, val_loader, idx2tag, device)
        val_f1 = float(val_metrics["overall"]["f1"])

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    return eval_ner_softmax(model, test_loader, idx2tag, device)


def eval_ner_softmax(
    model: NERTaggerSoftmax,
    loader: DataLoader,
    idx2tag: dict[int, str],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    true_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            logits = model(input_ids, lengths)
            pred_ids = logits.argmax(dim=-1)

            for b_idx in range(input_ids.size(0)):
                seq_len = int(lengths[b_idx].item())
                gold = tag_ids[b_idx, :seq_len].tolist()
                pred = pred_ids[b_idx, :seq_len].tolist()
                true_tags.append([idx2tag[t] for t in gold])
                pred_tags.append([idx2tag[t] for t in pred])

    return ner_entity_metrics(true_tags, pred_tags)


def build_loader(
    data: list[tuple[list[str], list[str]]],
    word2idx: dict[str, int],
    tag2idx: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = SequenceDataset(data=data, word2idx=word2idx, tag2idx=tag2idx)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_sequence_batch,
            pad_token_id=word2idx[PAD_TOKEN],
            pad_tag_id=tag2idx[PAD_TOKEN],
        ),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_train = load_conll(PROJECT_ROOT / args.pos_train)
    pos_val = load_conll(PROJECT_ROOT / args.pos_val)
    pos_test = load_conll(PROJECT_ROOT / args.pos_test)

    ner_train = load_conll(PROJECT_ROOT / args.ner_train)
    ner_val = load_conll(PROJECT_ROOT / args.ner_val)
    ner_test = load_conll(PROJECT_ROOT / args.ner_test)

    word2idx = build_word_vocab([pos_train, pos_val, pos_test, ner_train, ner_val, ner_test], min_freq=1)
    pos_tag2idx = build_tag_vocab(pos_train + pos_val + pos_test)
    ner_tag2idx = build_tag_vocab(ner_train + ner_val + ner_test)

    pos_idx2tag = {v: k for k, v in pos_tag2idx.items()}
    ner_idx2tag = {v: k for k, v in ner_tag2idx.items()}

    pretrained_matrix, embedding_dim = build_embedding_matrix(
        vocab=word2idx,
        repo_word2idx_path=PROJECT_ROOT / args.word2idx_path,
        repo_embedding_path=PROJECT_ROOT / args.pretrained_embeddings,
        device=device,
    )

    pos_train_loader = build_loader(pos_train, word2idx, pos_tag2idx, args.batch_size, shuffle=True)
    pos_val_loader = build_loader(pos_val, word2idx, pos_tag2idx, args.batch_size, shuffle=False)
    pos_test_loader = build_loader(pos_test, word2idx, pos_tag2idx, args.batch_size, shuffle=False)

    ner_train_loader = build_loader(ner_train, word2idx, ner_tag2idx, args.batch_size, shuffle=True)
    ner_val_loader = build_loader(ner_val, word2idx, ner_tag2idx, args.batch_size, shuffle=False)
    ner_test_loader = build_loader(ner_test, word2idx, ner_tag2idx, args.batch_size, shuffle=False)

    # POS: evaluate both frozen and fine-tuned embedding modes.
    pos_results: dict[str, object] = {}
    best_finetuned_state: dict[str, torch.Tensor] | None = None

    for freeze_embeddings in [True, False]:
        mode_key = "frozen_embeddings" if freeze_embeddings else "finetuned_embeddings"
        pos_model = POSTagger(
            vocab_size=len(word2idx),
            embedding_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_tags=len(pos_tag2idx),
            embedding_weights=pretrained_matrix,
            freeze_embeddings=freeze_embeddings,
            dropout=args.dropout,
            pad_idx=word2idx[PAD_TOKEN],
            bidirectional=True,
        ).to(device)

        out = train_pos(
            model=pos_model,
            train_loader=pos_train_loader,
            val_loader=pos_val_loader,
            tag_pad_idx=pos_tag2idx[PAD_TOKEN],
            idx2tag=pos_idx2tag,
            args=args,
            device=device,
        )
        pos_model.load_state_dict(out.model_state)
        metrics = evaluate_pos_model(
            model=pos_model,
            loader=pos_test_loader,
            tag_pad_idx=pos_tag2idx[PAD_TOKEN],
            idx2tag=pos_idx2tag,
            device=device,
        )

        pos_results[mode_key] = {
            "best_val_macro_f1": out.best_val_score,
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "labels": metrics["labels_text"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

        if not freeze_embeddings:
            best_finetuned_state = deepcopy(out.model_state)
            cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
            labels = [str(x) for x in metrics["labels_text"]]
            plot_confusion_matrix(
                cm=cm,
                labels=labels,
                output_path=PROJECT_ROOT / args.pos_confusion_out,
                title="POS Confusion Matrix (Fine-tuned Embeddings)",
            )
            plot_losses(
                train_losses=out.train_losses,
                val_losses=out.val_losses,
                output_path=PROJECT_ROOT / args.pos_loss_plot_out,
                title="POS Training vs Validation Loss",
            )

    if best_finetuned_state is not None:
        pos_model_save = {
            "state_dict": best_finetuned_state,
            "word2idx": word2idx,
            "tag2idx": pos_tag2idx,
            "embedding_dim": embedding_dim,
        }
        pos_path = PROJECT_ROOT / args.pos_model_out
        pos_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pos_model_save, pos_path)

    # NER with CRF.
    ner_model = NERTaggerCRF(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_tags=len(ner_tag2idx),
        embedding_weights=pretrained_matrix,
        freeze_embeddings=False,
        dropout=args.dropout,
        pad_idx=word2idx[PAD_TOKEN],
        bidirectional=True,
    ).to(device)

    ner_out = train_ner_crf(
        model=ner_model,
        train_loader=ner_train_loader,
        val_loader=ner_val_loader,
        idx2tag=ner_idx2tag,
        args=args,
        device=device,
    )
    ner_model.load_state_dict(ner_out.model_state)
    ner_metrics_crf = evaluate_ner_crf(ner_model, ner_test_loader, ner_idx2tag, device)

    ner_path = PROJECT_ROOT / args.ner_model_out
    ner_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": ner_out.model_state,
            "word2idx": word2idx,
            "tag2idx": ner_tag2idx,
            "embedding_dim": embedding_dim,
        },
        ner_path,
    )

    plot_losses(
        train_losses=ner_out.train_losses,
        val_losses=ner_out.val_losses,
        output_path=PROJECT_ROOT / args.ner_loss_plot_out,
        title="NER-CRF Training vs Validation Loss",
    )

    # NER baseline without CRF for required comparison.
    ner_softmax = NERTaggerSoftmax(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_tags=len(ner_tag2idx),
        embedding_weights=pretrained_matrix,
        freeze_embeddings=False,
        dropout=args.dropout,
        pad_idx=word2idx[PAD_TOKEN],
        bidirectional=True,
    ).to(device)

    ner_metrics_softmax = train_and_eval_ner_softmax(
        model=ner_softmax,
        train_loader=ner_train_loader,
        val_loader=ner_val_loader,
        test_loader=ner_test_loader,
        tag_pad_idx=ner_tag2idx[PAD_TOKEN],
        idx2tag=ner_idx2tag,
        args=args,
        device=device,
    )

    # Ablation study: each variant is run independently on the same split.
    ablation_results: dict[str, object] = {}

    # A1: unidirectional LSTM only.
    a1_model = NERTaggerCRF(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_tags=len(ner_tag2idx),
        embedding_weights=pretrained_matrix,
        freeze_embeddings=False,
        dropout=args.dropout,
        pad_idx=word2idx[PAD_TOKEN],
        bidirectional=False,
    ).to(device)
    a1_out = train_ner_crf(a1_model, ner_train_loader, ner_val_loader, ner_idx2tag, args, device)
    a1_model.load_state_dict(a1_out.model_state)
    ablation_results["A1_unidirectional_lstm"] = evaluate_ner_crf(
        a1_model, ner_test_loader, ner_idx2tag, device
    )

    # A2: no dropout.
    a2_model = NERTaggerCRF(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_tags=len(ner_tag2idx),
        embedding_weights=pretrained_matrix,
        freeze_embeddings=False,
        dropout=0.0,
        pad_idx=word2idx[PAD_TOKEN],
        bidirectional=True,
    ).to(device)
    a2_out = train_ner_crf(a2_model, ner_train_loader, ner_val_loader, ner_idx2tag, args, device)
    a2_model.load_state_dict(a2_out.model_state)
    ablation_results["A2_no_dropout"] = evaluate_ner_crf(a2_model, ner_test_loader, ner_idx2tag, device)

    # A3: random embeddings (no pretrained init).
    a3_model = NERTaggerCRF(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_tags=len(ner_tag2idx),
        embedding_weights=None,
        freeze_embeddings=False,
        dropout=args.dropout,
        pad_idx=word2idx[PAD_TOKEN],
        bidirectional=True,
    ).to(device)
    a3_out = train_ner_crf(a3_model, ner_train_loader, ner_val_loader, ner_idx2tag, args, device)
    a3_model.load_state_dict(a3_out.model_state)
    ablation_results["A3_random_embeddings"] = evaluate_ner_crf(a3_model, ner_test_loader, ner_idx2tag, device)

    # A4: softmax output instead of CRF.
    ablation_results["A4_softmax_instead_of_crf"] = ner_metrics_softmax

    pos_report = {
        "comparison": pos_results,
        "recommended_model": "finetuned_embeddings",
    }
    ner_report = {
        "with_crf": ner_metrics_crf,
        "without_crf": ner_metrics_softmax,
    }

    save_json(pos_report, PROJECT_ROOT / args.pos_metrics_out)
    save_json(ner_report, PROJECT_ROOT / args.ner_metrics_out)
    save_json(ablation_results, PROJECT_ROOT / args.ablation_out)

    print(f"Saved POS model: {args.pos_model_out}")
    print(f"Saved NER model: {args.ner_model_out}")
    print(f"Saved POS metrics: {args.pos_metrics_out}")
    print(f"Saved NER metrics: {args.ner_metrics_out}")
    print(f"Saved ablation report: {args.ablation_out}")


if __name__ == "__main__":
    main()
