from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.transformer_classifier import TransformerTopicClassifier
from utils.data_utils import load_corpus_lines, load_metadata
from utils.topic_classification import (
    ID_TO_TOPIC,
    TOPIC_TO_ID,
    TopicDataset,
    cls_metrics,
    collate_topic_batch,
    encode_documents,
    infer_topic,
    metadata_labels,
    plot_attention_heatmap,
    plot_confusion,
    plot_train_val_curves,
    stratified_indices,
    topic_distribution,
)
from utils.vocab import build_vocab_from_documents, load_word2idx, save_word2idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Commit 6 transformer classification pipeline")
    parser.add_argument("--cleaned-path", type=str, default="cleaned.txt")
    parser.add_argument("--metadata-path", type=str, default="Metadata.json")
    parser.add_argument("--word2idx-path", type=str, default="embeddings/word2idx.json")

    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-out", type=str, default="models/transformer_cls.pt")
    parser.add_argument("--metrics-out", type=str, default="models/transformer_metrics.json")
    parser.add_argument("--loss-plot-out", type=str, default="models/transformer_loss_curve.png")
    parser.add_argument("--acc-plot-out", type=str, default="models/transformer_accuracy_curve.png")
    parser.add_argument("--confusion-out", type=str, default="models/transformer_confusion_matrix.png")
    parser.add_argument(
        "--heatmap-dir",
        type=str,
        default="models/attention_heatmaps",
    )
    return parser.parse_args()


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    warmup = max(1, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)

        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate_classifier(
    model: TransformerTopicClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_items = 0
    true_y: list[int] = []
    pred_y: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total_items += labels.size(0)

            preds = logits.argmax(dim=-1)
            true_y.extend(labels.cpu().tolist())
            pred_y.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, total_items)
    accuracy = float(np.mean(np.asarray(true_y) == np.asarray(pred_y))) if true_y else 0.0
    return avg_loss, accuracy, true_y, pred_y


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    docs = load_corpus_lines(PROJECT_ROOT / args.cleaned_path)
    print(f"Loaded documents: {len(docs)}")

    metadata_path = PROJECT_ROOT / args.metadata_path
    meta_labels = [None] * len(docs)
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
        meta_labels = metadata_labels(metadata, len(docs))

    labels = [TOPIC_TO_ID[infer_topic(text, meta)] for text, meta in zip(docs, meta_labels)]

    word2idx_path = PROJECT_ROOT / args.word2idx_path
    if word2idx_path.exists():
        word2idx = load_word2idx(word2idx_path)
    else:
        word2idx, _ = build_vocab_from_documents(
            docs,
            max_vocab_size=10000,
            lowercase=False,
            include_pad=True,
            include_unk=True,
        )
        save_word2idx(word2idx, word2idx_path)

    examples = encode_documents(docs, labels, word2idx, max_len=args.max_len)

    train_idx, val_idx, test_idx = stratified_indices(labels, seed=args.seed)
    train_ds = TopicDataset([examples[i] for i in train_idx])
    val_ds = TopicDataset([examples[i] for i in val_idx])
    test_ds = TopicDataset([examples[i] for i in test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_topic_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_topic_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_topic_batch)

    model = TransformerTopicClassifier(
        vocab_size=len(word2idx),
        num_classes=5,
        d_model=128,
        num_layers=4,
        num_heads=4,
        dk=32,
        dv=32,
        d_ff=512,
        dropout=0.1,
        pad_idx=0,
        max_len=args.max_len + 1,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    best_state = model.state_dict()
    best_val_acc = -1.0

    step = 0
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0.0
        epoch_total = 0
        epoch_true: list[int] = []
        epoch_pred: list[int] = []

        for batch in tqdm(train_loader, desc=f"Transformer epoch {epoch + 1}/{args.epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=-1)
            epoch_true.extend(labels_b.cpu().tolist())
            epoch_pred.extend(preds.cpu().tolist())

            epoch_loss += float(loss.item()) * labels_b.size(0)
            epoch_total += labels_b.size(0)
            step += 1

        train_loss = epoch_loss / max(1, epoch_total)
        train_acc = float(np.mean(np.asarray(epoch_true) == np.asarray(epoch_pred))) if epoch_true else 0.0

        val_loss, val_acc, _, _ = evaluate_classifier(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate_classifier(model, test_loader, device)
    metrics = cls_metrics(y_true, y_pred)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    plot_confusion(cm, metrics["labels"], PROJECT_ROOT / args.confusion_out)
    plot_train_val_curves(
        train_vals=train_losses,
        val_vals=val_losses,
        title="Transformer Training vs Validation Loss",
        ylabel="Loss",
        output_path=PROJECT_ROOT / args.loss_plot_out,
    )
    plot_train_val_curves(
        train_vals=train_accs,
        val_vals=val_accs,
        title="Transformer Training vs Validation Accuracy",
        ylabel="Accuracy",
        output_path=PROJECT_ROOT / args.acc_plot_out,
    )

    # Attention heatmaps from final layer for at least 3 correctly classified examples.
    heatmap_dir = PROJECT_ROOT / args.heatmap_dir
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    correct_example_indices = [i for i, (a, b) in enumerate(zip(y_true, y_pred)) if a == b][:3]
    selected_heatmaps: list[dict[str, str]] = []

    if correct_example_indices:
        subset = [test_ds[i] for i in correct_example_indices]
        for sample_idx, sample in enumerate(subset, start=1):
            input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([sample.attention_mask], dtype=torch.bool, device=device)
            logits, all_attn = model(input_ids, attention_mask, return_attention=True)
            del logits

            if not all_attn:
                continue

            final_attn = all_attn[-1][0].detach().cpu().numpy()  # [H, T, T]
            token_count = int(sum(sample.attention_mask))
            token_labels = ["[CLS]"] + sample.tokens[:token_count]

            for head_idx in [0, 1]:
                if head_idx >= final_attn.shape[0]:
                    continue
                out_file = heatmap_dir / f"sample{sample_idx}_head{head_idx + 1}.png"
                plot_attention_heatmap(
                    attn_matrix=final_attn[head_idx],
                    token_labels=token_labels,
                    title=f"Final Layer Attention - Sample {sample_idx}, Head {head_idx + 1}",
                    output_path=out_file,
                )
                selected_heatmaps.append(
                    {
                        "sample": str(sample_idx),
                        "head": str(head_idx + 1),
                        "path": str(out_file.relative_to(PROJECT_ROOT)),
                    }
                )

    model_out = PROJECT_ROOT / args.model_out
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "word2idx": word2idx,
            "topic_to_id": TOPIC_TO_ID,
            "config": {
                "d_model": 128,
                "num_layers": 4,
                "num_heads": 4,
                "dk": 32,
                "dv": 32,
                "d_ff": 512,
                "max_len": args.max_len,
            },
        },
        model_out,
    )

    report = {
        "dataset_distribution": {
            "all": topic_distribution(labels),
            "train": topic_distribution([labels[i] for i in train_idx]),
            "val": topic_distribution([labels[i] for i in val_idx]),
            "test": topic_distribution([labels[i] for i in test_idx]),
        },
        "training": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_accuracy": train_accs,
            "val_accuracy": val_accs,
            "best_val_accuracy": best_val_acc,
        },
        "test": {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "labels": metrics["labels"],
        },
        "attention_heatmaps": selected_heatmaps,
    }
    save_json(report, PROJECT_ROOT / args.metrics_out)

    print(f"Saved Transformer model: {args.model_out}")
    print(f"Saved metrics report: {args.metrics_out}")
    print(f"Saved loss and accuracy plots: {args.loss_plot_out}, {args.acc_plot_out}")
    print(f"Saved confusion matrix: {args.confusion_out}")
    print(f"Saved attention heatmaps in: {args.heatmap_dir}")


if __name__ == "__main__":
    main()
