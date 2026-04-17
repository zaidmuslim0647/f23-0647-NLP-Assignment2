from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.annotation import (
    AnnotatedSentence,
    build_phrase_map,
    build_rule_lexicon,
    default_gazetteer,
    infer_topics_from_metadata,
    label_distribution,
    sample_balanced_sentences,
    stratified_split,
    tag_ner_bio,
    tag_pos,
    write_conll,
)
from utils.data_utils import load_corpus_lines, load_metadata, tokenize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Commit 4 dataset preparation")
    parser.add_argument("--cleaned-path", type=str, default="cleaned.txt")
    parser.add_argument("--metadata-path", type=str, default="Metadata.json")
    parser.add_argument("--total-sentences", type=int, default=500)
    parser.add_argument("--min-per-topic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pos-train-out", type=str, default="data/pos_train.conll")
    parser.add_argument("--pos-val-out", type=str, default="data/pos_val.conll")
    parser.add_argument("--pos-test-out", type=str, default="data/pos_test.conll")

    parser.add_argument("--ner-train-out", type=str, default="data/ner_train.conll")
    parser.add_argument("--ner-val-out", type=str, default="data/ner_val.conll")
    parser.add_argument("--ner-test-out", type=str, default="data/ner_test.conll")

    parser.add_argument(
        "--summary-out",
        type=str,
        default="data/dataset_annotation_summary.json",
    )
    return parser.parse_args()


def _save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _topic_dist(items: list[AnnotatedSentence]) -> dict[str, int]:
    counts = Counter(item.topic for item in items)
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()

    docs = load_corpus_lines(PROJECT_ROOT / args.cleaned_path)
    print(f"Loaded cleaned corpus documents: {len(docs)}")

    metadata_path = PROJECT_ROOT / args.metadata_path
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
        topics = infer_topics_from_metadata(metadata, num_docs=len(docs))
        print("Loaded metadata and inferred topics")
    else:
        topics = ["unknown"] * len(docs)
        print("Metadata not found; all documents assigned to topic='unknown'")

    sampled = sample_balanced_sentences(
        documents=docs,
        topics=topics,
        total_sentences=args.total_sentences,
        min_per_topic=args.min_per_topic,
        min_topics=3,
        seed=args.seed,
    )
    print(f"Sampled balanced sentence set: {len(sampled)}")

    token_counter: Counter[str] = Counter()
    for _, sentence in sampled:
        token_counter.update(tok.lower() for tok in tokenize_text(sentence, lowercase=False))

    lexicon = build_rule_lexicon(token_counter, min_per_major_tag=200)
    phrase_map = build_phrase_map(default_gazetteer())

    annotated: list[AnnotatedSentence] = []
    for topic, sentence in sampled:
        tokens = tokenize_text(sentence, lowercase=False)
        if not tokens:
            continue
        pos_tags = tag_pos(tokens, lexicon)
        ner_tags = tag_ner_bio(tokens, phrase_map)
        annotated.append(AnnotatedSentence(topic=topic, tokens=tokens, pos_tags=pos_tags, ner_tags=ner_tags))

    train_set, val_set, test_set = stratified_split(annotated, train_ratio=0.70, val_ratio=0.15, seed=args.seed)
    print(f"Split sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    write_conll(train_set, PROJECT_ROOT / args.pos_train_out, tag_type="pos")
    write_conll(val_set, PROJECT_ROOT / args.pos_val_out, tag_type="pos")
    write_conll(test_set, PROJECT_ROOT / args.pos_test_out, tag_type="pos")

    write_conll(train_set, PROJECT_ROOT / args.ner_train_out, tag_type="ner")
    write_conll(val_set, PROJECT_ROOT / args.ner_val_out, tag_type="ner")
    write_conll(test_set, PROJECT_ROOT / args.ner_test_out, tag_type="ner")

    summary = {
        "sampled_sentences": len(annotated),
        "topic_distribution_all": _topic_dist(annotated),
        "split_sizes": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "topic_distribution_split": {
            "train": _topic_dist(train_set),
            "val": _topic_dist(val_set),
            "test": _topic_dist(test_set),
        },
        "label_distribution_split": {
            "train": label_distribution(train_set),
            "val": label_distribution(val_set),
            "test": label_distribution(test_set),
        },
        "lexicon_size": len(lexicon),
        "gazetteer_entities": {
            "PER": len(default_gazetteer()["PER"]),
            "LOC": len(default_gazetteer()["LOC"]),
            "ORG": len(default_gazetteer()["ORG"]),
            "MISC": len(default_gazetteer()["MISC"]),
        },
    }
    _save_json(summary, PROJECT_ROOT / args.summary_out)

    print(f"Saved POS train/val/test to: {args.pos_train_out}, {args.pos_val_out}, {args.pos_test_out}")
    print(f"Saved NER train/val/test to: {args.ner_train_out}, {args.ner_val_out}, {args.ner_test_out}")
    print(f"Saved summary report to: {args.summary_out}")


if __name__ == "__main__":
    main()
