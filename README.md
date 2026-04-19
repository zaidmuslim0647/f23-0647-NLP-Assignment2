# CS-4063 NLP Assignment 2 (Commits 1-7)

This repository contains an end-to-end PyTorch implementation for Assignment 2 (word representations, sequence labeling, and transformer topic classification) with incremental scripts aligned to commit milestones.

## Repository Layout

```text
f23-0647_Assignment2_DS-C/
├── README.md
├── requirements.txt
├── f23-0647_Assignment2_DS-C.ipynb
├── embeddings/
├── models/
├── data/
├── scripts/
│   ├── build_vocab.py
│   ├── run_commit2_part1.py
│   ├── run_commit3_part1.py
│   ├── run_commit4_part2.py
│   ├── run_commit5_part2.py
│   ├── run_commit6_part3.py
│   └── run_commit7_finalize.py
└── src/
    ├── models/
    │   ├── sequence_models.py
    │   └── transformer_classifier.py
    └── utils/
        ├── annotation.py
        ├── data_utils.py
        ├── embeddings.py
        ├── sequence_labeling.py
        ├── topic_classification.py
        ├── vocab.py
        └── word2vec.py
```

## Expected Input Files

Place these files in the project root (or pass explicit paths):

- `cleaned.txt`
- `raw.txt`
- `Metadata.json`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commit-Wise Run Commands

### Commit 1 (Vocabulary)

```bash
python -m scripts.build_vocab \
  --cleaned-path cleaned.txt \
  --output-json embeddings/word2idx.json \
  --max-vocab-size 10000
```

### Commit 2 (TF-IDF + PPMI)

```bash
python -m scripts.run_commit2_part1 \
  --cleaned-path cleaned.txt \
  --metadata-path Metadata.json \
  --word2idx-path embeddings/word2idx.json
```

### Commit 3 (Skip-gram Word2Vec + Evaluation + C1-C4 Comparison)

```bash
python -m scripts.run_commit3_part1 \
  --cleaned-path cleaned.txt \
  --raw-path raw.txt \
  --word2idx-path embeddings/word2idx.json \
  --ppmi-path embeddings/ppmi_matrix.npy \
  --epochs 5 \
  --batch-size 512
```

Outputs include:

- `embeddings/embeddings_w2v.npy`
- `embeddings/w2v_nearest_neighbors.json`
- `embeddings/w2v_analogies.json`
- `embeddings/w2v_four_condition_comparison.json`
- `embeddings/w2v_loss_curve.png`

### Commit 4 (POS/NER Dataset Preparation)

```bash
python -m scripts.run_commit4_part2 \
  --cleaned-path cleaned.txt \
  --metadata-path Metadata.json \
  --total-sentences 500 \
  --min-per-topic 100
```

Outputs include train/val/test CoNLL files:

- `data/pos_train.conll`, `data/pos_val.conll`, `data/pos_test.conll`
- `data/ner_train.conll`, `data/ner_val.conll`, `data/ner_test.conll`
- `data/dataset_annotation_summary.json`

### Commit 5 (BiLSTM + CRF + Evaluation + Ablations)

```bash
python -m scripts.run_commit5_part2 \
  --pos-train data/pos_train.conll \
  --pos-val data/pos_val.conll \
  --pos-test data/pos_test.conll \
  --ner-train data/ner_train.conll \
  --ner-val data/ner_val.conll \
  --ner-test data/ner_test.conll
```

Outputs include:

- `models/bilstm_pos.pt`
- `models/bilstm_ner.pt`
- `models/pos_metrics.json`
- `models/ner_metrics.json`
- `models/ablation_results.json`

### Commit 6 (Custom Transformer Encoder for Topic Classification)

```bash
python -m scripts.run_commit6_part3 \
  --cleaned-path cleaned.txt \
  --metadata-path Metadata.json \
  --epochs 20 \
  --warmup-steps 50
```

Outputs include:

- `models/transformer_cls.pt`
- `models/transformer_metrics.json`
- `models/transformer_loss_curve.png`
- `models/transformer_accuracy_curve.png`
- `models/transformer_confusion_matrix.png`
- `models/attention_heatmaps/*.png`

### Commit 7 (Final Cleanup & Submission Checks)

```bash
python -m scripts.run_commit7_finalize \
  --notebook-path f23-0647_Assignment2_DS-C.ipynb \
  --report-path report.pdf
```

This creates:

- `data/submission_checklist.json`

## Notes

- The implementation is pure PyTorch and uses custom attention/encoder code (no `nn.Transformer*` modules).
- Re-run the notebook end-to-end before submission so all code cells have outputs.
- Ensure `report.pdf` is generated and present before zipping the final submission.
