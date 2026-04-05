# CS-4063 NLP Assignment 2 (Commit 1 Setup)

This repository contains the implementation scaffold for Assignment 2: a BBC Urdu Neural NLP pipeline built in pure PyTorch.

## Commit 1 Scope

- Project folder structure created
- Data loading utilities added
- Vocabulary builder added (top 10,000 tokens + `<UNK>` handling)
- Basic script to generate `embeddings/word2idx.json`

## Repository Layout

```text
i23-XXXX_Assignment2_DS-X/
├── README.md
├── requirements.txt
├── i23-XXXX_Assignment2_DS-X.ipynb
├── embeddings/
├── models/
├── data/
├── scripts/
│   ├── __init__.py
│   └── build_vocab.py
└── src/
    ├── __init__.py
    └── utils/
        ├── __init__.py
        ├── data_utils.py
        └── vocab.py
```

## Expected Input Files

Place these input files in the project root or pass absolute paths via CLI:

- `cleaned.txt`
- `raw.txt`
- `Metadata.json`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Vocabulary (Commit 1 Utility)

```bash
python -m scripts.build_vocab \
  --cleaned-path cleaned.txt \
  --output-json embeddings/word2idx.json \
  --max-vocab-size 10000
```

Optional token frequency export:

```bash
python -m scripts.build_vocab \
  --cleaned-path cleaned.txt \
  --output-json embeddings/word2idx.json \
  --freq-output embeddings/token_freq.json
```

## Notes

- Vocabulary reserves index 0 for `<PAD>` and index 1 for `<UNK>`.
- Remaining tokens are selected by descending corpus frequency.
- Ties are broken alphabetically for reproducibility.

## Planned Next Commits

- Commit 2: TF-IDF + PPMI + t-SNE + neighbours
- Commit 3: Skip-gram Word2Vec + evaluation + ablation conditions
- Commit 4+: sequence labeling and transformer tasks
