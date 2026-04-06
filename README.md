# CS-4063 NLP Assignment 2 (Commits 1-2)

This repository contains the implementation scaffold for Assignment 2: a BBC Urdu Neural NLP pipeline built in pure PyTorch.

## Commit 1 Scope

- Project folder structure created
- Data loading utilities added
- Vocabulary builder added (top 10,000 tokens + `<UNK>` handling)
- Basic script to generate `embeddings/word2idx.json`

## Commit 2 Scope

- TF-IDF matrix generation from `cleaned.txt`
- PPMI matrix generation with symmetric window size `k=5`
- t-SNE visualization for top 200 frequent tokens (`politics/sports/geography` legend)
- Top-k nearest neighbors (cosine on PPMI vectors) export
- Topic-discriminative top words export (if metadata topics align with corpus docs)

## Repository Layout

```text
f23-0647_Assignment2_DS-C/
├── README.md
├── requirements.txt
├── f23-0647_Assignment2_DS-C.ipynb
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── ppmi_tsne_top200.png
│   ├── ppmi_nearest_neighbors.json
│   └── top_discriminative_words_by_topic.json
├── models/
├── data/
├── scripts/
│   ├── __init__.py
│   ├── build_vocab.py
│   └── run_commit2_part1.py
└── src/
    ├── __init__.py
    └── utils/
        ├── __init__.py
        ├── data_utils.py
    ├── embeddings.py
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

## Run Commit 2 Pipeline (Part 1: TF-IDF + PPMI)

```bash
python -m scripts.run_commit2_part1 \
  --cleaned-path cleaned.txt \
  --metadata-path Metadata.json \
  --word2idx-path embeddings/word2idx.json \
  --tfidf-output embeddings/tfidf_matrix.npy \
  --ppmi-output embeddings/ppmi_matrix.npy \
  --neighbors-output embeddings/ppmi_nearest_neighbors.json \
  --tsne-output embeddings/ppmi_tsne_top200.png
```

Optional settings:

- `--max-vocab-size 10000`
- `--window-size 5`
- `--lowercase`

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

- Commit 3: Skip-gram Word2Vec + evaluation + ablation conditions
- Commit 4+: sequence labeling and transformer tasks
