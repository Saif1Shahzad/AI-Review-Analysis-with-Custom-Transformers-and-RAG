# Transformer-based review understanding with a retrieval-augmented explanation generator.

## Project Summary

This assignment implements a three-stage pipeline:

1. **Encoder (Transformer)**: multi-task learning for sentiment (3 classes) and length category (3 classes) plus a pooled embedding.
2. **Retrieval (RAG)**: cosine similarity over normalized train embeddings to fetch top-k similar reviews.
3. **Decoder (Transformer)**: decoder-only model that generates a short explanation conditioned on the review, labels, and retrieved context.

The notebook trains models, evaluates classification performance, analyzes retrieval quality, runs a RAG vs No-RAG ablation, and saves plots to `results/`.

## Repository Layout

- `i232634-NLP-Assignment3.ipynb`: main notebook with all steps.
- `data/`: Amazon review JSON.gz files (not included).
- `models/`: saved checkpoints (`encoder_best.pt`, `decoder_best.pt`, `decoder_norag_best.pt`).
- `results/`: plots, logs, and embeddings (`train_embeddings.npy`, `hyperparameter_log.csv`, etc.).
- `report.pdf`: compiled report (if available).

## Setup

**Python**: 3.10+ recommended.

Install dependencies (example):

```
pip install numpy pandas matplotlib scikit-learn tqdm torch streamlit
```

## Data

Place the Amazon review files in `data/` with these names:

- `beauty.json.gz`
- `cellphones.json.gz`
- `electronics.json.gz`
- `home.json.gz`
- `sports.json.gz`

The notebook uses `sports`, `cellphones`, and `electronics` by default.

## How to Run (Reproducibility)

1. **Open the notebook** `i232634-NLP-Assignment3.ipynb`.
2. **Run cells in order** from top to bottom.
3. Outputs are saved under `models/` and `results/`.

Reproducibility notes:

- Seeds are set in the notebook (`SEED = 42` for Python, NumPy, and PyTorch).
- Results may still vary slightly between CPU and GPU due to nondeterminism in some operations.

### Expected Outputs

After a full run:

- `models/encoder_best.pt`
- `models/decoder_best.pt`
- `models/decoder_norag_best.pt`
- `results/train_embeddings.npy`
- `results/encoder_learning_curves.png`
- `results/decoder_learning_curves.png`
- `results/retrieval_k_analysis.png`
- `results/rag_ablation.png`
- `results/hyperparameter_log.csv`

## Streamlit Demo (Optional)

The notebook includes a cell that writes `app.py` for a Streamlit demo.

Run:

```
streamlit run app.py
```

The app expects the trained checkpoints and `results/train_embeddings.npy` to already exist.

## Notes

- If the dataset files are missing, the notebook will skip loading and downstream stages will not run.
- For faster runs, reduce `REVIEWS_PER_CATEGORY` or the number of epochs.
