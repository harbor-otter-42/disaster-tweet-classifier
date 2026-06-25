# Disaster Tweet Classifier

A small, fully local tool that predicts whether a short message describes a
**real disaster**. It ships as a one-command web app plus a batch CLI, and is
deliberately built on an **explainable linear model** so every prediction can
show _why_ it was made — which matters more for a humanitarian triage aid than
chasing a fractional F1 gain from a black box.

Trained on the public _Natural Language Processing with Disaster Tweets_
dataset (~7,600 hand-labelled tweets).

## Quickstart (one command to run)

Requires **Python 3.10** (see `.python-version`).

```bash
# 1. clone
git clone <your-repo-url>
cd disaster-tweet-classifier

# 2. set up an isolated environment + install pinned deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. launch the UI (opens http://localhost:8501 in your browser)
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

A trained model artifact (`model.joblib`) is committed, so the app works
immediately — no training step needed at launch.

### Retrain (optional, ~7 seconds on CPU)

```bash
python train.py
```

Writes `model.joblib` and `metrics.json`.

### Batch predictions (CLI)

```bash
python predict.py --input data/sample_tweets.csv --output predictions.csv
```

Reads any CSV with a `text` column and writes `text,label,score`
(`label` = 1 for real disaster, `score` = P(real disaster)). This lets a
reviewer score a held-out test set without clicking through the UI. The same
batch scoring is also available in the app's **Batch (CSV)** tab with a
download button.

### Tests

```bash
pytest
```

Covers the text cleaning, the model, and the CLI contract.

## What's in the box

| Path                | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| `app.py`            | Streamlit UI: single message (with explanation) + batch CSV tab |
| `predict.py`        | Batch CLI for scoring a CSV                                     |
| `train.py`          | Trains the model, writes `model.joblib` + `metrics.json`        |
| `src/model.py`      | Word+char TF-IDF -> logistic-regression model                   |
| `src/text_clean.py` | Conservative tweet normalisation                                |
| `tests/`            | pytest suite for cleaning, model, and CLI                       |
| `model.joblib`      | Committed trained artifact (the app runs with zero setup)       |
| `metrics.json`      | Hold-out validation metrics from the last training run          |
| `data/`             | Training data + a small sample CSV for the CLI                  |

## How it works

1. **Normalise** the text (URLs and @mentions -> placeholder tokens, HTML
   entities decoded, hashtags split). Casing and hashtag words are kept because
   they carry signal.
2. **Vectorise** with two TF-IDF representations: word 1-2 grams (topical
   signal) and character 3-5 grams (robust to the typos, hashtags, and spelling
   variants typical of tweets).
3. **Classify** with logistic regression (`class_weight="balanced"`). Because
   the model is linear, each word's contribution to a prediction is just its
   TF-IDF weight times its learned coefficient — that's exactly what the UI's
   _"Why this prediction?"_ panel shows.

## Performance (20% stratified hold-out)

| Metric                    | Value |
| ------------------------- | ----- |
| Accuracy                  | 0.813 |
| F1 (disaster class)       | 0.779 |
| ROC-AUC                   | 0.870 |
| Brier score (calibration) | 0.138 |

Confusion matrix `[[TN 737, FP 132], [FN 153, TP 501]]`. Per the brief, the goal
is working code and sensible choices, not state-of-the-art F1 — this baseline is
competitive while staying instant to train and fully interpretable.

## Limitations & responsible use

- It is a **decision-support tool, not an authority** — a human should confirm
  before any operational action. False negatives (missing a real disaster) are
  the costliest error in this domain, so the UI always exposes confidence and
  rationale rather than a bare yes/no.
- Trained on 2015-era English tweets; it will be weaker on sarcasm, metaphor,
  non-English text, and event types unseen in training.
- No personal data is used at train time beyond the public tweet text.

## Data attribution

Dataset: _Natural Language Processing with Disaster Tweets_
(Kaggle `nlp-getting-started`), used for non-commercial evaluation.
