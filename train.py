"""Train the disaster-tweet classifier and write model.joblib + metrics.json.

Runs in a few seconds on CPU. A stratified hold-out split is used purely to
report honest validation metrics; the artifact that ships is retrained on the
full labelled set so the served model uses all available data.
"""

from __future__ import annotations

import json
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.model import DisasterModel

TRAIN_CSV = "data/train.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
RANDOM_STATE = 42


def _evaluate() -> dict:
    """Train on a stratified split and return validation metrics."""
    df = pd.read_csv(TRAIN_CSV)
    x_train, x_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["target"].tolist(),
        test_size=0.2,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )
    model = DisasterModel().fit(x_train, y_train)
    proba = model.predict_proba(x_val)
    preds = (proba >= 0.5).astype(int)
    return {
        "n_train": len(x_train),
        "n_val": len(x_val),
        "accuracy": round(accuracy_score(y_val, preds), 4),
        "f1": round(f1_score(y_val, preds), 4),
        "roc_auc": round(roc_auc_score(y_val, proba), 4),
        "brier_score": round(brier_score_loss(y_val, proba), 4),
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
        "report": classification_report(
            y_val, preds, target_names=["not_disaster", "disaster"], output_dict=True
        ),
    }


def main() -> None:
    """Evaluate, then fit the shipping model on all data and persist it."""
    start = time.perf_counter()
    metrics = _evaluate()

    df = pd.read_csv(TRAIN_CSV)
    DisasterModel().fit(df["text"].tolist(), df["target"].tolist()).save(MODEL_PATH)

    metrics["train_seconds"] = round(time.perf_counter() - start, 1)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved {MODEL_PATH} and {METRICS_PATH} in {metrics['train_seconds']}s")
    print(
        f"  val accuracy={metrics['accuracy']}  f1={metrics['f1']}  "
        f"roc_auc={metrics['roc_auc']}  brier={metrics['brier_score']}"
    )


if __name__ == "__main__":
    main()
