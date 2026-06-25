"""Batch prediction CLI.

Usage:
    python predict.py --input tweets.csv --output predictions.csv

Reads a CSV containing a ``text`` column and writes a CSV with
``text, label, score`` (label is 1 for real disaster, score is P(disaster)).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.model import DisasterModel

MODEL_PATH = "model.joblib"
TEXT_COLUMN = "text"


def _parse_args() -> argparse.Namespace:
    """Parse --input/--output paths."""
    parser = argparse.ArgumentParser(description="Batch disaster-tweet classifier.")
    parser.add_argument("--input", required=True, help="CSV with a 'text' column")
    parser.add_argument("--output", required=True, help="destination CSV path")
    parser.add_argument("--model", default=MODEL_PATH, help="model artifact path")
    return parser.parse_args()


def run(input_path: str, output_path: str, model_path: str) -> int:
    """Score every row of ``input_path`` and write predictions; return row count."""
    df = pd.read_csv(input_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Input CSV must have a '{TEXT_COLUMN}' column; found {list(df.columns)}"
        )

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    model = DisasterModel.load(model_path)
    scores = model.predict_proba(df[TEXT_COLUMN].tolist())

    out = pd.DataFrame(
        {
            "text": df[TEXT_COLUMN],
            "label": (scores >= 0.5).astype(int),
            "score": scores.round(4),
        }
    )
    out.to_csv(output_path, index=False)
    return len(out)


def main() -> None:
    """Entry point with graceful, non-tracebacky error reporting."""
    args = _parse_args()
    try:
        n = run(args.input, args.output, args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {n} predictions to {args.output}")


if __name__ == "__main__":
    main()
