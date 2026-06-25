"""Tests for the cleaning, model, and CLI contracts.

Run with:  pytest
The model fixture trains a small in-memory model so the suite needs no artifact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predict import run
from src.model import DisasterModel
from src.text_clean import clean_text


@pytest.fixture(scope="module")
def model() -> DisasterModel:
    """A tiny model trained on a handful of clear-cut examples."""
    texts = [
        "huge earthquake destroys city, hundreds feared dead",
        "wildfire forces mass evacuation near the coast",
        "flood emergency declared, rescue teams deployed",
        "i love this sunny weather today",
        "my new phone is the bomb, killing it",
        "great party last night with friends",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    return DisasterModel().fit(texts, labels)


def test_clean_text_replaces_urls_and_mentions() -> None:
    out = clean_text("Flood @gov check http://t.co/x #emergency")
    assert "httpurl" in out and "usermention" in out
    assert "http://t.co/x" not in out


def test_clean_text_handles_non_string() -> None:
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_prediction_shape_and_range(model: DisasterModel) -> None:
    pred = model.predict_one("massive earthquake hits the region")
    assert pred.label in (0, 1)
    assert 0.0 <= pred.score <= 1.0
    assert pred.label_text


def test_disaster_scores_higher_than_chitchat(model: DisasterModel) -> None:
    hot = model.predict_proba(["wildfire emergency evacuation deaths reported"])[0]
    cold = model.predict_proba(["having a lovely coffee this morning"])[0]
    assert hot > cold


def test_explanation_returns_signed_drivers(model: DisasterModel) -> None:
    drivers = model.explain("earthquake destroys city")
    assert isinstance(drivers, list)
    assert all(isinstance(w, str) and isinstance(s, float) for w, s in drivers)


def test_predict_cli_writes_expected_columns(tmp_path, model: DisasterModel) -> None:
    model_path = tmp_path / "m.joblib"
    model.save(str(model_path))
    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["earthquake kills many", "nice weather"]}).to_csv(src, index=False)
    dst = tmp_path / "out.csv"

    n = run(str(src), str(dst), str(model_path))

    out = pd.read_csv(dst)
    assert n == 2
    assert list(out.columns) == ["text", "label", "score"]
    assert set(out["label"]).issubset({0, 1})


def test_predict_cli_rejects_missing_text_column(tmp_path, model: DisasterModel) -> None:
    model_path = tmp_path / "m.joblib"
    model.save(str(model_path))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        run(str(bad), str(tmp_path / "o.csv"), str(model_path))
