"""Disaster-tweet classifier: word + character TF-IDF into logistic regression.

The model is intentionally linear so that every prediction is explainable: the
contribution of each word is simply its TF-IDF weight times the learned
coefficient. That makes the UI able to show *why* a tweet was flagged, which
matters more for a humanitarian triage tool than a fractional F1 gain from a
black-box transformer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text_clean import clean_text

LABELS = {0: "Not a real disaster", 1: "Real disaster"}
_TOP_K_WORDS = 6


def build_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """Return the (word, char) TF-IDF vectorizers used as model features."""
    word_vec = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
    )
    return word_vec, char_vec


@dataclass
class Prediction:
    """One classifier decision plus its human-readable rationale."""

    label: int
    label_text: str
    score: float
    drivers: list[tuple[str, float]]


class DisasterModel:
    """Thin container that fits/serves the word+char TF-IDF logistic model."""

    def __init__(self) -> None:
        self.word_vec, self.char_vec = build_vectorizers()
        self.clf = LogisticRegression(
            C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear"
        )
        self._n_word_features = 0

    def _features(self, texts: list[str], fit: bool) -> csr_matrix:
        """Vectorise texts; fit the vectorizers when ``fit`` is True."""
        if fit:
            word = self.word_vec.fit_transform(texts)
            char = self.char_vec.fit_transform(texts)
            self._n_word_features = word.shape[1]
        else:
            word = self.word_vec.transform(texts)
            char = self.char_vec.transform(texts)
        return hstack([word, char]).tocsr()

    def fit(self, texts: list[str], labels: list[int]) -> "DisasterModel":
        """Train the vectorizers and classifier on raw tweet text."""
        self.clf.fit(self._features(texts, fit=True), labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return P(real disaster) for each input text."""
        return self.clf.predict_proba(self._features(texts, fit=False))[:, 1]

    def explain(self, text: str) -> list[tuple[str, float]]:
        """Top word features that pushed this single prediction, signed."""
        word = self.word_vec.transform([text])
        coef = self.clf.coef_[0][: self._n_word_features]
        contributions = word.multiply(coef).tocoo()
        names = self.word_vec.get_feature_names_out()
        pairs = [(names[c], float(v)) for c, v in zip(contributions.col, contributions.data)]
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return pairs[:_TOP_K_WORDS]

    def predict_one(self, text: str) -> Prediction:
        """Classify a single tweet and attach its rationale."""
        score = float(self.predict_proba([text])[0])
        label = int(score >= 0.5)
        return Prediction(label, LABELS[label], score, self.explain(text))

    def save(self, path: str) -> None:
        """Persist the whole model to a single joblib artifact."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DisasterModel":
        """Load a model artifact produced by :meth:`save`."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found at {path!r}. Run `python train.py` first."
            )
        return joblib.load(path)
