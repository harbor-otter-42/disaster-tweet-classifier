"""Local Streamlit UI for the disaster-tweet classifier.

Run with:  streamlit run app.py
Opens on http://localhost:8501. A non-technical user can classify a single
message (with an explanation of *why*) or upload a CSV for batch scoring.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from src.model import DisasterModel, Prediction

MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
MAX_CHARS = 1000
EXAMPLES = [
    "Forest fire near La Ronge Sask. Canada",
    "7.1 magnitude earthquake off the coast, tsunami warning issued",
    "I just love how the sky looks like it's on fire tonight",
    "My new phone is the bomb, absolutely killing it today",
]


@st.cache_resource
def get_model() -> DisasterModel:
    """Load the model once per server process."""
    return DisasterModel.load(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict:
    """Load validation metrics for the model card, if present."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _set_example(text: str) -> None:
    """Populate the input box from an example button."""
    st.session_state.tweet = text


def render_result(pred: Prediction) -> None:
    """Render the label, confidence and driver explanation."""
    is_disaster = pred.label == 1
    confidence = pred.score if is_disaster else 1.0 - pred.score
    headline = "🟥 Real disaster" if is_disaster else "🟩 Not a real disaster"

    st.subheader(headline)
    st.progress(confidence, text=f"Confidence: {confidence:.0%}")
    st.caption(f"Model estimate of P(real disaster) = {pred.score:.2f}")

    with st.expander("Why this prediction?", expanded=True):
        if not pred.drivers:
            st.write("No strong word-level signals — the model is near its boundary.")
            return
        st.caption("Words pushing the call toward 🟥 disaster (red) or 🟩 not (green):")
        for word, weight in pred.drivers:
            colour = "#c0392b" if weight > 0 else "#27ae60"
            arrow = "→ disaster" if weight > 0 else "→ not"
            st.markdown(
                f"<span style='color:{colour};font-weight:600'>{word}</span> "
                f"<span style='color:#888'>({arrow}, weight {weight:+.2f})</span>",
                unsafe_allow_html=True,
            )


def single_tab(prefill: str | None) -> None:
    """Single-message classification with examples and explanation."""
    result_slot = st.container()  # results render at the top, above the controls

    st.write("Try an example:")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        column.button(example[:24] + "…", on_click=_set_example, args=(example,), help=example)

    text = st.text_area("Message text", key="tweet", max_chars=MAX_CHARS, height=120)
    if st.button("Classify", type="primary") or prefill:
        with result_slot:
            if not text or not text.strip():
                st.warning("Please enter some text to classify.")
            else:
                render_result(get_model().predict_one(text.strip()))


def batch_tab() -> None:
    """Upload a CSV with a 'text' column and score every row."""
    st.write("Upload a CSV with a **text** column to score many messages at once.")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return
    df = pd.read_csv(upload)
    if "text" not in df.columns:
        st.error(f"CSV needs a 'text' column; found {list(df.columns)}.")
        return
    texts = df["text"].fillna("").astype(str).tolist()
    scores = get_model().predict_proba(texts)
    result = pd.DataFrame(
        {"text": texts, "label": (scores >= 0.5).astype(int), "score": scores.round(4)}
    )
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Download predictions.csv", result.to_csv(index=False), "predictions.csv", "text/csv"
    )


def about() -> None:
    """A short model card: what it is, how it scored, and its limits."""
    m = get_metrics()
    with st.expander("About this model (model card)"):
        if m:
            st.write(
                f"**Validation** (20% hold-out, n={m.get('n_val', '—')}): "
                f"accuracy {m.get('accuracy')}, disaster-F1 {m.get('f1')}, "
                f"ROC-AUC {m.get('roc_auc')}, Brier {m.get('brier_score')}."
            )
        st.write(
            "Word + character TF-IDF features into a logistic-regression classifier, "
            "trained on ~7,600 labelled tweets. Linear by design so each prediction "
            "is explainable.\n\n"
            "**Use as decision-support, not an authority.** A human should confirm "
            "before any action. Weaker on sarcasm, metaphor, non-English text, and "
            "events newer than the training data; false negatives (missing a real "
            "disaster) are the costliest error and the reason confidence is always shown."
        )


def main() -> None:
    """Compose the page."""
    st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🌍")
    st.title("🌍 Disaster Tweet Classifier")
    st.write(
        "Estimate whether a short message describes a **real disaster**. "
        "Built as decision-support — a human should confirm before any action."
    )

    st.session_state.setdefault("tweet", "")
    prefill = st.query_params.get("text")
    if prefill and not st.session_state.tweet:
        st.session_state.tweet = prefill

    single, batch = st.tabs(["Single message", "Batch (CSV)"])
    with single:
        single_tab(prefill)
    with batch:
        batch_tab()

    about()
    st.divider()
    st.caption(
        "Linear TF-IDF + logistic-regression model. Predictions are probabilistic "
        "and can be wrong — always review confidence and rationale."
    )


if __name__ == "__main__":
    main()
