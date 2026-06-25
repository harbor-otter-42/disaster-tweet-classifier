"""Local Streamlit UI for the disaster-tweet classifier.

Run with:  streamlit run app.py
Opens on http://localhost:8501 and lets a non-technical user paste a tweet and
get a label, a calibrated confidence, and a plain explanation of *why*.
"""

from __future__ import annotations

import streamlit as st

from src.model import DisasterModel, Prediction

MODEL_PATH = "model.joblib"
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
            st.write("No strong word-level signals — the model is near its decision boundary.")
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


def main() -> None:
    """Compose the page."""
    st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🌍")
    st.title("🌍 Disaster Tweet Classifier")
    st.write(
        "Paste a short message and the model estimates whether it describes a "
        "**real disaster**. Built as decision-support — a human should confirm "
        "before any action is taken."
    )

    st.session_state.setdefault("tweet", "")
    prefill = st.query_params.get("text")
    if prefill and not st.session_state.tweet:
        st.session_state.tweet = prefill

    st.write("Try an example:")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        column.button(
            example[:24] + "…", on_click=_set_example, args=(example,), help=example
        )

    text = st.text_area("Message text", key="tweet", max_chars=MAX_CHARS, height=120)

    if st.button("Classify", type="primary") or prefill:
        if not text or not text.strip():
            st.warning("Please enter some text to classify.")
        else:
            render_result(get_model().predict_one(text.strip()))

    st.divider()
    st.caption(
        "Linear TF-IDF + logistic-regression model trained on ~7,600 labelled "
        "tweets. Predictions are probabilistic and can be wrong, especially on "
        "sarcasm, metaphor, and events newer than the training data."
    )


if __name__ == "__main__":
    main()
