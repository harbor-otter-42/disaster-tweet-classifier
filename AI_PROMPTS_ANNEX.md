# Annex — AI Tool Disclosure & Prompt Record

This annex is provided per the assessment's AI-use disclosure requirement.

## Tool and model

- **Tool:** Claude Code (Anthropic's agentic CLI)
- **Model:** Claude Opus 4.8 (`claude-opus-4-8`)
- **Division of labour:** I directed the build and made every engineering
  decision; AI was an implementation accelerator. The decisions — model choice
  (and the reasoning against a transformer/LLM), feature design, evaluation
  metrics, explainability, the responsible-AI/decision-support framing, and the
  anonymisation + reproducibility requirements — are mine. AI generated code and
  prose to my specifications and assisted with debugging. I reviewed every file,
  ran the training, the tests, and the clean-environment verification myself.

## What the AI relied upon

- **Dataset:** _Natural Language Processing with Disaster Tweets_ (Kaggle
  `nlp-getting-started`), the dataset named in the assessment. `train.csv` /
  `test.csv` were obtained from a public GitHub mirror of that competition data
  for reproducibility (7,613 labelled rows; columns `id,keyword,location,text,
target`; matches the canonical dataset).
- **Libraries (standard public APIs, no code copied verbatim):** scikit-learn
  (`TfidfVectorizer`, `LogisticRegression`, metrics), Streamlit, pandas, scipy,
  joblib, Pillow, pytest.
- No third-party source code, blog posts, or forum answers were copied into the
  submission.

## What I validated or changed (not accepted blindly)

- Ran training and confirmed the hold-out metrics that appear in the README.
- Ran a full clean-virtual-environment install from the pinned `requirements.txt`
  and verified the app, the batch CLI, and a from-scratch retrain work end-to-end.
- Ran the pytest suite (7 tests over cleaning, model, and CLI contract).
- Tested the CLI's error handling on malformed input.
- Decided word+char TF-IDF + logistic regression for the engineering reasons in
  the write-up; designed the explainability panel, the calibration check (Brier),
  and the decision-support framing.

## Record of prompts

A faithful, anonymised record of the substantive instructions I gave (identifying
details removed per the assessment rules):

1. "Build a binary classifier that predicts whether a short text describes a real
   disaster, wrapped in a simple local web UI runnable with one command, on the
   Kaggle disaster-tweets dataset. Plan it first."
2. "Read the assessment brief and extract the exact deliverables and constraints
   before writing code."
3. "Recommend a model that trains in under 5 minutes on CPU, runs offline on a
   fresh machine, and is explainable; justify it against transformer and
   zero-shot-LLM alternatives."
4. "Implement conservative tweet cleaning; a word + character TF-IDF
   logistic-regression model in a typed, documented module; and a `train.py` that
   saves the artifact plus honest hold-out metrics (accuracy, F1, ROC-AUC, Brier,
   confusion matrix)."
5. "Add per-prediction explainability from the linear model's coefficients."
6. "Build a Streamlit UI: single-message classification with label + confidence +
   the explanation, a batch CSV tab, a model card, graceful empty-input handling,
   and a decision-support disclaimer."
7. "Write `predict.py` as a CLI (`--input/--output`, CSV with a `text` column ->
   `text,label,score`) with graceful error handling."
8. "Add a pytest suite for the cleaning, model, and CLI contract."
9. "Pin dependencies and the Python version; verify the whole thing in a fresh
   virtual environment (clone -> venv -> install -> run) and fix anything broken."
10. "Produce an anonymity-safe screenshot at localhost with the address bar and a
    sample prediction visible; ensure no real name, employer, project names, or
    identifying metadata appear anywhere in the repo, commits, or files."
11. "Write the README, the one-page write-up, and this AI-disclosure annex."

> This is a readable summary of the directing prompts rather than a verbatim
> keystroke log; it accurately reflects the instructions that shaped the work.
