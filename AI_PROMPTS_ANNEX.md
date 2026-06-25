# Annex — AI Tool Disclosure & Prompt Record

This annex is provided per the assessment's AI-use disclosure requirement.

## Tool and model

- **Tool:** Claude Code (Anthropic's agentic CLI)
- **Model:** Claude Opus 4.8 (`claude-opus-4-8`)
- **How used:** the entire repository (model code, UI, CLI, tests of the
  pipeline, README, and this write-up) was produced in a directed,
  human-in-the-loop session. I specified the approach and constraints, reviewed
  every file, ran and validated the outputs, and made the engineering decisions
  (notably choosing a linear model over a transformer/LLM). The AI generated code
  and prose to my direction and helped debug.

## What the AI relied upon

- **Dataset:** _Natural Language Processing with Disaster Tweets_ (Kaggle
  `nlp-getting-started`), the dataset named in the assessment. `train.csv` /
  `test.csv` were obtained from a public GitHub mirror of that competition data
  for reproducibility.
- **Libraries (standard public APIs, no code copied verbatim):** scikit-learn
  (`TfidfVectorizer`, `LogisticRegression`, metrics), Streamlit, pandas, scipy,
  joblib, Pillow.
- No third-party source code, blog posts, or Stack Overflow answers were copied
  into the submission.

## What I validated or changed (not accepted blindly)

- Ran training and confirmed the hold-out metrics that appear in the README.
- Ran a full clean-virtual-environment install from the pinned `requirements.txt`
  and verified the app, the batch CLI, and a from-scratch retrain all work.
- Tested the CLI's error handling on malformed input.
- Chose word+char TF-IDF + logistic regression for the engineering reasons given
  in the write-up; chose the explainability panel and the decision-support
  framing.

## Record of prompts

The following is a faithful, anonymised record of the substantive instructions I
gave during the session (identifying details removed per the assessment rules).

1. "Here is the technical assessment: build a binary classifier that predicts
   whether a short text describes a real disaster, wrapped in a simple local web
   UI runnable with one command. The dataset is the Kaggle disaster-tweets set.
   Let's research, plan, then build a strong submission."

2. "Read the actual assessment brief and extract the exact deliverables and
   constraints before doing anything else."

3. "Plan the build under a 2-hour budget. Recommend a model approach that
   trains in under 5 minutes on CPU, runs offline on a fresh machine, and is
   explainable. Justify the choice against transformer and zero-shot-LLM
   alternatives."

4. "Implement: conservative tweet text cleaning; a word + character TF-IDF
   logistic-regression model in a small, typed, documented module; a `train.py`
   that saves the model artifact plus honest hold-out metrics (accuracy, F1,
   ROC-AUC, Brier, confusion matrix)."

5. "Add per-prediction explainability: show the top words pushing the decision
   toward/away from 'disaster', using the linear model's coefficients."

6. "Build a Streamlit UI: text input, label + calibrated confidence, the
   explanation panel, example buttons, graceful empty/whitespace handling, and a
   decision-support disclaimer. Add a deep-link query parameter so a prediction
   can render on load."

7. "Write `predict.py` as a CLI: `--input`/`--output`, read a CSV with a `text`
   column, write `text,label,score`, handle missing column / missing file / dirty
   rows gracefully."

8. "Pin dependencies and the Python version. Test the whole thing in a fresh
   virtual environment (clone → venv → install → run) and fix anything that
   breaks."

9. "Produce an anonymity-safe screenshot of the running app at localhost with the
   address bar and a sample prediction visible, captured from a throwaway browser
   profile so nothing identifying appears."

10. "Write the README, the one-page write-up, and this AI-disclosure annex.
    Ensure no real name, employer, project names, or other identifying details
    appear anywhere in the repository, commit history, or file metadata."

> Note: this is a readable summary of the directing prompts rather than a verbatim
> keystroke log; it accurately reflects the instructions that shaped every
> artifact in this repository.
