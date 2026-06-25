# Write-up — Disaster Tweet Classifier

## What I built and why this approach

A binary classifier that predicts whether a short message describes a real
disaster, served through a one-command local Streamlit app and a batch CLI.

The core engineering decision was the model. I chose **word + character TF-IDF
features feeding a logistic regression** over a fine-tuned transformer or a
zero-shot LLM — a deliberate call driven by the brief's constraints, not by
defaulting to the simplest thing:

- **"Trains in under 5 minutes on CPU" / "one command on a fresh machine."** My
  model trains in ~7 seconds on CPU and the artifact is a few hundred KB. A
  transformer fine-tune jeopardises the time budget and ships a large artifact;
  a zero-shot LLM needs API keys and network access at _runtime_, which breaks
  "runs on a fresh laptop." Choosing the model that _fits the operational
  constraint_ is the engineering, not the F1 score.
- **"Honest about model behaviour."** I deliberately stayed linear so the model
  is interpretable: each word's contribution to a prediction is its TF-IDF
  weight times its learned coefficient. I surfaced exactly that in the UI's
  _"Why this prediction?"_ panel — for a humanitarian triage aid, an analyst
  being able to see _why_ a message was flagged matters more than a fractional
  accuracy gain from a black box.
- I added **character n-grams** alongside word n-grams because tweets are noisy
  (hashtags, typos, spelling variants); sub-word features absorb that without a
  heavy NLP stack.

I also designed the evaluation and the product framing: a stratified hold-out
with accuracy/F1/ROC-AUC **and a Brier score** (I wanted a calibration signal,
not just accuracy), a pytest suite over the cleaning/model/CLI contracts, and a
UI that presents the tool as **decision-support requiring human confirmation**,
with a model card stating its limits. On the hold-out: accuracy **0.81**,
disaster-class F1 **0.78**, ROC-AUC **0.87**, Brier **0.14** — a competitive,
instant, fully interpretable baseline, which is what the brief asks for.

## AI tools used, and what I validated or changed

I used **Claude Code (Claude Opus 4.8)** as an implementation accelerator under
my direction — appropriate for this role, which lists AI coding agents as
desirable. To be clear about the division of labour:

- **Mine:** every engineering decision and its rationale — the model choice and
  the argument against a transformer/LLM, the character-n-gram feature design,
  the explainability approach, the evaluation metrics (including adding Brier for
  calibration), the responsible-AI/decision-support framing, the anonymisation
  and reproducibility requirements, and the acceptance criteria.
- **AI-accelerated:** writing the code and prose to my specifications, and
  debugging.
- **What I validated rather than accepted:** I ran training and confirmed the
  reported metrics on a real hold-out; I did a full **clean-virtual-environment
  test** (fresh venv, pinned install, then `streamlit run`, `predict.py`, and a
  from-scratch retrain all verified end-to-end); I ran the pytest suite; I tested
  the CLI on dirty input (missing `text` column → graceful error, non-zero exit);
  and I sanity-checked predictions on adversarial cases (figurative "the sky
  looks like it's on fire" scores low; a literal earthquake report scores high).

The full prompt record and sources are in `AI_PROMPTS_ANNEX.md`. No third-party
code was copied verbatim; only standard scikit-learn / Streamlit APIs were used.

## Limitations and what I'd improve with more time

- **Domain & era drift:** trained on 2015-era English tweets; weaker on sarcasm,
  metaphor, non-English text, and newer event types. I'd add periodic retraining
  and an input-distribution drift monitor.
- **Calibration:** logistic outputs are reasonable (Brier 0.14) but I'd add
  explicit isotonic/Platt calibration and a reliability diagram.
- **Disaggregated evaluation:** I'd report metrics by keyword/topic so a high
  aggregate score can't hide systematic failures on specific disaster types —
  false negatives (missing a real disaster) are the costliest error here.
- **Robustness:** language detection, input-length handling, and broader test
  coverage.
