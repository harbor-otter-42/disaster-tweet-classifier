# Write-up — Disaster Tweet Classifier

## What I built and why this approach

A binary classifier that predicts whether a short message describes a real
disaster, served through a one-command local Streamlit app and a batch CLI.

The model is **word + character TF-IDF features feeding a logistic regression**.
I chose this over a fine-tuned transformer or a zero-shot LLM deliberately,
because the brief's constraints point straight at it:

- **"Trains in under 5 minutes on CPU" / "one command on a fresh machine."** This
  model trains in ~7 seconds on CPU and the artifact is a few hundred KB. A
  transformer fine-tune risks the time budget and a much larger artifact; a
  zero-shot LLM would need API keys and network access at runtime, breaking the
  "runs on a fresh laptop" requirement.
- **"Honest about model behaviour."** A linear model is transparent: each word's
  contribution to a prediction is simply its TF-IDF weight times its learned
  coefficient. The UI's _"Why this prediction?"_ panel surfaces exactly that, so
  a non-technical user can see _why_ a message was flagged — which matters more
  for a triage aid than a fractional F1 gain from a black box.
- **Character n-grams** were added alongside word n-grams because tweets are full
  of hashtags, typos, and spelling variants; sub-word features make the model
  robust to those without any heavy NLP stack.

On a 20% stratified hold-out: accuracy **0.81**, disaster-class F1 **0.78**,
ROC-AUC **0.87**, Brier score **0.14**. Per the brief, the aim is working code
and sensible choices rather than state-of-the-art F1; this is a competitive,
instant, fully interpretable baseline.

The UI handles empty/whitespace input gracefully, shows label + calibrated
confidence + the probability, and frames itself as **decision-support requiring
human confirmation** — appropriate for anything touching real disaster response.

## AI tools used, and what I validated or changed

- **Tool / model:** Claude Code (CLI) with Claude Opus 4.8. Used for code
  generation, documentation drafting, debugging, and the screenshot tooling.
- **What I directed:** the architecture decision (linear model over transformer,
  and _why_), the explainability feature, the responsible-use framing, and the
  packaging/anonymity requirements.
- **What I validated rather than trusted blindly:**
  - Ran `train.py` and confirmed the reported metrics on a real hold-out split.
  - Did a full **clean-virtual-environment test**: created a fresh venv, installed
    the pinned `requirements.txt`, and verified `streamlit run app.py`,
    `predict.py`, and a from-scratch retrain all work end-to-end.
  - Tested `predict.py` on dirty/edge input (missing `text` column → graceful
    error with a non-zero exit code; empty rows handled).
  - Confirmed sensible predictions on adversarial examples (figurative "the sky
    looks like it's on fire" scores low; a literal earthquake report scores high).
- A full record of the prompts is in `AI_PROMPTS_ANNEX.md`. No third-party code
  was copied verbatim; only standard scikit-learn / Streamlit APIs were used.

## Limitations and what I'd improve with more time

- **Domain & era drift:** trained on 2015-era English tweets; weaker on sarcasm,
  metaphor, non-English text, and newer event types. I'd add periodic retraining
  and a drift monitor on the input distribution.
- **Calibration:** logistic outputs are reasonably calibrated (Brier 0.14) but I'd
  add explicit Platt/isotonic calibration and a reliability diagram.
- **Evaluation depth:** I'd add disaggregated metrics (e.g. by keyword/topic) so a
  high aggregate score can't hide systematic failures on specific disaster types —
  false negatives are the costliest error in this domain.
- **Robustness:** add input-length and language detection, and a small test suite
  (pytest) covering the cleaning, model, and CLI contracts.
