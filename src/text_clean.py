"""Lightweight, dependency-free text normalisation for short social-media text.

The cleaning is deliberately conservative: tweets carry signal in hashtags,
ALL-CAPS, and the presence of URLs, so we normalise rather than strip those.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_AMP_RE = re.compile(r"&amp;")


def clean_text(text: str) -> str:
    """Normalise a single tweet for vectorisation.

    URLs and @mentions are replaced with stable placeholder tokens (their
    *presence* is predictive, their exact value is noise); HTML entities are
    decoded; whitespace is collapsed. Casing and hashtags are kept because the
    downstream vectoriser lowercases and splits them itself.
    """
    if not isinstance(text, str):
        return ""
    text = _AMP_RE.sub(" and ", text)
    text = _URL_RE.sub(" httpurl ", text)
    text = _MENTION_RE.sub(" usermention ", text)
    text = text.replace("#", " hashtag_")
    text = _WS_RE.sub(" ", text)
    return text.strip()
