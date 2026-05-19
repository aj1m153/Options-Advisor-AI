"""
utils/finbert.py
────────────────
FinBERT-based sentiment analysis for financial headlines.
Model : ProsusAI/finbert  (~440 MB, downloaded once and cached by HuggingFace)
Output: per-headline {positive, negative, neutral} probabilities + aggregate score

If transformers/torch are not installed the module degrades gracefully to a
keyword-heuristic fallback so the rest of the app keeps working.
"""

from __future__ import annotations
from typing import List, Dict

# ── Optional heavy imports ────────────────────────────────────────────────────
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline as hf_pipeline,
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


# ── Keyword fallback (same logic as original app) ────────────────────────────
_POS = {"surge","beat","record","growth","profit","rise","gain","strong",
        "bullish","upgrade","buy","soar","exceed","rally","boom","outperform"}
_NEG = {"drop","fall","miss","loss","weak","cut","bearish","downgrade","sell",
        "crash","decline","plunge","warn","risk","slump","underperform","layoff"}


def _keyword_sentiment(headline: str) -> float:
    words = set(headline.lower().split())
    pos   = len(words & _POS)
    neg   = len(words & _NEG)
    denom = max(pos + neg, 1)
    return (pos - neg) / denom


# ── FinBERT loader (Streamlit-cache-aware but importable standalone) ──────────
_PIPE_CACHE: dict = {}   # module-level cache so callers can use this outside st


def load_finbert_pipeline():
    """
    Load ProsusAI/finbert once and cache at module level.
    Returns the HuggingFace pipeline or None if unavailable.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None

    key = "finbert"
    if key not in _PIPE_CACHE:
        try:
            tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            device = 0 if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else -1
            _PIPE_CACHE[key] = hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tok,
                device=device,
                top_k=None,        # return all three labels
            )
        except Exception:
            _PIPE_CACHE[key] = None

    return _PIPE_CACHE[key]


# ── Main analysis function ────────────────────────────────────────────────────
def analyze_sentiment(
    headlines: List[str],
    pipe=None,
    max_headlines: int = 15,
) -> Dict:
    """
    Analyse headlines and return:
        score   : float  −1.0 … +1.0
        label   : "positive" | "neutral" | "negative"
        method  : "finbert" | "keyword"
        details : list of per-headline dicts
    """
    headlines = [h for h in headlines if h.strip()][:max_headlines]

    if not headlines:
        return {"score": 0.0, "label": "neutral", "method": "none", "details": []}

    # ── FinBERT path ──────────────────────────────────────────────────────────
    if pipe is not None:
        details: List[Dict] = []
        for headline in headlines:
            try:
                raw = pipe(headline[:512])          # returns [[{label,score},...]]
                scores_map = {r["label"].lower(): r["score"] for r in raw[0]}
                pos  = scores_map.get("positive", 0.0)
                neg  = scores_map.get("negative", 0.0)
                neu  = scores_map.get("neutral",  0.0)
                sent = round(pos - neg, 4)
                details.append({
                    "headline":  headline,
                    "label":     max(scores_map, key=scores_map.get),
                    "positive":  round(pos,  4),
                    "negative":  round(neg,  4),
                    "neutral":   round(neu,  4),
                    "sentiment": sent,
                })
            except Exception:
                # individual headline failure → fall back for this headline
                details.append({
                    "headline":  headline,
                    "label":     "neutral",
                    "positive":  0.0,
                    "negative":  0.0,
                    "neutral":   1.0,
                    "sentiment": _keyword_sentiment(headline),
                })

        avg   = sum(d["sentiment"] for d in details) / len(details)
        label = "positive" if avg > 0.12 else ("negative" if avg < -0.12 else "neutral")
        return {"score": round(avg, 4), "label": label, "method": "finbert", "details": details}

    # ── Keyword fallback ─────────────────────────────────────────────────────
    details = []
    for headline in headlines:
        sv = _keyword_sentiment(headline)
        details.append({
            "headline":  headline,
            "label":     "positive" if sv > 0 else ("negative" if sv < 0 else "neutral"),
            "positive":  max(sv, 0),
            "negative":  max(-sv, 0),
            "neutral":   1.0 - abs(sv),
            "sentiment": sv,
        })

    avg   = sum(d["sentiment"] for d in details) / len(details)
    label = "positive" if avg > 0.12 else ("negative" if avg < -0.12 else "neutral")
    return {"score": round(avg, 4), "label": label, "method": "keyword", "details": details}
