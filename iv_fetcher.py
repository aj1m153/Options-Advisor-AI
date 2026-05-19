"""
utils/iv_fetcher.py
───────────────────
Fetches real implied-volatility data from Polygon.io.

Free-tier key  → 15-min delayed data, 5 calls/min
Starter ($29)  → real-time data, unlimited calls

Sign up at: https://polygon.io  (free key works for this module)

Returns:
    atm_iv      float   annualised IV (0–1) of nearest ATM option
    atm_call_iv float   call-side ATM IV
    atm_put_iv  float   put-side ATM IV
    skew        float   put_iv − call_iv  (positive = put premium / fear)
    iv_rank     float   0–100 (where today's IV sits vs 52-week range)
    source      str     "polygon" | "fallback"
"""

from __future__ import annotations
import numpy as np
import requests
from typing import Optional, Dict


POLYGON_BASE = "https://api.polygon.io"
_TIMEOUT     = 10


# ── Internal helpers ──────────────────────────────────────────────────────────
def _get(endpoint: str, api_key: str, params: dict | None = None) -> Optional[dict]:
    p = {"apiKey": api_key, **(params or {})}
    try:
        r = requests.get(f"{POLYGON_BASE}{endpoint}", params=p, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _spot_price(ticker: str, api_key: str) -> Optional[float]:
    """Get most recent closing price."""
    data = _get(f"/v2/aggs/ticker/{ticker}/prev", api_key, {"adjusted": "true"})
    if data:
        results = data.get("results", [])
        if results:
            return float(results[0]["c"])
    return None


def _one_year_closes(ticker: str, api_key: str) -> Optional[np.ndarray]:
    """Fetch up to 365 daily closing prices for IV-rank computation."""
    from datetime import date, timedelta
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    data  = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        api_key,
        {"adjusted": "true", "sort": "asc", "limit": 400},
    )
    if data:
        bars = data.get("results", [])
        if bars:
            return np.array([b["c"] for b in bars])
    return None


def _hv_iv_rank(current_iv: float, closes: np.ndarray, period: int = 30) -> float:
    """Compute IV rank using historical HV as IV proxy."""
    if len(closes) < period + 5:
        return 50.0
    log_ret = np.log(closes[1:] / closes[:-1])
    hvs = [
        log_ret[max(0, i - period):i].std() * np.sqrt(252)
        for i in range(period, len(log_ret))
    ]
    hv_min = min(hvs)
    hv_max = max(hvs)
    return float(np.clip((current_iv - hv_min) / max(hv_max - hv_min, 1e-6) * 100, 0, 100))


# ── Public API ────────────────────────────────────────────────────────────────
def fetch_polygon_iv(ticker: str, api_key: str) -> Optional[Dict]:
    """
    Main entry point.  Returns a dict or None on failure.
    """
    if not api_key:
        return None

    ticker = ticker.upper()

    # ── Step 1: nearest ATM options snapshot ─────────────────────────────────
    snap = _get(
        f"/v3/snapshot/options/{ticker}",
        api_key,
        {"limit": 100, "order": "asc"},
    )
    if not snap:
        return None

    contracts = snap.get("results", [])
    if not contracts:
        return None

    # ── Step 2: spot price ────────────────────────────────────────────────────
    spot = _spot_price(ticker, api_key)
    if spot is None:
        return None

    # ── Step 3: collect ATM IVs ───────────────────────────────────────────────
    call_ivs, put_ivs = [], []
    for c in contracts:
        details = c.get("details", {})
        strike  = details.get("strike_price") or 0
        ctype   = (details.get("contract_type") or "").lower()
        iv      = c.get("implied_volatility") or 0

        if iv <= 0 or strike <= 0:
            continue
        if abs(strike - spot) / spot > 0.03:   # only within ±3% of spot
            continue

        if ctype == "call":
            call_ivs.append(float(iv))
        elif ctype == "put":
            put_ivs.append(float(iv))

    if not call_ivs and not put_ivs:
        return None

    atm_call = float(np.mean(call_ivs)) if call_ivs else None
    atm_put  = float(np.mean(put_ivs))  if put_ivs  else None
    atm_iv   = float(np.mean([v for v in [atm_call, atm_put] if v is not None]))
    skew     = round((atm_put - atm_call), 4) if (atm_put and atm_call) else 0.0

    # ── Step 4: IV rank ───────────────────────────────────────────────────────
    closes   = _one_year_closes(ticker, api_key)
    iv_rank  = _hv_iv_rank(atm_iv, closes) if closes is not None else 50.0

    return {
        "atm_iv":      round(atm_iv,   4),
        "atm_call_iv": round(atm_call, 4) if atm_call else None,
        "atm_put_iv":  round(atm_put,  4) if atm_put  else None,
        "iv_rank":     round(iv_rank,  1),
        "skew":        skew,
        "spot":        round(spot,     2),
        "source":      "polygon",
    }


def fetch_vix(api_key: str) -> Optional[float]:
    """Fetch the current CBOE VIX level from Polygon."""
    data = _get("/v2/aggs/ticker/I:VIX/prev", api_key, {"adjusted": "true"})
    if data:
        results = data.get("results", [])
        if results:
            return float(results[0]["c"])
    return None
