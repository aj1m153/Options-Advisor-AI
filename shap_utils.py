"""
utils/shap_utils.py
───────────────────
SHAP-based explainability for the Random Forest strategy recommender.

Produces a Plotly waterfall chart showing which features pushed the model
toward (or away from) a specific strategy recommendation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass


# ── Human-readable feature labels ────────────────────────────────────────────
FEAT_LABELS: Dict[str, str] = {
    "iv_rank":            "IV Rank (0–100)",
    "hv_30":              "Historical Vol 30-Day",
    "hv_10":              "Historical Vol 10-Day",
    "hv_iv_ratio":        "HV / IV Ratio",
    "trend_score":        "Trend Score (−1 … +1)",
    "rsi":                "RSI (14)",
    "momentum_20d":       "20-Day Price Momentum",
    "momentum_5d":        "5-Day Price Momentum",
    "earnings_proximity": "Earnings Proximity",
    "sentiment_score":    "News Sentiment",
}

DARK = dict(template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117", font=dict(color="#e0e0e0"))


# ── Core SHAP computation ────────────────────────────────────────────────────
def compute_shap(
    model,
    features_dict: Dict[str, float],
    class_idx: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Returns (shap_values_array, base_value) for a single sample and class.
    Both are None when SHAP is unavailable.
    """
    if not SHAP_AVAILABLE:
        return None, None

    try:
        X          = pd.DataFrame([features_dict])
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)      # list[n_classes] of ndarray (1, n_feat)
        base_vals  = explainer.expected_value       # list[n_classes] or float

        sv = np.array(shap_vals[class_idx]).flatten()  # → (n_features,)
        bv = float(base_vals[class_idx]) if hasattr(base_vals, "__len__") else float(base_vals)
        return sv, bv

    except Exception:
        return None, None


# ── Plotly waterfall chart ────────────────────────────────────────────────────
def plot_shap_waterfall(
    shap_values:   np.ndarray,
    features_dict: Dict[str, float],
    base_value:    float,
    strategy_name: str,
    strategy_color: str = "#00b894",
) -> go.Figure:
    """
    Horizontal waterfall bar chart showing each feature's SHAP contribution.
    Green bars = push toward the strategy; red bars = push away.
    """
    keys    = list(features_dict.keys())
    vals    = list(features_dict.values())
    labels  = [FEAT_LABELS.get(k, k) for k in keys]

    # Sort ascending by |SHAP| so the most important feature is at the top
    order   = np.argsort(np.abs(shap_values))
    sv      = shap_values[order]
    fl      = [f"{labels[i]}\n= {vals[i]:.3g}" for i in order]

    colors  = ["#00b894" if v >= 0 else "#d63031" for v in sv]
    text    = [f"{v:+.4f}" for v in sv]

    fig = go.Figure(go.Bar(
        y=fl,
        x=sv,
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))

    # Base-value reference line
    fig.add_vline(
        x=0,
        line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash"),
    )

    fig.update_layout(
        **DARK,
        title=(
            f"🔬 Why <b>{strategy_name}</b>? — SHAP Feature Contributions<br>"
            f"<sub>Base probability: {base_value:.3f} | "
            f"Green = supports this strategy | Red = works against it</sub>"
        ),
        xaxis_title="SHAP value (impact on class probability)",
        height=460,
        margin=dict(l=10, r=90, t=80, b=40),
        xaxis=dict(zeroline=False),
    )
    return fig


# ── Summary bar chart (global feature importance) ────────────────────────────
def plot_shap_summary(model, feature_names: list) -> go.Figure:
    """Mean |SHAP| across all classes — a global feature importance view."""
    if not SHAP_AVAILABLE:
        return go.Figure()

    import pandas as _pd

    fi     = np.array(model.feature_importances_)
    labels = [FEAT_LABELS.get(k, k) for k in feature_names]
    df     = _pd.DataFrame({"Feature": labels, "Importance": fi}).sort_values("Importance")

    fig = go.Figure(go.Bar(
        y=df["Feature"], x=df["Importance"],
        orientation="h",
        marker_color="#a29bfe",
        text=[f"{v:.4f}" for v in df["Importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        **DARK,
        title="Global Feature Importance (Random Forest — mean decrease in impurity)",
        height=360,
        margin=dict(l=10, r=70, t=50, b=30),
    )
    return fig
