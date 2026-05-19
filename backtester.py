"""
utils/backtester.py
────────────────────
Historical options strategy backtester.

Pricing   : Black-Scholes (scipy.stats.norm)
IV proxy  : 20-day historical volatility (from yfinance price data)
Supported : Long Call, Long Put, Covered Call, Straddle, Iron Condor,
            Bull Call Spread, Bear Put Spread, Cash-Secured Put, Strangle
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    from scipy.stats import norm as _norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DARK = dict(template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117", font=dict(color="#e0e0e0"))

SUPPORTED_STRATEGIES = [
    "Long Call", "Long Put", "Covered Call", "Straddle",
    "Strangle", "Iron Condor", "Bull Call Spread",
    "Bear Put Spread", "Cash-Secured Put",
]


# ── Black-Scholes pricer ──────────────────────────────────────────────────────
def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """European option price via Black-Scholes."""
    if T <= 1e-6:
        return max(0.0, (S - K) if opt == "call" else (K - S))
    if sigma <= 1e-6 or not SCIPY_AVAILABLE:
        return max(0.0, (S - K) if opt == "call" else (K - S))

    d1 = (np.log(S / max(K, 1e-8)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return float(max(0.0, S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)))
    return float(max(0.0, K * np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)))


# ── Historical volatility ─────────────────────────────────────────────────────
def _hv(prices: pd.Series, period: int = 20) -> float:
    lr = np.log(prices / prices.shift(1)).dropna()
    if len(lr) < period:
        return 0.25
    return float(lr.tail(period).std() * np.sqrt(252))


# ── Trade record ──────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_date:  str
    exit_date:   str
    entry_spot:  float
    exit_spot:   float
    sigma:       float          # IV at entry
    pnl_dollar:  float          # per-contract P&L
    pnl_pct:     float          # as fraction of cost/collateral
    won:         bool
    cost:        float          # debit paid (or collateral for credit strats)
    exit_reason: str            # "expiry" | "stop_loss" | "take_profit"
    details:     dict = field(default_factory=dict)


# ── Strategy payoff functions ─────────────────────────────────────────────────
def _trade_pnl(
    strategy: str,
    S0: float, S_end: float,
    sigma: float, T: float, r: float,
    otm_pct: float,              # % OTM offset for spread strategies
    stop_pct: float,
    tp_pct: float,
) -> Dict:
    """
    Returns dict with keys: pnl (per-share), cost, exit_reason, details.
    All prices are per-share; multiply by 100 for one contract.
    """
    K_atm  = S0
    K_otm_c = S0 * (1 + otm_pct)   # OTM call strike
    K_otm_p = S0 * (1 - otm_pct)   # OTM put strike
    wing    = S0 * otm_pct          # spread width

    pnl     = 0.0
    cost    = 0.01
    details = {}

    if strategy == "Long Call":
        prem  = bs_price(S0, K_atm, T, r, sigma, "call")
        payoff = max(0, S_end - K_atm)
        pnl    = payoff - prem
        cost   = prem
        details = {"strike": K_atm, "premium": round(prem, 4)}

    elif strategy == "Long Put":
        prem  = bs_price(S0, K_atm, T, r, sigma, "put")
        payoff = max(0, K_atm - S_end)
        pnl    = payoff - prem
        cost   = prem
        details = {"strike": K_atm, "premium": round(prem, 4)}

    elif strategy == "Covered Call":
        K_cc   = S0 * 1.05          # sell 5% OTM call
        prem   = bs_price(S0, K_cc, T, r, sigma, "call")
        stock_pnl = S_end - S0
        call_pnl  = prem - max(0, S_end - K_cc)
        pnl       = stock_pnl + call_pnl
        cost      = S0              # effective cost is stock price
        details   = {"cc_strike": K_cc, "call_prem": round(prem, 4)}

    elif strategy == "Cash-Secured Put":
        K_csp = S0 * 0.97           # sell 3% OTM put
        prem  = bs_price(S0, K_csp, T, r, sigma, "put")
        payoff = prem - max(0, K_csp - S_end)
        pnl    = payoff
        cost   = K_csp              # cash required
        details = {"put_strike": K_csp, "put_prem": round(prem, 4)}

    elif strategy == "Straddle":
        cp = bs_price(S0, K_atm, T, r, sigma, "call")
        pp = bs_price(S0, K_atm, T, r, sigma, "put")
        total_prem = cp + pp
        payoff     = abs(S_end - K_atm)
        pnl        = payoff - total_prem
        cost       = total_prem
        details    = {"call_prem": round(cp, 4), "put_prem": round(pp, 4)}

    elif strategy == "Strangle":
        cp = bs_price(S0, K_otm_c, T, r, sigma, "call")
        pp = bs_price(S0, K_otm_p, T, r, sigma, "put")
        total_prem = cp + pp
        payoff     = max(0, S_end - K_otm_c) + max(0, K_otm_p - S_end)
        pnl        = payoff - total_prem
        cost       = total_prem
        details    = {"call_strike": K_otm_c, "put_strike": K_otm_p}

    elif strategy == "Iron Condor":
        # Sell OTM call spread + sell OTM put spread
        K_cs = S0 + wing;   K_cl = S0 + wing * 2   # short/long call
        K_ps = S0 - wing;   K_pl = S0 - wing * 2   # short/long put
        credit = (
            bs_price(S0, K_cs, T, r, sigma, "call") -
            bs_price(S0, K_cl, T, r, sigma, "call") +
            bs_price(S0, K_ps, T, r, sigma, "put")  -
            bs_price(S0, K_pl, T, r, sigma, "put")
        )
        call_loss = max(0, S_end - K_cs) - max(0, S_end - K_cl)
        put_loss  = max(0, K_ps - S_end) - max(0, K_pl - S_end)
        pnl  = credit - call_loss - put_loss
        cost = max(wing * 0.5, 0.01)   # margin proxy
        details = {"credit": round(credit, 4), "wing_width": round(wing, 2)}

    elif strategy == "Bull Call Spread":
        K_lo = K_atm;   K_hi = K_otm_c
        debit  = (bs_price(S0, K_lo, T, r, sigma, "call") -
                  bs_price(S0, K_hi, T, r, sigma, "call"))
        payoff = min(max(0, S_end - K_lo), K_hi - K_lo)
        pnl    = payoff - debit
        cost   = debit
        details = {"buy_strike": K_lo, "sell_strike": K_hi, "debit": round(debit, 4)}

    elif strategy == "Bear Put Spread":
        K_hi = K_atm;   K_lo = K_otm_p
        debit  = (bs_price(S0, K_hi, T, r, sigma, "put") -
                  bs_price(S0, K_lo, T, r, sigma, "put"))
        payoff = min(max(0, K_hi - S_end), K_hi - K_lo)
        pnl    = payoff - debit
        cost   = debit
        details = {"buy_strike": K_hi, "sell_strike": K_lo, "debit": round(debit, 4)}

    # Apply stop-loss / take-profit
    exit_reason = "expiry"
    if cost > 1e-6:
        if pnl < -stop_pct * cost:
            pnl = -stop_pct * cost
            exit_reason = "stop_loss"
        elif pnl > tp_pct * cost:
            pnl = tp_pct * cost
            exit_reason = "take_profit"

    return {"pnl": pnl, "cost": cost, "exit_reason": exit_reason, "details": details}


# ── Main backtest function ────────────────────────────────────────────────────
def backtest(
    ticker:        str,
    strategy_name: str,
    start_date:    str,
    end_date:      str,
    holding_days:  int   = 30,
    risk_free:     float = 0.045,
    stop_pct:      float = 0.50,
    take_profit_pct: float = 1.00,
    initial_capital: float = 50_000,
    position_size_pct: float = 0.05,  # 5% of capital per trade
    otm_pct:       float = 0.05,
) -> Dict:
    """
    Run a historical backtest.  Returns a dict with:
        trades         list[dict]
        equity_curve   list[float]
        bh_curve       list[float]    buy-and-hold comparison
        metrics        dict
        error          str (only on failure)
    """
    try:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
    except Exception as e:
        return {"error": f"Data fetch failed: {e}"}

    if hist is None or len(hist) < holding_days + 35:
        return {"error": "Not enough historical data. Use a longer date range or a different ticker."}

    close     = hist["Close"].squeeze()
    idx       = close.index
    T         = holding_days / 252

    trades:    List[Trade] = []
    equity     = [float(initial_capital)]
    bh_base    = float(close.iloc[30])

    i = 30   # warmup
    while i + holding_days < len(idx):
        S0    = float(close.iloc[i])
        S_end = float(close.iloc[i + holding_days])
        sigma = _hv(close.iloc[:i + 1], 20)

        result = _trade_pnl(
            strategy_name, S0, S_end, sigma, T,
            risk_free, otm_pct, stop_pct, take_profit_pct,
        )

        cost      = result["cost"]
        pnl_share = result["pnl"]
        pnl_pct   = pnl_share / cost if cost > 1e-6 else 0.0

        # Position sizing: allocate position_size_pct of current equity, 1 contract = 100 shares
        alloc     = equity[-1] * position_size_pct
        contracts = max(1, int(alloc / max(cost * 100, 1)))
        pnl_trade = pnl_share * 100 * contracts

        trades.append(Trade(
            entry_date  = str(idx[i].date()),
            exit_date   = str(idx[i + holding_days].date()),
            entry_spot  = round(S0,    2),
            exit_spot   = round(S_end, 2),
            sigma       = round(sigma * 100, 1),
            pnl_dollar  = round(pnl_trade, 2),
            pnl_pct     = round(pnl_pct * 100, 2),
            won         = pnl_share > 0,
            cost        = round(cost, 4),
            exit_reason = result["exit_reason"],
            details     = {**result["details"], "contracts": contracts},
        ))

        equity.append(max(0.0, equity[-1] + pnl_trade))
        i += holding_days

    if not trades:
        return {"error": "No trades generated. Try widening the date range."}

    # ── Buy-and-hold benchmark ──────────────────────────────────────────────
    bh_curve = []
    entry_idxs = [30] + [30 + j * holding_days for j in range(1, len(trades))]
    for j, t in enumerate(trades):
        bh_price = float(close.iloc[min(entry_idxs[j] + holding_days,
                                        len(close) - 1)])
        bh_ret   = bh_price / bh_base
        bh_curve.append(initial_capital * bh_ret)

    # ── Metrics ─────────────────────────────────────────────────────────────
    pnls      = [t.pnl_dollar for t in trades]
    wins      = [t for t in trades if t.won]
    losses    = [t for t in trades if not t.won]

    total_ret  = (equity[-1] - equity[0]) / equity[0] * 100
    win_rate   = len(wins) / len(trades) * 100
    avg_win    = np.mean([t.pnl_dollar for t in wins])   if wins   else 0
    avg_loss   = np.mean([t.pnl_dollar for t in losses]) if losses else 0
    pf_denom   = abs(sum(t.pnl_dollar for t in losses)) or 1e-8
    pf         = sum(t.pnl_dollar for t in wins) / pf_denom

    eq = np.array(equity)
    peak   = np.maximum.accumulate(eq)
    dd_arr = (eq - peak) / np.where(peak > 0, peak, 1) * 100
    max_dd = float(np.min(dd_arr))

    pct_rets   = np.array([t.pnl_pct / 100 for t in trades])
    ppa        = 252 / holding_days
    sharpe     = (pct_rets.mean() * ppa) / (pct_rets.std() * np.sqrt(ppa) + 1e-8)

    bh_total_ret = (bh_curve[-1] - initial_capital) / initial_capital * 100

    return {
        "trades":      [t.__dict__ for t in trades],
        "equity_curve": equity,
        "bh_curve":    bh_curve,
        "metrics": {
            "total_return_pct":  round(total_ret,  2),
            "bh_return_pct":     round(bh_total_ret, 2),
            "win_rate_pct":      round(win_rate,   2),
            "total_trades":      len(trades),
            "avg_win_usd":       round(avg_win,    2),
            "avg_loss_usd":      round(avg_loss,   2),
            "profit_factor":     round(pf,          2),
            "max_drawdown_pct":  round(max_dd,      2),
            "sharpe_ratio":      round(sharpe,      2),
            "final_capital":     round(equity[-1],  2),
        },
    }


# ── Visualisation helpers ─────────────────────────────────────────────────────
def plot_equity_curve(equity: list, bh_curve: list, strategy: str, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=equity, mode="lines",
        name=f"{strategy}",
        line=dict(color="#00b894", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,184,148,0.07)",
    ))
    if bh_curve:
        fig.add_trace(go.Scatter(
            y=bh_curve, mode="lines",
            name=f"Buy & Hold {ticker}",
            line=dict(color="#fdcb6e", width=1.8, dash="dot"),
        ))

    fig.update_layout(
        **DARK,
        title=f"Portfolio Equity Curve — {strategy} on {ticker}",
        xaxis_title="Trade #",
        yaxis_title="Portfolio Value ($)",
        height=360,
        margin=dict(l=0, r=0, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_trade_pnl_bar(trades: list) -> go.Figure:
    dates  = [t["entry_date"] for t in trades]
    pnls   = [t["pnl_dollar"] for t in trades]
    colors = ["#00b894" if p >= 0 else "#d63031" for p in pnls]

    fig = go.Figure(go.Bar(
        x=dates, y=pnls, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(
        **DARK,
        title="Per-Trade P&L ($)",
        height=280,
        margin=dict(l=0, r=0, t=40, b=40),
    )
    return fig


def plot_drawdown(equity: list) -> go.Figure:
    eq   = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak > 0, peak, 1) * 100

    fig = go.Figure(go.Scatter(
        y=dd, mode="lines", fill="tozeroy",
        fillcolor="rgba(214,48,49,0.15)",
        line=dict(color="#d63031", width=1.5),
        name="Drawdown %",
    ))
    fig.update_layout(
        **DARK,
        title="Portfolio Drawdown (%)",
        yaxis_title="%",
        height=240,
        margin=dict(l=0, r=0, t=40, b=30),
    )
    return fig
