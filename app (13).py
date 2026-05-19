# ============================================================
#  📈 Options Advisor AI — Streamlit App
#  Analyzes US Stocks & ETFs and recommends the optimal
#  options strategy using ML trained on market features.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Advisor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .metric-box {
      background: #161b27;
      border: 1px solid #2d3346;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 10px;
  }
  .strat-card {
      background: #161b27;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 14px;
  }
  .badge {
      display: inline-block;
      padding: 3px 12px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
  }
  .headline-row {
      padding: 8px 14px;
      border-radius: 0 8px 8px 0;
      background: #161b27;
      margin: 5px 0;
  }
  div[data-testid="stExpander"] { border: 1px solid #2d3346; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  STRATEGY LIBRARY  (11 strategies)
# ─────────────────────────────────────────────────────────────
STRATEGIES = {
    "Long Call": {
        "emoji": "🚀",
        "description": "Buy a call option giving you the right to purchase shares at the strike price before expiry.",
        "beginner": "Think the stock will shoot up? Pay a small fee now for the right to buy at today's price later.",
        "market_view": "Strongly Bullish",
        "iv_pref": "Low IV — cheap premium to buy",
        "max_profit": "Unlimited",
        "max_loss": "Premium paid",
        "complexity": "Beginner",
        "color": "#00b894",
    },
    "Long Put": {
        "emoji": "📉",
        "description": "Buy a put option giving you the right to sell shares at the strike price before expiry.",
        "beginner": "Think the stock will crash? Pay a fee for the right to sell at today's price even if it falls.",
        "market_view": "Strongly Bearish",
        "iv_pref": "Low IV — cheap premium to buy",
        "max_profit": "Strike price (stock → $0)",
        "max_loss": "Premium paid",
        "complexity": "Beginner",
        "color": "#d63031",
    },
    "Covered Call": {
        "emoji": "💰",
        "description": "Own 100 shares and sell a call option against them to collect premium income.",
        "beginner": "Already own shares? Earn extra income by agreeing to sell at a higher price if the stock reaches it.",
        "market_view": "Neutral to Slightly Bullish",
        "iv_pref": "High IV — rich premium to collect",
        "max_profit": "Premium received + upside to strike",
        "max_loss": "Stock cost basis minus premium",
        "complexity": "Beginner",
        "color": "#0984e3",
    },
    "Cash-Secured Put": {
        "emoji": "🏦",
        "description": "Sell a put option while holding enough cash to purchase the shares if assigned.",
        "beginner": "Want to buy a stock at a lower price? Get paid upfront to agree to buy it cheaper.",
        "market_view": "Neutral to Slightly Bullish",
        "iv_pref": "High IV — rich premium to collect",
        "max_profit": "Premium received",
        "max_loss": "Strike price minus premium",
        "complexity": "Beginner",
        "color": "#6c5ce7",
    },
    "Bull Call Spread": {
        "emoji": "📈",
        "description": "Buy a lower-strike call and sell a higher-strike call to reduce cost and cap upside.",
        "beginner": "Moderately bullish? This is a cheaper way to bet on a stock going up with limited risk.",
        "market_view": "Moderately Bullish",
        "iv_pref": "Neutral — works in most IV environments",
        "max_profit": "Spread width minus net debit",
        "max_loss": "Net debit paid",
        "complexity": "Intermediate",
        "color": "#00cec9",
    },
    "Bear Put Spread": {
        "emoji": "🐻",
        "description": "Buy a higher-strike put and sell a lower-strike put to reduce cost on a bearish bet.",
        "beginner": "Moderately bearish? This is a cheaper way to bet on a stock going down with limited risk.",
        "market_view": "Moderately Bearish",
        "iv_pref": "Neutral — works in most IV environments",
        "max_profit": "Spread width minus net debit",
        "max_loss": "Net debit paid",
        "complexity": "Intermediate",
        "color": "#e17055",
    },
    "Straddle": {
        "emoji": "⚡",
        "description": "Buy both a call and a put at the same strike. Profits from a large move in either direction.",
        "beginner": "Big news coming (earnings, FDA, etc.)? Bet on a big move without picking a direction.",
        "market_view": "Neutral — expects large move either way",
        "iv_pref": "Low IV — buy before volatility spikes",
        "max_profit": "Unlimited",
        "max_loss": "Total premium paid (both legs)",
        "complexity": "Intermediate",
        "color": "#fdcb6e",
    },
    "Strangle": {
        "emoji": "🌪️",
        "description": "Buy an OTM call and an OTM put. Cheaper than a straddle but requires a bigger price move.",
        "beginner": "Like a Straddle but cheaper — buys options further from the current price, so needs a huge move.",
        "market_view": "Neutral — expects very large move",
        "iv_pref": "Very Low IV preferred",
        "max_profit": "Unlimited",
        "max_loss": "Total premium on both legs",
        "complexity": "Intermediate",
        "color": "#fd79a8",
    },
    "Iron Condor": {
        "emoji": "🦅",
        "description": "Sell OTM call spread + sell OTM put spread. Profit if the stock stays within a price range.",
        "beginner": "Think the stock will go nowhere? Collect income by betting it stays inside a price range.",
        "market_view": "Neutral — expects low volatility / sideways",
        "iv_pref": "High IV — sell rich premium on both sides",
        "max_profit": "Net credit received",
        "max_loss": "Spread width minus credit",
        "complexity": "Advanced",
        "color": "#a29bfe",
    },
    "Butterfly Spread": {
        "emoji": "🦋",
        "description": "Buy 1 ITM call, sell 2 ATM calls, buy 1 OTM call. Maximum profit if price lands exactly at center strike.",
        "beginner": "Think the stock will be at a very specific price at expiry? This cheap strategy pays max at that target.",
        "market_view": "Neutral — expects price to pin at strike",
        "iv_pref": "Low IV — cheap debit to pay",
        "max_profit": "Spread width minus debit",
        "max_loss": "Net debit paid",
        "complexity": "Advanced",
        "color": "#55efc4",
    },
    "Calendar Spread": {
        "emoji": "📅",
        "description": "Sell a near-term option and buy a same-strike longer-dated option. Profit from time decay differential.",
        "beginner": "Think the stock stays flat short-term but moves later? Profit from time working against the short option.",
        "market_view": "Neutral short-term / possible later move",
        "iv_pref": "Flat or rising long-term IV",
        "max_profit": "Extrinsic value differential",
        "max_loss": "Net debit paid",
        "complexity": "Advanced",
        "color": "#74b9ff",
    },
}

COMPLEXITY_COLORS = {"Beginner": "#00b894", "Intermediate": "#fdcb6e", "Advanced": "#d63031"}

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA",
    "META", "NFLX", "SPY",  "QQQ",  "IWM",  "GLD",
    "TLT",  "XLF",  "XLE",  "ARKK", "JPM",  "BAC",
    "AMD",  "INTC", "V",    "COIN", "PLTR", "UBER",
]


# ─────────────────────────────────────────────────────────────
#  TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_hv(prices: pd.Series, period: int = 30) -> pd.Series:
    """Annualised historical volatility."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252)


# ─────────────────────────────────────────────────────────────
#  DATA FETCHING  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str):
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1y")
        info = t.info
        return hist, info
    except Exception:
        return None, {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_data(ticker: str):
    try:
        t       = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return None, None, None
        chain = t.option_chain(expiries[0])
        return list(expiries), chain.calls, chain.puts
    except Exception:
        return None, None, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_sentiment(ticker: str):
    positive = {"surge","beat","record","growth","profit","rise","gain","strong",
                "bullish","upgrade","buy","soar","exceed","rally","boom"}
    negative = {"drop","fall","miss","loss","weak","cut","bearish","downgrade",
                "sell","crash","decline","plunge","warn","risk","slump"}
    try:
        news = yf.Ticker(ticker).news or []
        scores, headlines = [], []
        for item in news[:10]:
            title = item.get("title", "")
            headlines.append(title)
            words = set(title.lower().split())
            pos   = len(words & positive)
            neg   = len(words & negative)
            denom = max(pos + neg, 1)
            scores.append((pos - neg) / denom)
        avg = float(np.mean(scores)) if scores else 0.0
        return avg, headlines
    except Exception:
        return 0.0, []


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def compute_features(ticker: str):
    """Return (features_dict, context_dict) or (None, None) on failure."""
    hist, info = fetch_stock_data(ticker)
    if hist is None or len(hist) < 60:
        return None, None

    close = hist["Close"].squeeze()
    price = float(close.iloc[-1])

    # ── Moving averages & trend ──────────────────────────────
    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50

    trend = 0.0
    if price > ma20:  trend += 0.20
    if price > ma50:  trend += 0.30
    if price > ma200: trend += 0.30
    if ma20  > ma50:  trend += 0.10
    if ma50  > ma200: trend += 0.10
    trend_score = float(np.clip(trend * 2 - 1, -1, 1))   # −1 … +1

    # ── Momentum ─────────────────────────────────────────────
    mom20 = float((price / close.iloc[-21] - 1)) if len(close) >= 21 else 0.0
    mom5  = float((price / close.iloc[-6]  - 1)) if len(close) >= 6  else 0.0

    # ── Volatility / IV Rank ──────────────────────────────────
    hv30 = float(compute_hv(close, 30).iloc[-1])
    hv10 = float(compute_hv(close, 10).iloc[-1])

    hv_series = compute_hv(close, 30).dropna()
    if len(hv_series) >= 20:
        window = min(len(hv_series), 252)
        hv_min = float(hv_series.iloc[-window:].min())
        hv_max = float(hv_series.iloc[-window:].max())
        iv_rank = (hv30 - hv_min) / max(hv_max - hv_min, 1e-6) * 100
    else:
        iv_rank = 50.0

    # ── Options-derived ATM IV ────────────────────────────────
    atm_iv = None
    expiries, calls_df, puts_df = fetch_options_data(ticker)
    if calls_df is not None and len(calls_df) > 0:
        calls_df = calls_df.dropna(subset=["impliedVolatility", "strike"])
        if len(calls_df) > 0:
            closest = (calls_df["strike"] - price).abs().idxmin()
            atm_iv  = float(calls_df.loc[closest, "impliedVolatility"])

    hv_iv_ratio = hv30 / max(atm_iv if atm_iv else hv30, 1e-8)

    # ── RSI ───────────────────────────────────────────────────
    rsi = float(np.clip(compute_rsi(close).iloc[-1], 0, 100))

    # ── Sentiment ─────────────────────────────────────────────
    sentiment, headlines = fetch_news_sentiment(ticker)

    # ── Earnings proximity ────────────────────────────────────
    earnings_proximity = 0.0
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None:
            # calendar may be a dict or DataFrame depending on yf version
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = pd.to_datetime(cal["Earnings Date"][0])
            elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
                ed = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
            else:
                ed = None
            if ed is not None:
                if hasattr(ed, "tzinfo") and ed.tzinfo is not None:
                    ed = ed.tz_localize(None)
                days = (ed - datetime.now()).days
                if 0 < days < 60:
                    earnings_proximity = max(0.0, 1.0 - days / 60.0)
    except Exception:
        pass

    features = {
        "iv_rank":           float(np.clip(iv_rank,       0, 100)),
        "hv_30":             float(np.clip(hv30,           0,   2)),
        "hv_10":             float(np.clip(hv10,           0,   2)),
        "hv_iv_ratio":       float(np.clip(hv_iv_ratio,    0,   3)),
        "trend_score":       float(np.clip(trend_score,   -1,   1)),
        "rsi":               float(np.clip(rsi,            0, 100)),
        "momentum_20d":      float(np.clip(mom20,       -0.5, 0.5)),
        "momentum_5d":       float(np.clip(mom5,        -0.3, 0.3)),
        "earnings_proximity":float(np.clip(earnings_proximity, 0, 1)),
        "sentiment_score":   float(np.clip(sentiment,   -1,   1)),
    }

    context = {
        "price":    price,
        "ma20":     ma20,
        "ma50":     ma50,
        "ma200":    ma200,
        "atm_iv":   atm_iv,
        "hv_30":    hv30,
        "expiries": expiries,
        "headlines":headlines,
        "info":     info,
    }
    return features, context


# ─────────────────────────────────────────────────────────────
#  ML MODEL — synthetic-data expert-system approach
# ─────────────────────────────────────────────────────────────
def _expert_label(iv_rank, hv_iv_ratio, trend, rsi, earnings_prox, sentiment):
    """Rule-based expert system that generates training labels."""
    high_iv     = iv_rank > 55
    low_iv      = iv_rank < 30
    strong_bull = trend >  0.55 and rsi > 60
    bull        = trend >  0.25 or (rsi > 58 and sentiment > 0.15)
    strong_bear = trend < -0.55 and rsi < 40
    bear        = trend < -0.25 or (rsi < 42 and sentiment < -0.15)
    neutral     = not bull and not bear
    catalyst    = earnings_prox > 0.60
    hv_cheap    = hv_iv_ratio > 0.90   # historical vol > implied vol → IV cheap

    if catalyst and low_iv:
        if strong_bull: return "Long Call"
        if strong_bear: return "Long Put"
        return "Straddle"

    if catalyst and high_iv:
        if bull:  return "Bull Call Spread"
        if bear:  return "Bear Put Spread"
        return "Strangle"

    if strong_bull and low_iv:  return "Long Call"
    if strong_bear and low_iv:  return "Long Put"

    if bull  and high_iv: return "Covered Call"
    if bull  and not high_iv: return "Bull Call Spread"
    if bear  and high_iv: return "Bear Put Spread"
    if bear  and low_iv:  return "Long Put"

    if neutral and high_iv:  return "Iron Condor"
    if neutral and low_iv:
        return "Butterfly Spread" if iv_rank < 15 else "Calendar Spread"
    if neutral:
        return "Iron Condor"

    if high_iv and trend > 0: return "Cash-Secured Put"
    return "Iron Condor"


@st.cache_resource(show_spinner=False)
def train_model():
    """Generate 8 000 synthetic examples, train a Random Forest."""
    np.random.seed(42)
    n = 8_000

    X = pd.DataFrame({
        "iv_rank":           np.random.uniform(0,  100, n),
        "hv_30":             np.random.uniform(0.05, 1.2, n),
        "hv_10":             np.random.uniform(0.05, 1.5, n),
        "hv_iv_ratio":       np.random.uniform(0.2,  2.5, n),
        "trend_score":       np.random.uniform(-1,   1,   n),
        "rsi":               np.random.uniform(20,   80,  n),
        "momentum_20d":      np.random.uniform(-0.3, 0.3, n),
        "momentum_5d":       np.random.uniform(-0.15,0.15,n),
        "earnings_proximity":np.random.beta(1, 5,         n),
        "sentiment_score":   np.random.uniform(-1,   1,   n),
    })

    labels = [
        _expert_label(
            row["iv_rank"], row["hv_iv_ratio"], row["trend_score"],
            row["rsi"],     row["earnings_proximity"], row["sentiment_score"]
        )
        for _, row in X.iterrows()
    ]

    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    clf.fit(X, y_enc)
    return clf, le


def get_recommendations(features: dict, model, le):
    X     = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]
    top3  = np.argsort(proba)[-3:][::-1]
    return [
        {"strategy": le.classes_[i], "confidence": float(proba[i] * 100)}
        for i in top3
    ]


# ─────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────
DARK = dict(template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117", font=dict(color="#e0e0e0"))


def plot_price_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"{ticker} Price", "Volume", "RSI (14)"],
        row_heights=[0.60, 0.20, 0.20],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"],   close=hist["Close"],
        name="Price", showlegend=False,
    ), row=1, col=1)

    # MAs
    for period, color, name in [(20,"#00b894","MA20"),(50,"#fdcb6e","MA50"),(200,"#d63031","MA200")]:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"].rolling(period).mean(),
            name=name, line=dict(color=color, width=1.5),
        ), row=1, col=1)

    # Volume
    bar_colors = ["#00b894" if c >= o else "#d63031"
                  for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        marker_color=bar_colors, name="Volume", showlegend=False,
    ), row=2, col=1)

    # RSI
    rsi = compute_rsi(hist["Close"])
    fig.add_trace(go.Scatter(
        x=hist.index, y=rsi,
        line=dict(color="#a29bfe", width=1.5), name="RSI", showlegend=False,
    ), row=3, col=1)
    for level, color in [(70, "red"), (30, "green")]:
        fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1),
                      row=3, col=1)

    fig.update_layout(
        **DARK, height=660,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def plot_hv_chart(hist: pd.DataFrame) -> go.Figure:
    hv30 = compute_hv(hist["Close"], 30) * 100
    hv10 = compute_hv(hist["Close"], 10) * 100
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hv30, name="HV 30-Day",
                             line=dict(color="#00b894", width=2)))
    fig.add_trace(go.Scatter(x=hist.index, y=hv10, name="HV 10-Day",
                             line=dict(color="#fdcb6e", width=1.5, dash="dot")))
    fig.update_layout(**DARK, title="Historical Volatility (%)", height=290,
                      margin=dict(l=0, r=0, t=40, b=0), yaxis_tickformat=".1f")
    return fig


def plot_pnl(strategy_name: str, spot: float) -> go.Figure:
    K  = round(spot)        # ATM strike
    pr = spot * 0.03        # ~3% premium proxy
    sw = spot * 0.05        # 5% spread-width proxy

    prices = np.linspace(spot * 0.70, spot * 1.30, 400)

    pnl_funcs = {
        "Long Call":
            lambda S: np.maximum(-pr, S - K - pr),
        "Long Put":
            lambda S: np.maximum(-pr, K - S - pr),
        "Covered Call":
            lambda S: np.minimum(K - spot + pr, S - spot + pr),
        "Cash-Secured Put":
            lambda S: np.where(S >= K, pr, S - K + pr),
        "Bull Call Spread":
            lambda S: np.clip(S - K, -pr, sw - pr),
        "Bear Put Spread":
            lambda S: np.clip(K - S, -pr, sw - pr),
        "Straddle":
            lambda S: np.abs(S - K) - 2 * pr,
        "Strangle":
            lambda S: (np.maximum(S - (K + sw * 0.5), 0) +
                       np.maximum((K - sw * 0.5) - S, 0) - 1.5 * pr),
        "Iron Condor":
            lambda S: np.where(
                np.abs(S - K) < sw, pr,
                pr - np.maximum(np.abs(S - K) - sw, 0) * (pr / (sw * 0.5 + 1e-8))
            ),
        "Butterfly Spread":
            lambda S: np.maximum(0, sw - np.abs(S - K)) - pr * 0.5,
        "Calendar Spread":
            lambda S: pr * 0.8 - np.abs(S - K) * 0.015,
    }

    pnl   = pnl_funcs.get(strategy_name, lambda S: np.zeros_like(S))(prices)
    color = STRATEGIES.get(strategy_name, {}).get("color", "#00b894")

    pos_pnl = np.where(pnl >= 0, pnl, np.nan)
    neg_pnl = np.where(pnl <  0, pnl, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pos_pnl, fill="tozeroy",
                             fillcolor="rgba(0,184,148,0.18)",
                             line=dict(color="#00b894", width=2), name="Profit"))
    fig.add_trace(go.Scatter(x=prices, y=neg_pnl, fill="tozeroy",
                             fillcolor="rgba(214,48,49,0.18)",
                             line=dict(color="#d63031", width=2), name="Loss"))

    fig.add_hline(y=0,    line=dict(color="white", dash="dash", width=1))
    fig.add_vline(x=spot, line=dict(color="yellow", dash="dot", width=1.5),
                  annotation_text="Current Price", annotation_position="top right")

    fig.update_layout(
        **DARK, title=f"{strategy_name} — Illustrative P&L at Expiry",
        xaxis_title="Stock Price", yaxis_title="Profit / Loss ($)",
        height=320, showlegend=False,
        margin=dict(l=0, r=0, t=40, b=40),
    )
    return fig


def plot_feature_importance(model) -> go.Figure:
    feat_names = ["IV Rank","HV-30","HV-10","HV/IV Ratio",
                  "Trend","RSI","Mom-20D","Mom-5D",
                  "Earnings","Sentiment"]
    df = pd.DataFrame({"Feature": feat_names,
                       "Importance": model.feature_importances_}
                      ).sort_values("Importance")
    fig = go.Figure(go.Bar(x=df["Importance"], y=df["Feature"],
                           orientation="h", marker_color="#00b894"))
    fig.update_layout(**DARK, title="Random Forest Feature Importance",
                      height=340, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_oi_chart(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=calls["strike"], y=calls["openInterest"],
                         name="Calls OI", marker_color="#00b894", opacity=0.8))
    fig.add_trace(go.Bar(x=puts["strike"],  y=puts["openInterest"],
                         name="Puts OI",  marker_color="#d63031", opacity=0.8))
    fig.add_vline(x=spot, line=dict(color="yellow", dash="dash"),
                  annotation_text="Spot")
    fig.update_layout(**DARK, title="Open Interest by Strike",
                      barmode="overlay", height=320,
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_sentiment_gauge(score: float) -> go.Figure:
    val = score * 50 + 50  # map −1…+1 → 0…100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "", "valueformat": ".0f"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Sentiment Score (0–100)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#00b894" if score >= 0 else "#d63031"},
            "steps": [
                {"range": [0,  33], "color": "#2d1b1b"},
                {"range": [33, 66], "color": "#1e2020"},
                {"range": [66,100], "color": "#1b2d1b"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(**DARK, height=260,
                      margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Options Advisor AI")
    st.caption("ML-powered strategy recommender for US Stocks & ETFs")
    st.divider()

    user_level = st.radio(
        "👤 Your Experience Level",
        ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"],
        help="Adapts language, charts, and detail depth to your level.",
    )
    level = user_level.split()[1]   # "Beginner" | "Intermediate" | "Advanced"

    st.divider()
    ticker_input = st.text_input("🔍 Enter Ticker Symbol", value="AAPL",
                                 max_chars=10).upper().strip()

    st.markdown("**Quick Picks:**")
    cols = st.columns(3)
    quick = ["SPY","AAPL","NVDA","TSLA","QQQ","META","AMD","GLD","IWM"]
    for i, qt in enumerate(quick):
        if cols[i % 3].button(qt, key=f"qt_{qt}", use_container_width=True):
            ticker_input = qt

    st.divider()
    analyze_btn = st.button("🔬 Analyse Now", type="primary",
                            use_container_width=True)
    st.divider()
    st.caption("⚠️ For **educational purposes only**. Not financial advice. Options involve substantial risk of loss.")


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
if "ticker"   not in st.session_state: st.session_state["ticker"]   = "AAPL"
if "features" not in st.session_state: st.session_state["features"] = None
if "context"  not in st.session_state: st.session_state["context"]  = None

if analyze_btn:
    st.session_state["ticker"] = ticker_input

ticker = st.session_state["ticker"]


# ─────────────────────────────────────────────────────────────
#  HEADER & DATA LOAD
# ─────────────────────────────────────────────────────────────
st.title(f"📈 Options Advisor AI — {ticker}")

# Train ML model (first run only; cached thereafter)
with st.spinner("⚙️ Initialising ML model (first load only)…"):
    model, le = train_model()

with st.spinner(f"📡 Fetching live data for **{ticker}**…"):
    hist, info    = fetch_stock_data(ticker)
    features, ctx = compute_features(ticker)

if hist is None or features is None:
    st.error(f"❌ Could not fetch data for **{ticker}**. "
             "Check the symbol and try again (e.g. AAPL, SPY, NVDA).")
    st.stop()

st.session_state["features"] = features
st.session_state["context"]  = ctx

price        = ctx["price"]
price_chg    = float(hist["Close"].pct_change().iloc[-1] * 100)
ivr          = features["iv_rank"]
hv30_pct     = features["hv_30"] * 100
rsi_val      = features["rsi"]
trend_val    = features["trend_score"]
trend_label  = ("Bullish 📈" if trend_val > 0.2
                else ("Bearish 📉" if trend_val < -0.2 else "Neutral ➡️"))

# ── Top KPIs ─────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price",     f"${price:,.2f}",         f"{price_chg:+.2f}%")
c2.metric("IV Rank",   f"{ivr:.0f} / 100",       help="0=cheapest options ever, 100=most expensive")
c3.metric("HV 30-Day", f"{hv30_pct:.1f}%",       help="Annualised 30-day historical volatility")
c4.metric("RSI (14)",  f"{rsi_val:.0f}",          help="<30 oversold, >70 overbought")
c5.metric("Trend",     trend_label)
c6.metric("Sector",    info.get("sector", "—")[:12] if info else "—")

st.divider()


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Analysis",
    "🤖 ML Recommendation",
    "📋 Options Chain",
    "📚 Strategy Library",
    "📰 News & Sentiment",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — MARKET ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Price & Technicals — {ticker}")
    st.plotly_chart(plot_price_chart(hist, ticker), use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(plot_hv_chart(hist), use_container_width=True)

    with col_right:
        st.subheader("Market Condition Summary")

        def indicator_row(label, value, interp, color):
            st.markdown(
                f"""<div class="metric-box" style="border-left:4px solid {color}">
                    <b>{label}</b>: {value} &nbsp;
                    <span style="color:{color};font-size:13px">{interp}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        indicator_row(
            "RSI (14)", f"{rsi_val:.0f}",
            "Overbought ⚠️" if rsi_val > 70 else ("Oversold 💡" if rsi_val < 30 else "Neutral"),
            "#d63031" if rsi_val > 70 else ("#00b894" if rsi_val < 30 else "#fdcb6e"),
        )
        indicator_row(
            "IV Rank", f"{ivr:.0f}",
            "High — sell strategies 📤" if ivr > 60 else ("Low — buy strategies 📥" if ivr < 30 else "Moderate"),
            "#d63031" if ivr > 60 else ("#00b894" if ivr < 30 else "#fdcb6e"),
        )
        indicator_row(
            "Trend Score", f"{trend_val:+.2f}",
            "Strong Uptrend 🚀" if trend_val > 0.5 else (
            "Uptrend 📈"         if trend_val > 0.2 else (
            "Downtrend 📉"       if trend_val < -0.2 else "Sideways ↔️")),
            "#00b894" if trend_val > 0.2 else ("#d63031" if trend_val < -0.2 else "#fdcb6e"),
        )
        indicator_row(
            "20-Day Return", f"{features['momentum_20d']*100:+.1f}%",
            "Strong Momentum 💪" if features["momentum_20d"] > 0.10 else (
            "Declining ⚠️"       if features["momentum_20d"] < -0.10 else "Moderate"),
            "#00b894" if features["momentum_20d"] > 0.05 else (
            "#d63031" if features["momentum_20d"] < -0.05 else "#fdcb6e"),
        )
        ep = features["earnings_proximity"]
        indicator_row(
            "Earnings Proximity", f"{ep:.0%}",
            "Earnings Very Soon! ⚡" if ep > 0.8 else (
            "Approaching 📅"         if ep > 0.5 else "No Catalyst Detected"),
            "#d63031" if ep > 0.8 else ("#fdcb6e" if ep > 0.5 else "#636e72"),
        )
        indicator_row(
            "News Sentiment", f"{features['sentiment_score']:+.2f}",
            "Positive 🟢" if features["sentiment_score"] > 0.2 else (
            "Negative 🔴" if features["sentiment_score"] < -0.2 else "Neutral 🟡"),
            "#00b894" if features["sentiment_score"] > 0.2 else (
            "#d63031" if features["sentiment_score"] < -0.2 else "#fdcb6e"),
        )

        if level == "Advanced":
            with st.expander("📋 Raw Feature Vector"):
                df_feat = pd.DataFrame(list(features.items()),
                                       columns=["Feature", "Value"])
                df_feat["Value"] = df_feat["Value"].round(6)
                st.dataframe(df_feat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — ML RECOMMENDATION
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 AI-Powered Strategy Recommendation")

    if level == "Beginner":
        st.info(
            f"📘 Our AI has analysed **{ticker}** across 10 market signals — volatility, "
            "trend, momentum, upcoming earnings, and news sentiment — and selected the "
            "best options strategy for current conditions."
        )
    elif level == "Intermediate":
        st.info(
            "The Random Forest model uses **10 engineered features** and was trained on "
            "8 000 synthetic scenarios labelled by an expert rule system covering 11 strategies."
        )
    else:
        with st.expander("🔬 Model Architecture & Training Details"):
            st.markdown("""
| Parameter | Value |
|-----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Trees** | 200 (max_depth = 12, min_samples_leaf = 5) |
| **Training set** | 8 000 synthetic samples |
| **Label method** | Expert rule system → smooth ML decision boundary |
| **Features** | IV Rank, HV-30, HV-10, HV/IV Ratio, Trend, RSI, Mom-20D, Mom-5D, Earnings, Sentiment |
| **Output classes** | 11 options strategies |

**Why synthetic data?** Ground-truth "best strategy" labels require hindsight. We encode practitioner knowledge as rules, generate representative scenarios, and let the RF learn smooth decision boundaries — a standard approach in quantitative finance.
""")

    # ── Recommendations ──────────────────────────────────────
    recs   = get_recommendations(features, model, le)
    medals = ["🥇 Best Match", "🥈 Runner-Up", "🥉 Alternative"]

    for i, rec in enumerate(recs):
        sname  = rec["strategy"]
        conf   = rec["confidence"]
        sinfo  = STRATEGIES.get(sname, {})
        color  = sinfo.get("color", "#ffffff")
        ccolor = COMPLEXITY_COLORS.get(sinfo.get("complexity",""), "#888")
        desc   = sinfo.get("beginner" if level == "Beginner" else "description", "")

        st.markdown(
            f"""<div class="strat-card" style="border-left:5px solid {color}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="color:#b2bec3;font-size:13px">{medals[i]}</span>
                    <span class="badge" style="background:{ccolor};color:#000">
                        {sinfo.get('complexity','—')}
                    </span>
                </div>
                <h3 style="margin:0 0 6px 0">{sinfo.get('emoji','')} {sname}</h3>
                <p style="color:#b2bec3;margin:0 0 10px 0;font-size:14px">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        prog_col, val_col = st.columns([4, 1])
        prog_col.progress(int(conf))
        val_col.markdown(f"**{conf:.1f}%**")

        if level in ("Intermediate", "Advanced"):
            ia, ib, ic = st.columns(3)
            ia.markdown(f"📊 **View:** {sinfo.get('market_view','—')}")
            ib.markdown(f"📈 **IV Pref:** {sinfo.get('iv_pref','—')}")
            ic.markdown(
                f"✅ Max Profit: {sinfo.get('max_profit','—')}  \n"
                f"❌ Max Loss: {sinfo.get('max_loss','—')}"
            )

        st.plotly_chart(plot_pnl(sname, price),
                        use_container_width=True, key=f"pnl_tab2_{i}")
        st.markdown("")

    # ── Feature importance (Advanced) ──────────────────────
    if level == "Advanced":
        st.divider()
        st.plotly_chart(plot_feature_importance(model), use_container_width=True)

    # ── Reasoning bullets ───────────────────────────────────
    st.divider()
    st.subheader("💡 Why These Strategies? — Decision Reasoning")

    reasons = []
    if ivr > 60:
        reasons.append(f"**High IV Rank ({ivr:.0f}/100):** Implied volatility is expensive. "
                        "Selling premium (Iron Condor, Covered Call) is favoured.")
    elif ivr < 30:
        reasons.append(f"**Low IV Rank ({ivr:.0f}/100):** Options are cheap. "
                        "Buying strategies (Long Call/Put, Straddle) offer better value.")
    else:
        reasons.append(f"**Moderate IV Rank ({ivr:.0f}/100):** Spreads that limit both "
                        "cost and risk work well in this environment.")

    if trend_val > 0.3:
        reasons.append(f"**Bullish Trend (score {trend_val:+.2f}):** Price trades above key "
                        "moving averages — directional bullish strategies are appropriate.")
    elif trend_val < -0.3:
        reasons.append(f"**Bearish Trend (score {trend_val:+.2f}):** Price trades below key "
                        "moving averages — bearish protection strategies are appropriate.")
    else:
        reasons.append(f"**Neutral Trend (score {trend_val:+.2f}):** Sideways price action "
                        "favours non-directional strategies.")

    ep = features["earnings_proximity"]
    if ep > 0.6:
        reasons.append(f"**Earnings Approaching ({ep:.0%} proximity):** Volatility typically "
                        "spikes into earnings. Consider vol-buying strategies before the event.")

    sent = features["sentiment_score"]
    if sent > 0.3:
        reasons.append(f"**Positive Sentiment ({sent:+.2f}):** Recent headlines lean bullish, "
                        "supporting upside-biased strategies.")
    elif sent < -0.3:
        reasons.append(f"**Negative Sentiment ({sent:+.2f}):** Recent headlines lean bearish, "
                        "supporting defensive or bearish strategies.")

    for r in reasons:
        st.markdown(f"• {r}")


# ══════════════════════════════════════════════════════════════
#  TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"📋 Live Options Chain — {ticker}")

    expiries, calls_raw, puts_raw = fetch_options_data(ticker)

    if expiries is None:
        st.warning("⚠️ No options data available for this ticker. "
                   "Try a major stock like AAPL, SPY, NVDA.")
    else:
        selected_expiry = st.selectbox("Select Expiry Date", expiries[:10])

        try:
            chain     = yf.Ticker(ticker).option_chain(selected_expiry)
            calls_df  = chain.calls.copy()
            puts_df   = chain.puts.copy()

            keep_cols = ["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]
            calls_df  = calls_df[keep_cols].copy()
            puts_df   = puts_df[keep_cols].copy()

            for df in (calls_df, puts_df):
                df["impliedVolatility"] = (df["impliedVolatility"] * 100).round(1)
                df.rename(columns={
                    "strike":"Strike","lastPrice":"Last","bid":"Bid","ask":"Ask",
                    "volume":"Volume","openInterest":"Open Int","impliedVolatility":"IV %"
                }, inplace=True)

            if level == "Beginner":
                st.info("📘 **Green row = closest to current stock price (ATM).** "
                        "Calls profit when the stock goes UP; Puts profit when it goes DOWN.")

            def highlight_atm(df, spot):
                """Highlight the at-the-money row."""
                styles  = pd.DataFrame("", index=df.index, columns=df.columns)
                closest = (df["Strike"] - spot).abs().idxmin()
                styles.loc[closest] = "background-color:rgba(0,184,148,0.25);font-weight:700"
                return styles

            col_c, col_p = st.columns(2)
            with col_c:
                st.markdown("**📗 Calls**")
                st.dataframe(
                    calls_df.style.apply(highlight_atm, spot=price, axis=None),
                    use_container_width=True, height=480,
                )
            with col_p:
                st.markdown("**📕 Puts**")
                st.dataframe(
                    puts_df.style.apply(highlight_atm, spot=price, axis=None),
                    use_container_width=True, height=480,
                )

            st.caption(f"🟢 Highlighted = nearest ATM (${price:.2f})")

            if level in ("Intermediate", "Advanced"):
                st.divider()
                st.plotly_chart(
                    plot_oi_chart(chain.calls, chain.puts, price),
                    use_container_width=True,
                )
                if level == "Advanced":
                    st.subheader("Volatility Smile / Skew")
                    atm_calls = chain.calls.dropna(subset=["impliedVolatility","strike"])
                    if len(atm_calls) > 0:
                        fig_smile = go.Figure()
                        fig_smile.add_trace(go.Scatter(
                            x=atm_calls["strike"],
                            y=atm_calls["impliedVolatility"] * 100,
                            mode="lines+markers",
                            line=dict(color="#00b894", width=2),
                            name="Call IV",
                        ))
                        atm_puts = chain.puts.dropna(subset=["impliedVolatility","strike"])
                        if len(atm_puts) > 0:
                            fig_smile.add_trace(go.Scatter(
                                x=atm_puts["strike"],
                                y=atm_puts["impliedVolatility"] * 100,
                                mode="lines+markers",
                                line=dict(color="#d63031", width=2),
                                name="Put IV",
                            ))
                        fig_smile.add_vline(x=price, line=dict(color="yellow", dash="dash"),
                                            annotation_text="Spot")
                        fig_smile.update_layout(**DARK, title="IV Smile",
                                               height=300, margin=dict(l=0,r=0,t=40,b=0),
                                               yaxis_title="IV (%)")
                        st.plotly_chart(fig_smile, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load options chain: {e}")


# ══════════════════════════════════════════════════════════════
#  TAB 4 — STRATEGY LIBRARY
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📚 Full Options Strategy Library")

    complexity_filter = st.multiselect(
        "Filter by Complexity",
        ["Beginner", "Intermediate", "Advanced"],
        default=["Beginner", "Intermediate", "Advanced"],
    )

    st.subheader("P&L Diagram Explorer")
    selected_strat = st.selectbox("Choose strategy to visualise:", list(STRATEGIES.keys()))
    st.plotly_chart(plot_pnl(selected_strat, price), use_container_width=True)

    st.divider()
    st.subheader("All Strategies")

    for sname, sinfo in STRATEGIES.items():
        if sinfo["complexity"] not in complexity_filter:
            continue
        color  = sinfo["color"]
        ccolor = COMPLEXITY_COLORS[sinfo["complexity"]]

        with st.expander(
            f"{sinfo['emoji']} **{sname}** "
            f"— {sinfo['complexity']} — {sinfo['market_view']}"
        ):
            left, right = st.columns([2, 1])
            with left:
                if level == "Beginner":
                    st.markdown(f"**{sinfo['beginner']}**")
                else:
                    st.markdown(f"**{sinfo['description']}**")
                st.markdown(f"📊 **Market View:** {sinfo['market_view']}")
                st.markdown(f"📈 **IV Preference:** {sinfo['iv_pref']}")
            with right:
                st.markdown(
                    f"<span class='badge' style='background:{ccolor};color:#000'>"
                    f"{sinfo['complexity']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"✅ **Max Profit:** {sinfo['max_profit']}")
                st.markdown(f"❌ **Max Loss:** {sinfo['max_loss']}")


# ══════════════════════════════════════════════════════════════
#  TAB 5 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"📰 News & Market Sentiment — {ticker}")

    sent_score, headlines = ctx["headlines"], []  # re-use cached values
    sent_score_val = features["sentiment_score"]
    _, headlines   = fetch_news_sentiment(ticker)

    sent_label = ("🟢 Bullish" if sent_score_val > 0.2
                  else ("🔴 Bearish" if sent_score_val < -0.2 else "🟡 Neutral"))

    g_col, h_col = st.columns([1, 2])

    with g_col:
        st.metric("Overall Sentiment", sent_label, f"{sent_score_val:+.2f}")
        if level != "Beginner":
            st.markdown("""
**Score range:** −1.0 (very bearish) → +1.0 (very bullish)  
**Method:** Keyword NLP on recent headlines  
**Source:** Yahoo Finance News API
""")

    with h_col:
        st.plotly_chart(plot_sentiment_gauge(sent_score_val), use_container_width=True)

    st.divider()
    st.subheader("Recent Headlines")

    pos_words = {"surge","beat","record","growth","profit","rise","gain",
                 "strong","bullish","upgrade","buy","soar","rally"}
    neg_words = {"drop","fall","miss","loss","weak","cut","bearish",
                 "downgrade","sell","crash","decline","plunge","warn"}

    if headlines:
        for headline in headlines[:10]:
            words = set(headline.lower().split())
            pos   = len(words & pos_words)
            neg   = len(words & neg_words)
            if pos > neg:
                icon, border = "🟢", "#00b894"
            elif neg > pos:
                icon, border = "🔴", "#d63031"
            else:
                icon, border = "🟡", "#fdcb6e"

            st.markdown(
                f"""<div class="headline-row" style="border-left:3px solid {border}">
                    {icon} {headline}
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No recent headlines found for this ticker.")

    st.divider()
    st.caption(
        "⚠️ Sentiment analysis uses keyword heuristics and is for educational illustration only. "
        "Do not trade solely on this signal."
    )
