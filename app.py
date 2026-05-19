# ============================================================
#  📈 Options Advisor AI v2 — Streamlit App
#  Five upgrades over v1:
#    1. FinBERT NLP sentiment  (replaces keyword heuristic)
#    2. SHAP waterfall explainability
#    3. Polygon.io real IV / VIX data
#    4. Historical options strategy backtester (Black-Scholes)
#    5. Alpaca paper-trading integration
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
from datetime import datetime, date
from collections import Counter

warnings.filterwarnings("ignore")

# ── Path resolution — works locally, on Streamlit Cloud, and Docker ──────────
import sys, os

# Always add the directory containing this file so `utils/` is importable
# regardless of the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Fallback: also add the parent directory.
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

def _stub(*a, **kw):
    return None

# FinBERT ─────────────────────────────────────────────────────────────────────
try:
    from utils.finbert import load_finbert_pipeline, analyze_sentiment, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    load_finbert_pipeline = _stub
    def analyze_sentiment(headlines, pipe=None, **kw):
        pos = {"surge","beat","record","growth","profit","rise","gain","strong",
               "bullish","upgrade","buy","soar","rally"}
        neg = {"drop","fall","miss","loss","weak","cut","bearish","downgrade",
               "sell","crash","decline","warn"}
        details, scores = [], []
        for h in (headlines or []):
            words = set(h.lower().split())
            p = len(words & pos); n = len(words & neg); d = max(p+n,1)
            sv = (p-n)/d
            details.append({
                "headline":h,
                "label":"positive" if sv>0 else ("negative" if sv<0 else "neutral"),
                "positive":max(sv,0),"negative":max(-sv,0),
                "neutral":1-abs(sv),"sentiment":sv,
            })
            scores.append(sv)
        avg = float(sum(scores)/len(scores)) if scores else 0.0
        return {"score":round(avg,4),
                "label":"positive" if avg>.12 else ("negative" if avg<-.12 else "neutral"),
                "method":"keyword","details":details}

# SHAP ────────────────────────────────────────────────────────────────────────
try:
    from utils.shap_utils import compute_shap, plot_shap_waterfall, SHAP_AVAILABLE, FEAT_LABELS
except ImportError:
    SHAP_AVAILABLE = False
    compute_shap = lambda *a, **kw: (None, None)
    plot_shap_waterfall = _stub
    FEAT_LABELS = {}

# Polygon IV ──────────────────────────────────────────────────────────────────
try:
    from utils.iv_fetcher import fetch_polygon_iv, fetch_vix
except ImportError:
    fetch_polygon_iv = _stub
    fetch_vix = _stub

# Backtester ──────────────────────────────────────────────────────────────────
try:
    from utils.backtester import (
        backtest, plot_equity_curve, plot_trade_pnl_bar, plot_drawdown,
        SUPPORTED_STRATEGIES, SCIPY_AVAILABLE,
    )
except ImportError:
    SCIPY_AVAILABLE = False
    SUPPORTED_STRATEGIES = []
    def backtest(*a, **kw):
        return {"error": "scipy not installed. Run: pip install scipy"}
    plot_equity_curve = plot_trade_pnl_bar = plot_drawdown = _stub

# Alpaca ──────────────────────────────────────────────────────────────────────
try:
    from utils.alpaca_trader import (
        get_client as alpaca_get_client,
        get_account, get_positions, get_orders,
        place_market_order, place_limit_order,
        cancel_all_orders, close_position,
        ALPACA_AVAILABLE,
    )
except ImportError:
    ALPACA_AVAILABLE = False
    alpaca_get_client = get_account = get_positions = get_orders = _stub
    place_market_order = place_limit_order = cancel_all_orders = close_position = _stub

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Options Advisor AI v2", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
  .block-container{padding-top:1.5rem}
  .metric-box{background:#161b27;border:1px solid #2d3346;border-radius:10px;padding:14px 18px;margin-bottom:10px}
  .strat-card{background:#161b27;border-radius:12px;padding:20px;margin-bottom:14px}
  .badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:700}
  .headline-row{padding:8px 14px;border-radius:0 8px 8px 0;background:#161b27;margin:5px 0}
  div[data-testid="stExpander"]{border:1px solid #2d3346;border-radius:10px}
</style>""", unsafe_allow_html=True)

DARK = dict(template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117", font=dict(color="#e0e0e0"))

# ─────────────────────────────────────────────────────────────
#  STRATEGY LIBRARY
# ─────────────────────────────────────────────────────────────
STRATEGIES = {
    "Long Call":        {"emoji":"🚀","description":"Buy a call giving the right to purchase shares at strike before expiry.","beginner":"Think the stock will shoot up? Pay a small fee for the right to buy at today's price later.","market_view":"Strongly Bullish","iv_pref":"Low IV — cheap premium","max_profit":"Unlimited","max_loss":"Premium paid","complexity":"Beginner","color":"#00b894"},
    "Long Put":         {"emoji":"📉","description":"Buy a put giving the right to sell shares at strike before expiry.","beginner":"Think the stock will crash? Pay a fee for the right to sell at today's price even if it falls.","market_view":"Strongly Bearish","iv_pref":"Low IV — cheap premium","max_profit":"Strike → $0","max_loss":"Premium paid","complexity":"Beginner","color":"#d63031"},
    "Covered Call":     {"emoji":"💰","description":"Own 100 shares and sell a call to collect premium income.","beginner":"Already own shares? Earn extra income by agreeing to sell at a higher price if reached.","market_view":"Neutral to Slightly Bullish","iv_pref":"High IV — rich premium","max_profit":"Premium + upside to strike","max_loss":"Stock cost minus premium","complexity":"Beginner","color":"#0984e3"},
    "Cash-Secured Put": {"emoji":"🏦","description":"Sell a put while holding enough cash to buy shares if assigned.","beginner":"Want to buy a stock cheaper? Get paid upfront to agree to buy it at a lower price.","market_view":"Neutral to Slightly Bullish","iv_pref":"High IV — rich premium","max_profit":"Premium received","max_loss":"Strike minus premium","complexity":"Beginner","color":"#6c5ce7"},
    "Bull Call Spread": {"emoji":"📈","description":"Buy a lower-strike call and sell a higher-strike call to reduce cost.","beginner":"Moderately bullish? This is a cheaper way to bet on a stock going up.","market_view":"Moderately Bullish","iv_pref":"Neutral","max_profit":"Spread width minus debit","max_loss":"Net debit","complexity":"Intermediate","color":"#00cec9"},
    "Bear Put Spread":  {"emoji":"🐻","description":"Buy a higher-strike put and sell a lower-strike put to reduce cost.","beginner":"Moderately bearish? Cheaper way to bet on a stock falling.","market_view":"Moderately Bearish","iv_pref":"Neutral","max_profit":"Spread width minus debit","max_loss":"Net debit","complexity":"Intermediate","color":"#e17055"},
    "Straddle":         {"emoji":"⚡","description":"Buy a call and put at the same strike. Profits from large moves either way.","beginner":"Big news coming? Bet on a big move without picking a direction.","market_view":"Neutral — large move expected","iv_pref":"Low IV before catalyst","max_profit":"Unlimited","max_loss":"Total premium paid","complexity":"Intermediate","color":"#fdcb6e"},
    "Strangle":         {"emoji":"🌪️","description":"Buy OTM call and OTM put. Cheaper than straddle but needs bigger move.","beginner":"Like a straddle but cheaper — needs a very large move to profit.","market_view":"Neutral — very large move expected","iv_pref":"Very Low IV","max_profit":"Unlimited","max_loss":"Total premium paid","complexity":"Intermediate","color":"#fd79a8"},
    "Iron Condor":      {"emoji":"🦅","description":"Sell OTM call spread + OTM put spread. Profit if stock stays in range.","beginner":"Think the stock will go nowhere? Collect income by betting it stays flat.","market_view":"Neutral — low volatility","iv_pref":"High IV — sell rich premium","max_profit":"Net credit","max_loss":"Spread width minus credit","complexity":"Advanced","color":"#a29bfe"},
    "Butterfly Spread": {"emoji":"🦋","description":"Buy ITM call, sell 2 ATM calls, buy OTM call. Max profit at center strike.","beginner":"Think the stock will land at an exact price? This cheap strategy pays max there.","market_view":"Neutral — price pins at strike","iv_pref":"Low IV","max_profit":"Spread minus debit","max_loss":"Net debit","complexity":"Advanced","color":"#55efc4"},
    "Calendar Spread":  {"emoji":"📅","description":"Sell near-term option, buy same-strike longer-dated option.","beginner":"Expect the stock to stay flat short-term but move later? Profit from the time gap.","market_view":"Neutral short-term","iv_pref":"Flat / rising long-term IV","max_profit":"Extrinsic value differential","max_loss":"Net debit","complexity":"Advanced","color":"#74b9ff"},
}
COMPLEXITY_COLORS = {"Beginner":"#00b894","Intermediate":"#fdcb6e","Advanced":"#d63031"}

# ─────────────────────────────────────────────────────────────
#  TECHNICAL HELPERS
# ─────────────────────────────────────────────────────────────
def compute_rsi(prices, period=14):
    d = prices.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + g / (l + 1e-10)))

def compute_hv(prices, period=30):
    return np.log(prices / prices.shift(1)).rolling(period).std() * np.sqrt(252)

# ─────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.history(period="1y"), t.info
    except Exception:
        return None, {}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_data(ticker):
    try:
        t = yf.Ticker(ticker)
        exp = t.options
        if not exp:
            return None, None, None
        chain = t.option_chain(exp[0])
        return list(exp), chain.calls, chain.puts
    except Exception:
        return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_headlines(ticker):
    try:
        news = yf.Ticker(ticker).news or []
        return [n.get("title","") for n in news[:15] if n.get("title")]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def compute_features(ticker, polygon_iv_data=None):
    hist, info = fetch_stock_data(ticker)
    if hist is None or len(hist) < 60:
        return None, None

    close = hist["Close"].squeeze()
    price = float(close.iloc[-1])
    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50

    trend = sum([price>ma20, price>ma50, price>ma200, ma20>ma50, ma50>ma200]) * 0.2
    trend_score = float(np.clip(trend * 2 - 1, -1, 1))
    mom20 = float(price / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
    mom5  = float(price / close.iloc[-6]  - 1) if len(close) >= 6  else 0.0
    hv30  = float(compute_hv(close, 30).iloc[-1])
    hv10  = float(compute_hv(close, 10).iloc[-1])

    if polygon_iv_data and polygon_iv_data.get("iv_rank") is not None:
        iv_rank = float(polygon_iv_data["iv_rank"])
        atm_iv  = float(polygon_iv_data.get("atm_iv", hv30))
    else:
        hv_s = compute_hv(close, 30).dropna()
        w    = min(len(hv_s), 252)
        iv_rank = float(np.clip(
            (hv30 - hv_s.iloc[-w:].min()) /
            max(hv_s.iloc[-w:].max() - hv_s.iloc[-w:].min(), 1e-6) * 100,
            0, 100))
        atm_iv = hv30
        _, calls_df, _ = fetch_options_data(ticker)
        if calls_df is not None and len(calls_df) > 0:
            c = calls_df.dropna(subset=["impliedVolatility","strike"])
            if len(c) > 0:
                atm_iv = float(c.loc[(c["strike"]-price).abs().idxmin(), "impliedVolatility"])

    hv_iv_ratio = hv30 / max(atm_iv, 1e-8)
    rsi = float(np.clip(compute_rsi(close).iloc[-1], 0, 100))

    ep = 0.0
    try:
        cal = yf.Ticker(ticker).calendar
        ed  = None
        if isinstance(cal, dict) and "Earnings Date" in cal:
            ed = pd.to_datetime(cal["Earnings Date"][0])
        elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
            ed = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
        if ed is not None:
            if hasattr(ed, "tzinfo") and ed.tzinfo:
                ed = ed.tz_localize(None)
            days = (ed - datetime.now()).days
            if 0 < days < 60:
                ep = max(0.0, 1.0 - days / 60.0)
    except Exception:
        pass

    features = {
        "iv_rank":            float(np.clip(iv_rank,      0, 100)),
        "hv_30":              float(np.clip(hv30,          0,   2)),
        "hv_10":              float(np.clip(hv10,          0,   2)),
        "hv_iv_ratio":        float(np.clip(hv_iv_ratio,   0,   3)),
        "trend_score":        float(np.clip(trend_score,  -1,   1)),
        "rsi":                float(np.clip(rsi,           0, 100)),
        "momentum_20d":       float(np.clip(mom20,      -0.5, 0.5)),
        "momentum_5d":        float(np.clip(mom5,       -0.3, 0.3)),
        "earnings_proximity": float(np.clip(ep,            0,   1)),
        "sentiment_score":    0.0,
    }
    context = {
        "price":price, "ma20":ma20, "ma50":ma50, "ma200":ma200,
        "atm_iv":atm_iv, "hv_30":hv30,
        "expiries":fetch_options_data(ticker)[0], "info":info,
    }
    return features, context

# ─────────────────────────────────────────────────────────────
#  ML MODEL
# ─────────────────────────────────────────────────────────────
def _label(iv_rank, hv_iv_ratio, trend, rsi, ep, sent):
    hi    = iv_rank > 55; lo = iv_rank < 30
    sbull = trend > 0.55 and rsi > 60
    bull  = trend > 0.25 or (rsi > 58 and sent > 0.15)
    sbear = trend < -0.55 and rsi < 40
    bear  = trend < -0.25 or (rsi < 42 and sent < -0.15)
    neu   = not bull and not bear
    cat   = ep > 0.60
    if cat and lo:  return "Long Call" if sbull else ("Long Put" if sbear else "Straddle")
    if cat and hi:  return "Bull Call Spread" if bull else ("Bear Put Spread" if bear else "Strangle")
    if sbull and lo: return "Long Call"
    if sbear and lo: return "Long Put"
    if bull  and hi: return "Covered Call"
    if bull:         return "Bull Call Spread"
    if bear  and hi: return "Bear Put Spread"
    if bear  and lo: return "Long Put"
    if neu   and hi: return "Iron Condor"
    if neu   and lo: return "Butterfly Spread" if iv_rank < 15 else "Calendar Spread"
    if neu:          return "Iron Condor"
    return "Cash-Secured Put" if (hi and trend > 0) else "Iron Condor"

@st.cache_resource(show_spinner=False)
def train_model():
    np.random.seed(42)
    n = 8_000
    X = pd.DataFrame({
        "iv_rank":            np.random.uniform(0, 100, n),
        "hv_30":              np.random.uniform(.05, 1.2, n),
        "hv_10":              np.random.uniform(.05, 1.5, n),
        "hv_iv_ratio":        np.random.uniform(.2, 2.5, n),
        "trend_score":        np.random.uniform(-1, 1, n),
        "rsi":                np.random.uniform(20, 80, n),
        "momentum_20d":       np.random.uniform(-.3, .3, n),
        "momentum_5d":        np.random.uniform(-.15,.15, n),
        "earnings_proximity": np.random.beta(1, 5, n),
        "sentiment_score":    np.random.uniform(-1, 1, n),
    })
    y  = [_label(r.iv_rank, r.hv_iv_ratio, r.trend_score, r.rsi,
                 r.earnings_proximity, r.sentiment_score) for r in X.itertuples()]
    le = LabelEncoder()
    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf.fit(X, le.fit_transform(y))
    return clf, le

def get_recs(features, model, le):
    p = model.predict_proba(pd.DataFrame([features]))[0]
    return [{"strategy":le.classes_[i],"confidence":float(p[i]*100),"class_idx":int(i)}
            for i in np.argsort(p)[-3:][::-1]]

# ─────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────
def plot_price_chart(hist, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[.60,.20,.20],
                        subplot_titles=[f"{ticker} Price","Volume","RSI (14)"])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                                  low=hist["Low"], close=hist["Close"],
                                  name="Price", showlegend=False), row=1, col=1)
    for p,c,n in [(20,"#00b894","MA20"),(50,"#fdcb6e","MA50"),(200,"#d63031","MA200")]:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(p).mean(),
                                  name=n, line=dict(color=c, width=1.5)), row=1, col=1)
    bar_c = ["#00b894" if c >= o else "#d63031" for c,o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=bar_c,
                          name="Volume", showlegend=False), row=2, col=1)
    rsi = compute_rsi(hist["Close"])
    fig.add_trace(go.Scatter(x=hist.index, y=rsi, line=dict(color="#a29bfe",width=1.5),
                              name="RSI", showlegend=False), row=3, col=1)
    for yv, col in [(70,"red"),(30,"green")]:
        fig.add_hline(y=yv, line=dict(color=col,dash="dash",width=1), row=3, col=1)
    fig.update_layout(**DARK, height=660, xaxis_rangeslider_visible=False,
                       margin=dict(l=0,r=0,t=30,b=0))
    return fig

def plot_hv_chart(hist):
    hv30 = compute_hv(hist["Close"], 30) * 100
    hv10 = compute_hv(hist["Close"], 10) * 100
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hv30, name="HV 30-Day", line=dict(color="#00b894",width=2)))
    fig.add_trace(go.Scatter(x=hist.index, y=hv10, name="HV 10-Day", line=dict(color="#fdcb6e",width=1.5,dash="dot")))
    fig.update_layout(**DARK, title="Historical Volatility (%)", height=290, margin=dict(l=0,r=0,t=40,b=0))
    return fig

def plot_pnl_diagram(strategy_name, spot):
    K = round(spot); pr = spot*0.03; sw = spot*0.05
    prices = np.linspace(spot*.70, spot*1.30, 400)
    fns = {
        "Long Call":        lambda S: np.maximum(-pr, S-K-pr),
        "Long Put":         lambda S: np.maximum(-pr, K-S-pr),
        "Covered Call":     lambda S: np.minimum(K-spot+pr, S-spot+pr),
        "Cash-Secured Put": lambda S: np.where(S>=K, pr, S-K+pr),
        "Bull Call Spread": lambda S: np.clip(S-K, -pr, sw-pr),
        "Bear Put Spread":  lambda S: np.clip(K-S, -pr, sw-pr),
        "Straddle":         lambda S: np.abs(S-K)-2*pr,
        "Strangle":         lambda S: (np.maximum(S-(K+sw*.5),0)+np.maximum((K-sw*.5)-S,0)-1.5*pr),
        "Iron Condor":      lambda S: np.where(np.abs(S-K)<sw, pr, pr-np.maximum(np.abs(S-K)-sw,0)*(pr/(sw*.5+1e-8))),
        "Butterfly Spread": lambda S: np.maximum(0,sw-np.abs(S-K))-pr*.5,
        "Calendar Spread":  lambda S: pr*.8-np.abs(S-K)*.015,
    }
    pnl = fns.get(strategy_name, lambda S: np.zeros_like(S))(prices)
    pp  = np.where(pnl>=0, pnl, np.nan)
    np_ = np.where(pnl<0,  pnl, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=pp,  fill="tozeroy", fillcolor="rgba(0,184,148,0.18)", line=dict(color="#00b894",width=2), name="Profit"))
    fig.add_trace(go.Scatter(x=prices, y=np_, fill="tozeroy", fillcolor="rgba(214,48,49,0.18)",  line=dict(color="#d63031",width=2), name="Loss"))
    fig.add_hline(y=0, line=dict(color="white",dash="dash",width=1))
    fig.add_vline(x=spot, line=dict(color="yellow",dash="dot",width=1.5), annotation_text="Now")
    fig.update_layout(**DARK, title=f"{strategy_name} — Illustrative P&L at Expiry",
                       xaxis_title="Stock Price", yaxis_title="P&L ($)", height=300,
                       showlegend=False, margin=dict(l=0,r=0,t=40,b=40))
    return fig

def plot_sentiment_gauge(score):
    val = score * 50 + 50
    fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
        domain={"x":[0,1],"y":[0,1]}, title={"text":"Sentiment (0–100)"},
        gauge={"axis":{"range":[0,100]}, "bar":{"color":"#00b894" if score>=0 else "#d63031"},
               "steps":[{"range":[0,33],"color":"#2d1b1b"},{"range":[33,66],"color":"#1e2020"},{"range":[66,100],"color":"#1b2d1b"}],
               "threshold":{"line":{"color":"white","width":3},"thickness":.75,"value":50}}))
    fig.update_layout(**DARK, height=250, margin=dict(l=20,r=20,t=40,b=10))
    return fig

def plot_oi_chart(calls, puts, spot):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=calls["strike"], y=calls["openInterest"], name="Calls OI", marker_color="#00b894", opacity=.8))
    fig.add_trace(go.Bar(x=puts["strike"],  y=puts["openInterest"],  name="Puts OI",  marker_color="#d63031", opacity=.8))
    fig.add_vline(x=spot, line=dict(color="yellow",dash="dash"), annotation_text="Spot")
    fig.update_layout(**DARK, title="Open Interest by Strike", barmode="overlay", height=320, margin=dict(l=0,r=0,t=40,b=0))
    return fig

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Options Advisor AI")
    st.caption("v2 · FinBERT · SHAP · Polygon · Backtester · Alpaca")
    st.divider()

    user_level = st.radio("👤 Experience Level",
                          ["🟢 Beginner","🟡 Intermediate","🔴 Advanced"])
    level = user_level.split()[1]

    st.divider()
    ticker_input = st.text_input("🔍 Ticker", value="AAPL", max_chars=10).upper().strip()
    cols_ = st.columns(3)
    for i, qt in enumerate(["SPY","AAPL","NVDA","TSLA","QQQ","META","AMD","GLD","IWM"]):
        if cols_[i % 3].button(qt, key=f"qt_{qt}", use_container_width=True):
            ticker_input = qt
    analyze_btn = st.button("🔬 Analyse Now", type="primary", use_container_width=True)

    st.divider()
    with st.expander("🔑 API Keys (Optional)"):
        st.caption("Enables Polygon IV data and Alpaca paper trading.")
        polygon_key   = st.text_input("Polygon.io API Key", type="password", key="pol_key",  placeholder="free at polygon.io")
        alpaca_key    = st.text_input("Alpaca API Key",      type="password", key="alp_key",  placeholder="alpaca.markets → Paper")
        alpaca_secret = st.text_input("Alpaca Secret Key",   type="password", key="alp_sec",  placeholder="alpaca.markets → Paper")
        st.markdown("[Polygon free key →](https://polygon.io)  |  [Alpaca signup →](https://alpaca.markets)")

    st.divider()
    ok  = "🟢"
    na  = "🟡"
    st.markdown("**Module Status:**")
    st.markdown(f"{ok if TRANSFORMERS_AVAILABLE else na} FinBERT {'active' if TRANSFORMERS_AVAILABLE else '(`pip install transformers torch`)'}")
    st.markdown(f"{ok if SHAP_AVAILABLE        else na} SHAP    {'active' if SHAP_AVAILABLE else '(`pip install shap`)'}")
    st.markdown(f"{ok if SCIPY_AVAILABLE        else na} Scipy   {'active' if SCIPY_AVAILABLE else '(`pip install scipy`)'}")
    st.markdown(f"{ok if ALPACA_AVAILABLE       else na} Alpaca  {'active' if ALPACA_AVAILABLE else '(`pip install alpaca-py`)'}")
    st.divider()
    st.caption("⚠️ Educational only. Not financial advice.")

# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in [("ticker","AAPL"),("features",None),("context",None),
             ("sentiment_result",None),("alpaca_client",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if analyze_btn:
    st.session_state.update({"ticker":ticker_input,"features":None,"sentiment_result":None})

ticker = st.session_state["ticker"]

# ─────────────────────────────────────────────────────────────
#  BOOT: model + FinBERT
# ─────────────────────────────────────────────────────────────
with st.spinner("⚙️ Initialising ML model…"):
    model, le = train_model()

@st.cache_resource(show_spinner=False)
def _get_finbert():
    return load_finbert_pipeline() if TRANSFORMERS_AVAILABLE else None

finbert_pipe = _get_finbert()

# ─────────────────────────────────────────────────────────────
#  FETCH DATA
# ─────────────────────────────────────────────────────────────
with st.spinner(f"📡 Fetching data for **{ticker}**…"):
    hist, info = fetch_stock_data(ticker)
    polygon_iv = None
    if polygon_key:
        polygon_iv = fetch_polygon_iv(ticker, polygon_key)
    features, ctx = compute_features(ticker, polygon_iv)

if hist is None or features is None:
    st.error(f"❌ Cannot fetch data for **{ticker}**. Try AAPL, SPY, or NVDA.")
    st.stop()

# Sentiment analysis
headlines = fetch_headlines(ticker)
if st.session_state["sentiment_result"] is None or analyze_btn:
    msg = "🧠 Running FinBERT sentiment…" if finbert_pipe else "📰 Analysing headlines…"
    with st.spinner(msg):
        sent = analyze_sentiment(headlines, pipe=finbert_pipe)
    st.session_state["sentiment_result"] = sent
else:
    sent = st.session_state["sentiment_result"]

features["sentiment_score"] = float(np.clip(sent["score"], -1, 1))
st.session_state.update({"features":features,"context":ctx})

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
price      = ctx["price"]
price_chg  = float(hist["Close"].pct_change().iloc[-1] * 100)
ivr        = features["iv_rank"]
hv30_pct   = features["hv_30"] * 100
rsi_val    = features["rsi"]
trend_val  = features["trend_score"]
iv_src     = "polygon.io 🟢" if (polygon_iv and polygon_iv.get("source")=="polygon") else "HV proxy"
trend_lbl  = "Bullish 📈" if trend_val>.2 else ("Bearish 📉" if trend_val<-.2 else "Neutral ➡️")
sent_icon  = "🟢" if sent["score"]>.1 else ("🔴" if sent["score"]<-.1 else "🟡")

st.title(f"📈 Options Advisor AI v2 — {ticker}")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Price",    f"${price:,.2f}",      f"{price_chg:+.2f}%")
c2.metric("IV Rank",  f"{ivr:.0f}/100",      help=f"Source: {iv_src}")
c3.metric("HV 30D",   f"{hv30_pct:.1f}%")
c4.metric("RSI",      f"{rsi_val:.0f}")
c5.metric("Trend",    trend_lbl)
c6.metric("Sentiment",f"{sent_icon} {sent['score']:+.2f}", sent["method"].upper())
st.divider()

# ─────────────────────────────────────────────────────────────
#  8 TABS
# ─────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "📊 Market Analysis",
    "🤖 ML Recommendation",
    "🔬 SHAP Explainability",
    "📋 Options Chain",
    "📚 Strategy Library",
    "📰 FinBERT Sentiment",
    "📉 Backtester",
    "🏦 Paper Trading",
])

# ════════════════════════════════════════════════════════════
#  TAB 1 · MARKET ANALYSIS
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Price & Technicals — {ticker}")
    if polygon_iv:
        pa,pb,pc,pd_ = st.columns(4)
        pa.metric("ATM IV (Polygon)", f"{polygon_iv['atm_iv']*100:.1f}%")
        pb.metric("IV Rank (Polygon)",f"{polygon_iv['iv_rank']:.0f}/100")
        pc.metric("Put/Call Skew",    f"{polygon_iv.get('skew',0)*100:+.1f}%")
        vix = fetch_vix(polygon_key) if polygon_key else None
        pd_.metric("VIX", f"{vix:.1f}" if vix else "—")
        st.divider()

    st.plotly_chart(plot_price_chart(hist, ticker), use_container_width=True)
    cL, cR = st.columns(2)
    with cL:
        st.plotly_chart(plot_hv_chart(hist), use_container_width=True)
    with cR:
        st.subheader("Condition Summary")
        def ind(lbl, val, interp, col):
            st.markdown(f'<div class="metric-box" style="border-left:4px solid {col}"><b>{lbl}</b>: {val}&nbsp;<span style="color:{col};font-size:13px">{interp}</span></div>', unsafe_allow_html=True)
        ind("RSI", f"{rsi_val:.0f}", "Overbought⚠️" if rsi_val>70 else ("Oversold💡" if rsi_val<30 else "Neutral"), "#d63031" if rsi_val>70 else ("#00b894" if rsi_val<30 else "#fdcb6e"))
        ind("IV Rank", f"{ivr:.0f}", "High—sell📤" if ivr>60 else ("Low—buy📥" if ivr<30 else "Moderate"), "#d63031" if ivr>60 else ("#00b894" if ivr<30 else "#fdcb6e"))
        ind("Trend", f"{trend_val:+.2f}", "Uptrend🚀" if trend_val>.5 else ("Up📈" if trend_val>.2 else ("Down📉" if trend_val<-.2 else "Sideways↔️")), "#00b894" if trend_val>.2 else ("#d63031" if trend_val<-.2 else "#fdcb6e"))
        ind("20D Return", f"{features['momentum_20d']*100:+.1f}%", "Strong💪" if features['momentum_20d']>.10 else ("Declining⚠️" if features['momentum_20d']<-.10 else "Moderate"), "#00b894" if features['momentum_20d']>.05 else ("#d63031" if features['momentum_20d']<-.05 else "#fdcb6e"))
        ep = features["earnings_proximity"]
        ind("Earnings", f"{ep:.0%}", "Soon⚡" if ep>.8 else ("Approaching📅" if ep>.5 else "No catalyst"), "#d63031" if ep>.8 else ("#fdcb6e" if ep>.5 else "#636e72"))
        ind(f"Sentiment ({sent['method'].upper()})", f"{features['sentiment_score']:+.2f}", "Positive🟢" if sent['score']>.15 else ("Negative🔴" if sent['score']<-.15 else "Neutral🟡"), "#00b894" if sent['score']>.15 else ("#d63031" if sent['score']<-.15 else "#fdcb6e"))


# ════════════════════════════════════════════════════════════
#  TAB 2 · ML RECOMMENDATION
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 AI Strategy Recommendation")
    if level=="Beginner": st.info(f"📘 AI analysed **{ticker}** across 10 signals and selected the best strategies.")
    elif level=="Intermediate": st.info("Random Forest (200 trees) with FinBERT sentiment + Polygon IV as live features.")
    else:
        with st.expander("🔬 Model Details"):
            st.markdown("""| Parameter | Value |\n|---|---|\n| Algorithm | Random Forest (200 trees, depth 12) |\n| Training | 8 000 synthetic expert-labelled samples |\n| Sentiment | FinBERT (ProsusAI/finbert) real NLP |\n| IV | Polygon.io live IV if key provided |\n| Classes | 11 strategies |""")

    recs   = get_recs(features, model, le)
    medals = ["🥇 Best Match","🥈 Runner-Up","🥉 Alternative"]
    for i, rec in enumerate(recs):
        sn = rec["strategy"]; conf = rec["confidence"]
        si = STRATEGIES.get(sn,{}); color = si.get("color","#fff")
        desc = si.get("beginner" if level=="Beginner" else "description","")
        st.markdown(f'<div class="strat-card" style="border-left:5px solid {color}"><div style="display:flex;justify-content:space-between;margin-bottom:6px"><span style="color:#b2bec3;font-size:13px">{medals[i]}</span><span class="badge" style="background:{COMPLEXITY_COLORS.get(si.get("complexity",""),"#888")};color:#000">{si.get("complexity","")}</span></div><h3 style="margin:0 0 6px 0">{si.get("emoji","")} {sn}</h3><p style="color:#b2bec3;margin:0;font-size:14px">{desc}</p></div>', unsafe_allow_html=True)
        pc_, vc_ = st.columns([4,1]); pc_.progress(int(conf)); vc_.markdown(f"**{conf:.1f}%**")
        if level in ("Intermediate","Advanced"):
            ia,ib,ic = st.columns(3)
            ia.markdown(f"📊 {si.get('market_view','—')}"); ib.markdown(f"📈 {si.get('iv_pref','—')}")
            ic.markdown(f"✅ {si.get('max_profit','—')}  \n❌ {si.get('max_loss','—')}")
        st.plotly_chart(plot_pnl_diagram(sn, price), use_container_width=True, key=f"pnl2_{i}")

    st.divider(); st.subheader("💡 Decision Reasoning")
    reasons = []
    if ivr>60:   reasons.append(f"**High IV Rank ({ivr:.0f}/100)** → Premium expensive. Selling strategies favoured.")
    elif ivr<30: reasons.append(f"**Low IV Rank ({ivr:.0f}/100)** → Options cheap. Buying strategies favoured.")
    else:        reasons.append(f"**Moderate IV Rank ({ivr:.0f}/100)** → Spreads work well.")
    if trend_val>.3:   reasons.append(f"**Bullish Trend ({trend_val:+.2f})** → Price above key MAs.")
    elif trend_val<-.3: reasons.append(f"**Bearish Trend ({trend_val:+.2f})** → Price below key MAs.")
    else:               reasons.append(f"**Neutral Trend ({trend_val:+.2f})** → Sideways action favours non-directional strategies.")
    if ep>.6: reasons.append(f"**Earnings Approaching ({ep:.0%})** → Vol spikes expected. Vol-buying favoured.")
    ss = features["sentiment_score"]
    if ss>.2:   reasons.append(f"**Positive FinBERT Sentiment ({ss:+.2f})** → Headlines lean bullish.")
    elif ss<-.2: reasons.append(f"**Negative FinBERT Sentiment ({ss:+.2f})** → Headlines lean bearish.")
    for r in reasons: st.markdown(f"• {r}")


# ════════════════════════════════════════════════════════════
#  TAB 3 · SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 SHAP Explainability")
    if not SHAP_AVAILABLE:
        st.warning("Install SHAP: `pip install shap`")
    else:
        if level=="Beginner":
            st.info("📘 **SHAP** shows which market conditions pushed the AI toward each strategy. **Green bars** support it; **red bars** work against it.")
        else:
            st.info("TreeExplainer produces exact SHAP values for the Random Forest. Each bar = one feature's additive contribution to the strategy's predicted probability.")

        recs_shap = get_recs(features, model, le)
        sel_strat = st.selectbox(
            "Explain strategy:",
            [r["strategy"] for r in recs_shap],
            format_func=lambda s: f"{STRATEGIES[s]['emoji']} {s} ({next(r['confidence'] for r in recs_shap if r['strategy']==s):.1f}%)",
        )
        sel_rec   = next(r for r in recs_shap if r["strategy"]==sel_strat)
        class_idx = sel_rec["class_idx"]

        with st.spinner("Computing SHAP values…"):
            sv, bv = compute_shap(model, features, class_idx)

        if sv is not None:
            st.plotly_chart(plot_shap_waterfall(sv, features, bv, sel_strat, STRATEGIES[sel_strat]["color"]), use_container_width=True)
            if level in ("Intermediate","Advanced"):
                feat_names = [FEAT_LABELS.get(k,k) for k in features.keys()]
                df_s = pd.DataFrame({
                    "Feature":   feat_names,
                    "Value":     [round(v,4) for v in features.values()],
                    "SHAP":      [round(float(v),4) for v in sv],
                    "Direction": ["↑ supports" if float(v)>0 else ("↓ opposes" if float(v)<0 else "—") for v in sv],
                }).sort_values("SHAP", key=abs, ascending=False)
                st.dataframe(df_s, use_container_width=True)
            st.divider()
            st.subheader("Global Feature Importance")
            fi = model.feature_importances_
            fl = [FEAT_LABELS.get(k,k) for k in features.keys()]
            df_fi = pd.DataFrame({"Feature":fl,"Importance":fi}).sort_values("Importance")
            fig_fi = go.Figure(go.Bar(y=df_fi["Feature"], x=df_fi["Importance"], orientation="h",
                                      marker_color="#a29bfe", text=[f"{v:.4f}" for v in df_fi["Importance"]], textposition="outside"))
            fig_fi.update_layout(**DARK, title="Random Forest Feature Importance", height=360, margin=dict(l=10,r=70,t=50,b=30))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.error("Could not compute SHAP values.")


# ════════════════════════════════════════════════════════════
#  TAB 4 · OPTIONS CHAIN
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"📋 Live Options Chain — {ticker}")
    expiries, calls_raw, puts_raw = fetch_options_data(ticker)
    if expiries is None:
        st.warning("⚠️ Options data unavailable for this ticker.")
    else:
        sel_exp = st.selectbox("Expiry", expiries[:10])
        try:
            chain = yf.Ticker(ticker).option_chain(sel_exp)
            cdf   = chain.calls[["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]].copy()
            pdf   = chain.puts[["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]].copy()
            for df in (cdf,pdf):
                df["impliedVolatility"] = (df["impliedVolatility"]*100).round(1)
                df.rename(columns={"strike":"Strike","lastPrice":"Last","bid":"Bid","ask":"Ask","volume":"Volume","openInterest":"Open Int","impliedVolatility":"IV %"}, inplace=True)
            def hl_atm(df, sp):
                s = pd.DataFrame("", index=df.index, columns=df.columns)
                s.loc[(df["Strike"]-sp).abs().idxmin()] = "background-color:rgba(0,184,148,0.25);font-weight:700"
                return s
            cc,pp = st.columns(2)
            with cc:
                st.markdown("**📗 Calls**")
                st.dataframe(cdf.style.apply(hl_atm, sp=price, axis=None), use_container_width=True, height=460)
            with pp:
                st.markdown("**📕 Puts**")
                st.dataframe(pdf.style.apply(hl_atm, sp=price, axis=None), use_container_width=True, height=460)
            st.caption(f"🟢 ATM = ${price:.2f}")
            if level in ("Intermediate","Advanced"):
                st.plotly_chart(plot_oi_chart(chain.calls, chain.puts, price), use_container_width=True)
        except Exception as e:
            st.error(f"Options chain error: {e}")


# ════════════════════════════════════════════════════════════
#  TAB 5 · STRATEGY LIBRARY
# ════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📚 Strategy Library")
    cf = st.multiselect("Filter:", ["Beginner","Intermediate","Advanced"], default=["Beginner","Intermediate","Advanced"])
    sel = st.selectbox("P&L diagram:", list(STRATEGIES.keys()))
    st.plotly_chart(plot_pnl_diagram(sel, price), use_container_width=True)
    st.divider()
    for sn, si in STRATEGIES.items():
        if si["complexity"] not in cf: continue
        with st.expander(f"{si['emoji']} **{sn}** — {si['complexity']} — {si['market_view']}"):
            la,ra = st.columns([2,1])
            with la: st.markdown(f"**{si['beginner' if level=='Beginner' else 'description']}**"); st.markdown(f"📊 {si['market_view']}  |  📈 {si['iv_pref']}")
            with ra: st.markdown(f"<span class='badge' style='background:{COMPLEXITY_COLORS[si['complexity']]};color:#000'>{si['complexity']}</span>", unsafe_allow_html=True); st.markdown(f"✅ {si['max_profit']}  \n❌ {si['max_loss']}")


# ════════════════════════════════════════════════════════════
#  TAB 6 · FINBERT SENTIMENT
# ════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"📰 FinBERT Sentiment — {ticker}")
    method_badge = "🟢 **FinBERT NLP** (ProsusAI/finbert)" if sent["method"]=="finbert" else "🟡 **Keyword Heuristic** — install `transformers torch` for FinBERT"
    st.markdown(method_badge)
    if sent["method"]=="finbert": st.success("✅ Full financial-domain NLP model active")
    else: st.warning("Run `pip install transformers torch` and restart to enable FinBERT.")

    ga,ha = st.columns([1,2])
    with ga:
        score = sent["score"]; lbl = "🟢 Bullish" if score>.12 else ("🔴 Bearish" if score<-.12 else "🟡 Neutral")
        st.metric("Aggregate", lbl, f"{score:+.4f}")
        if level!="Beginner": st.markdown(f"**Method:** {sent['method'].upper()}  \n**Headlines:** {len(sent['details'])}")
    with ha: st.plotly_chart(plot_sentiment_gauge(score), use_container_width=True)

    st.divider(); st.subheader("Per-Headline Scores")
    for d in sent.get("details",[]):
        bord = "#00b894" if d["label"]=="positive" else ("#d63031" if d["label"]=="negative" else "#fdcb6e")
        icon = "🟢" if d["label"]=="positive" else ("🔴" if d["label"]=="negative" else "🟡")
        extra = f"pos={d['positive']:.2f} neg={d['negative']:.2f} neu={d['neutral']:.2f}" if sent["method"]=="finbert" else f"score={d['sentiment']:+.2f}"
        st.markdown(f'<div class="headline-row" style="border-left:3px solid {bord}">{icon} {d["headline"]}<br><small style="color:#636e72">{extra}</small></div>', unsafe_allow_html=True)

    if level in ("Intermediate","Advanced") and sent["method"]=="finbert" and sent.get("details"):
        st.divider(); st.subheader("Distribution")
        counts = Counter(d["label"] for d in sent["details"])
        fig_d  = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()),
                                   marker_color=["#00b894" if k=="positive" else "#d63031" if k=="negative" else "#fdcb6e" for k in counts]))
        fig_d.update_layout(**DARK, title="Headline Label Distribution", height=250, margin=dict(l=0,r=0,t=40,b=30))
        st.plotly_chart(fig_d, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 7 · BACKTESTER
# ════════════════════════════════════════════════════════════
with tab7:
    st.subheader("📉 Historical Strategy Backtester")
    if not SCIPY_AVAILABLE:
        st.warning("Install scipy for Black-Scholes: `pip install scipy`")
    else:
        if level=="Beginner": st.info("📘 Test how an options strategy would have performed historically.")
        elif level=="Intermediate": st.info("Uses Black-Scholes with 20-day HV as IV proxy. Entry every N days; exit at expiry or stop/TP.")
        else: st.info("Black-Scholes pricer with HV(20) IV proxy. Fixed % position sizing. Buy-and-hold benchmark included.")

        ba,bb = st.columns([2,1])
        with ba:
            bt_strat = st.selectbox("Strategy:", SUPPORTED_STRATEGIES)
            bc,bd    = st.columns(2)
            bt_start = bc.date_input("Start", value=date(2022,1,1), min_value=date(2015,1,1))
            bt_end   = bd.date_input("End",   value=date(2024,12,31), min_value=date(2015,1,1))
        with bb:
            hold_days = st.slider("Holding Period (days)", 10, 90, 30)
            init_cap  = st.number_input("Starting Capital ($)", 10_000, 1_000_000, 50_000, step=10_000)
            stop_p    = st.slider("Stop-Loss (%)", 10, 100, 50) / 100
            tp_p      = st.slider("Take-Profit (%)", 50, 300, 100) / 100
            otm_p     = st.slider("OTM % / Wing Width", 1, 15, 5) / 100

        if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
            with st.spinner(f"Backtesting {bt_strat} on {ticker}…"):
                res = backtest(ticker=ticker, strategy_name=bt_strat,
                               start_date=str(bt_start), end_date=str(bt_end),
                               holding_days=hold_days, initial_capital=float(init_cap),
                               stop_pct=stop_p, take_profit_pct=tp_p, otm_pct=otm_p)

            if "error" in res:
                st.error(res["error"])
            else:
                m = res["metrics"]
                st.subheader("Performance Metrics")
                m1,m2,m3,m4,m5,m6 = st.columns(6)
                m1.metric("Total Return",  f"{m['total_return_pct']:+.1f}%",  f"BH: {m['bh_return_pct']:+.1f}%")
                m2.metric("Win Rate",       f"{m['win_rate_pct']:.1f}%")
                m3.metric("Profit Factor",  f"{m['profit_factor']:.2f}",      help=">1 = profitable")
                m4.metric("Sharpe",         f"{m['sharpe_ratio']:.2f}")
                m5.metric("Max Drawdown",   f"{m['max_drawdown_pct']:.1f}%")
                m6.metric("Final Capital",  f"${m['final_capital']:,.0f}")

                st.plotly_chart(plot_equity_curve(res["equity_curve"], res["bh_curve"], bt_strat, ticker), use_container_width=True)
                ca,cb = st.columns(2)
                with ca: st.plotly_chart(plot_trade_pnl_bar(res["trades"]), use_container_width=True)
                with cb: st.plotly_chart(plot_drawdown(res["equity_curve"]), use_container_width=True)

                if level in ("Intermediate","Advanced"):
                    st.subheader("Trade Log")
                    tdf = pd.DataFrame(res["trades"])
                    show_cols = [c for c in ["entry_date","exit_date","entry_spot","exit_spot","sigma","pnl_dollar","pnl_pct","won","exit_reason"] if c in tdf.columns]
                    tdf = tdf[show_cols]; tdf.columns = [c.replace("_"," ").title() for c in tdf.columns]
                    def _cpnl(val):
                        try: return "color:#00b894" if float(val)>=0 else "color:#d63031"
                        except: return ""
                    pnl_cols = [c for c in tdf.columns if "Pnl" in c]
                    st.dataframe(tdf.style.applymap(_cpnl, subset=pnl_cols) if pnl_cols else tdf, use_container_width=True)
                    st.caption(f"Avg win ${m['avg_win_usd']:+,.2f}  |  Avg loss ${m['avg_loss_usd']:+,.2f}  |  {m['total_trades']} trades")

                st.caption("⚠️ Illustrative only. Real options pricing includes spreads, pin risk, and early exercise.")


# ════════════════════════════════════════════════════════════
#  TAB 8 · PAPER TRADING
# ════════════════════════════════════════════════════════════
with tab8:
    st.subheader("🏦 Alpaca Paper Trading")
    if not ALPACA_AVAILABLE:
        st.warning("`alpaca-py` not installed. Run: `pip install alpaca-py`")
    elif not alpaca_key or not alpaca_secret:
        st.info("Enter your Alpaca paper-trading API keys in the sidebar.\n\n1. Signup free at [alpaca.markets](https://alpaca.markets)\n2. Paper Trading → API Keys → Generate\n3. Paste into sidebar")
    else:
        if st.session_state["alpaca_client"] is None:
            st.session_state["alpaca_client"] = alpaca_get_client(alpaca_key, alpaca_secret)
        client = st.session_state["alpaca_client"]
        acct   = get_account(client)

        if "error" in acct:
            st.error(f"❌ Connection failed: {acct['error']}")
        else:
            st.success("✅ Connected to Alpaca Paper Trading")
            ac1,ac2,ac3,ac4 = st.columns(4)
            ac1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
            ac2.metric("Cash",            f"${acct['cash']:,.2f}")
            ac3.metric("Buying Power",    f"${acct['buying_power']:,.2f}")
            ac4.metric("Total P&L",       f"${acct['pnl_dollar']:+,.2f}", f"{acct['pnl_pct']:+.2f}%")
            st.divider()

            pt1, pt2 = st.columns([1,1])
            with pt1:
                st.subheader("📊 Open Positions")
                if st.button("🔄 Refresh"): st.rerun()
                positions = get_positions(client)
                if positions:
                    pdf_ = pd.DataFrame(positions)
                    def _cp(v):
                        try: return "color:#00b894" if float(v)>=0 else "color:#d63031"
                        except: return ""
                    plc = [c for c in pdf_.columns if "pnl" in c.lower()]
                    st.dataframe(pdf_.style.applymap(_cp, subset=plc), use_container_width=True, height=280)
                    if level in ("Intermediate","Advanced"):
                        csym = st.selectbox("Close position:", [p["symbol"] for p in positions])
                        if st.button(f"❌ Close {csym}"):
                            r = close_position(client, csym)
                            st.success(r["message"]) if r.get("success") else st.error(r.get("error"))
                else:
                    st.info("No open positions.")

            with pt2:
                st.subheader("📝 Place Paper Order")
                osym  = st.text_input("Symbol", value=ticker, key="osym").upper()
                oside = st.radio("Side", ["Buy","Sell"], horizontal=True, key="oside")
                otype = st.radio("Type", ["Market","Limit"], horizontal=True, key="otype")
                oqty  = st.number_input("Qty (shares)", 1, 10000, 1, key="oqty")
                olmt  = st.number_input("Limit Price ($)", 0.01, 99999.0, round(price,2), 0.01, key="olmt") if otype=="Limit" else None

                top_r = get_recs(features, model, le)[0]
                si_r  = STRATEGIES.get(top_r["strategy"],{})
                st.markdown(f"AI recommends **{si_r.get('emoji','')} {top_r['strategy']}** ({top_r['confidence']:.0f}% conf) for {ticker}")
                st.caption("Note: places a stock order as proxy. Full options ordering requires dedicated options API workflow.")

                if st.button("✅ Submit Order", type="primary", use_container_width=True):
                    with st.spinner("Placing order…"):
                        res = place_market_order(client, osym, oqty, oside.lower()) if otype=="Market" else place_limit_order(client, osym, oqty, oside.lower(), olmt)
                    if res.get("success"): st.success(f"✅ {res.get('order_id','—')} | {res.get('status','—')}")
                    else: st.error(f"❌ {res.get('error','Unknown')}")

            st.divider()
            st.subheader("📋 Recent Orders")
            oc1, oc2 = st.columns([3,1])
            with oc2:
                if st.button("❌ Cancel All"):
                    r = cancel_all_orders(client)
                    st.success(r.get("message")) if r.get("success") else st.error(r.get("error"))
            orders = get_orders(client, 15)
            if orders: st.dataframe(pd.DataFrame(orders), use_container_width=True, height=300)
            else: st.info("No recent orders.")
