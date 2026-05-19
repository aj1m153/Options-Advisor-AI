# 📈 Options Advisor AI

A Streamlit app that uses Machine Learning to recommend the optimal options strategy for any US stock or ETF, based on live market data, historical volatility, options chain data, and news sentiment.

---

## 🚀 Quick Start

### 1. Clone / Download
Place `app.py` and `requirements.txt` in the same folder.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                      │
│  Sidebar: Ticker + Experience Level + Analyse button        │
├──────────┬──────────┬────────────┬──────────┬──────────────┤
│ Market   │ ML Rec.  │ Options    │ Strategy │ News &       │
│ Analysis │ (Tab 2)  │ Chain      │ Library  │ Sentiment    │
│ (Tab 1)  │          │ (Tab 3)    │ (Tab 4)  │ (Tab 5)      │
└──────────┴──────────┴────────────┴──────────┴──────────────┘
                         │
              ┌──────────▼──────────┐
              │   FEATURE ENGINE    │
              │ IV Rank, HV, RSI,   │
              │ Trend, Momentum,    │
              │ Earnings, Sentiment │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  RANDOM FOREST ML   │
              │  200 trees          │
              │  8 000 synthetic    │
              │  training samples   │
              │  11 strategy classes│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   DATA SOURCES      │
              │  yfinance (live)    │
              │  Options chain      │
              │  Yahoo News         │
              └─────────────────────┘
```

---

## 🧠 ML Model Details

| Item | Detail |
|------|--------|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Trees | 200, max_depth=12 |
| Training data | 8 000 synthetic samples |
| Label method | Expert rule system → synthetic labels → ML learns smooth boundaries |
| Features | IV Rank, HV-30, HV-10, HV/IV Ratio, Trend Score, RSI, Mom-20D, Mom-5D, Earnings Proximity, Sentiment |
| Output | Top-3 strategy recommendations with confidence % |

---

## 📊 Strategies Covered (11)

| Strategy | Complexity | Market View |
|----------|-----------|-------------|
| Long Call | Beginner | Strongly Bullish |
| Long Put | Beginner | Strongly Bearish |
| Covered Call | Beginner | Neutral–Bullish |
| Cash-Secured Put | Beginner | Neutral–Bullish |
| Bull Call Spread | Intermediate | Moderately Bullish |
| Bear Put Spread | Intermediate | Moderately Bearish |
| Straddle | Intermediate | Neutral / big move |
| Strangle | Intermediate | Neutral / huge move |
| Iron Condor | Advanced | Neutral / low vol |
| Butterfly Spread | Advanced | Neutral / pin |
| Calendar Spread | Advanced | Neutral short-term |

---

## 🔮 Potential Upgrades

- **Real IV data** via CBOE API or Tradier
- **FinBERT** for proper NLP sentiment scoring
- **SHAP explainability** plots for advanced users
- **Backtesting engine** to validate strategy P&L
- **Paper trading** integration via Alpaca or TD Ameritrade
- **Greeks calculator** (Delta, Gamma, Theta, Vega, Rho)
- **Multi-leg trade builder** with breakeven calculator

---

## ⚠️ Disclaimer
This application is for **educational purposes only**. It does not constitute financial advice. Options trading involves substantial risk of loss. Always consult a licensed financial advisor before trading.
