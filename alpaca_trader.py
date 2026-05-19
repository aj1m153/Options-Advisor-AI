"""
utils/alpaca_trader.py
───────────────────────
Alpaca paper-trading integration.

Get free paper-trading API keys at https://alpaca.markets
(Account → Paper Trading → API Keys)

Alpaca's paper environment supports:
  • Equities (full)
  • Options (via the options trading endpoint — enabled on most accounts)

This module exposes helpers used by the Streamlit UI tab.
"""

from __future__ import annotations
from typing import Optional, Dict, List

ALPACA_AVAILABLE = False
try:
    from alpaca.trading.client  import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    pass


# ── Client factory ────────────────────────────────────────────────────────────
def get_client(api_key: str, secret_key: str) -> Optional[object]:
    if not ALPACA_AVAILABLE or not api_key or not secret_key:
        return None
    try:
        return TradingClient(api_key, secret_key, paper=True)
    except Exception:
        return None


# ── Account summary ───────────────────────────────────────────────────────────
def get_account(client) -> Dict:
    if client is None:
        return {"error": "Not connected."}
    try:
        a = client.get_account()
        pv     = float(a.portfolio_value)
        equity = float(a.equity)
        cash   = float(a.cash)
        start  = 100_000.0          # paper accounts start at $100k
        pnl    = equity - start
        return {
            "portfolio_value": round(pv,     2),
            "equity":          round(equity, 2),
            "cash":            round(cash,   2),
            "buying_power":    round(float(a.buying_power), 2),
            "pnl_dollar":      round(pnl,    2),
            "pnl_pct":         round(pnl / start * 100, 2),
            "status":          str(a.status),
            "day_trade_count": getattr(a, "daytrade_count", "—"),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Positions ─────────────────────────────────────────────────────────────────
def get_positions(client) -> List[Dict]:
    if client is None:
        return []
    try:
        positions = client.get_all_positions()
        return [
            {
                "symbol":      p.symbol,
                "qty":         float(p.qty),
                "side":        str(p.side).replace("PositionSide.", ""),
                "avg_entry":   round(float(p.avg_entry_price), 2),
                "current":     round(float(p.current_price or 0), 2),
                "market_val":  round(float(p.market_value or 0), 2),
                "pnl":         round(float(p.unrealized_pl or 0), 2),
                "pnl_pct":     round(float(p.unrealized_plpc or 0) * 100, 2),
            }
            for p in positions
        ]
    except Exception:
        return []


# ── Orders ────────────────────────────────────────────────────────────────────
def get_orders(client, limit: int = 20) -> List[Dict]:
    if client is None:
        return []
    try:
        req    = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        orders = client.get_orders(filter=req)
        return [
            {
                "id":          str(o.id)[:8] + "…",
                "symbol":      o.symbol,
                "side":        str(o.side).replace("OrderSide.", ""),
                "qty":         float(o.qty or 0),
                "type":        str(o.type).replace("OrderType.", ""),
                "status":      str(o.status).replace("OrderStatus.", ""),
                "submitted":   str(o.submitted_at)[:19] if o.submitted_at else "—",
                "filled_avg":  round(float(o.filled_avg_price or 0), 2),
            }
            for o in orders
        ]
    except Exception:
        return []


# ── Place orders ──────────────────────────────────────────────────────────────
def place_market_order(
    client,
    symbol: str,
    qty:    float,
    side:   str,        # "buy" | "sell"
) -> Dict:
    """Submit a market order (equity).  Returns result dict."""
    if not ALPACA_AVAILABLE:
        return {"error": "alpaca-py not installed."}
    if client is None:
        return {"error": "Client not connected. Check your API keys."}
    try:
        req = MarketOrderRequest(
            symbol        = symbol.upper(),
            qty           = qty,
            side          = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return {
            "success":  True,
            "order_id": str(order.id)[:8] + "…",
            "symbol":   order.symbol,
            "side":     str(order.side),
            "qty":      float(order.qty),
            "status":   str(order.status),
        }
    except Exception as e:
        return {"error": str(e)}


def place_limit_order(
    client,
    symbol:      str,
    qty:         float,
    side:        str,
    limit_price: float,
) -> Dict:
    """Submit a limit order (equity)."""
    if not ALPACA_AVAILABLE:
        return {"error": "alpaca-py not installed."}
    if client is None:
        return {"error": "Client not connected."}
    try:
        req = LimitOrderRequest(
            symbol        = symbol.upper(),
            qty           = qty,
            side          = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price   = round(limit_price, 2),
            time_in_force = TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return {
            "success":     True,
            "order_id":    str(order.id)[:8] + "…",
            "symbol":      order.symbol,
            "limit_price": float(order.limit_price or 0),
            "status":      str(order.status),
        }
    except Exception as e:
        return {"error": str(e)}


def cancel_all_orders(client) -> Dict:
    if client is None:
        return {"error": "Not connected."}
    try:
        client.cancel_orders()
        return {"success": True, "message": "All open orders cancelled."}
    except Exception as e:
        return {"error": str(e)}


def close_position(client, symbol: str) -> Dict:
    if client is None:
        return {"error": "Not connected."}
    try:
        client.close_position(symbol.upper())
        return {"success": True, "message": f"Position in {symbol} closed."}
    except Exception as e:
        return {"error": str(e)}
