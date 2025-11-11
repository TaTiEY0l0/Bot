"""
Aster Scalper Bot
=================

Trading bot de scalping pour le DEX Aster (perpétuels ASTERUSDT) avec levier x50.

Installation rapide (Windows, Python >= 3.10)
--------------------------------------------
Ouvrez PowerShell ou l'invite de commandes et exécutez :

    pip install ta pandas numpy requests websocket-client websockets rich pydantic python-dotenv

Exécution (paper trading par défaut) :

    python aster_scalper.py --paper

Passage en réel (confirmation interactive) :

    python aster_scalper.py --live --auto-cancel-open

Mode replay hors-ligne :

    python aster_scalper.py --replay data/ticks.csv

Exemple de fichier .env (à placer à côté du script)
---------------------------------------------------
# --- ASTER API (placeholders à remplacer) ---
ASTER_API_KEY=VOTRE_CLE_API
ASTER_API_SECRET=VOTRE_SECRET_API
ASTER_BASE_URL=https://api.asterdex.example
ASTER_WS_URL=wss://ws.asterdex.example/stream

# --- PARAMETRES OPTIONNELS ---
ASTER_ACCOUNT_MODE=perp
ASTER_PAPER_INITIAL_BALANCE=13.0

Tableau des endpoints Aster attendus (adapter avec l'API réelle)
----------------------------------------------------------------
REST :
    - POST /perp/leverage/set                -> set_leverage(symbol, leverage)
    - GET  /perp/account/balance             -> get_balance()
    - GET  /perp/account/position            -> get_position(symbol)
    - GET  /perp/market/ticker               -> get_ticker(symbol)
    - GET  /perp/market/depth                -> get_orderbook(symbol)
    - POST /perp/order/place                 -> place_limit(...)
    - POST /perp/order/cancel                -> cancel_order(id)
    - POST /perp/order/cancel_all            -> cancel_all(symbol)

WebSocket :
    - trades stream    -> listen_trades()
    - ticker stream    -> listen_ticker()
    - order stream     -> (optionnel, pour mises à jour temps réel)

Toutes les implémentations réseau sont regroupées dans AsterClient. Il suffit
de brancher les URLs, paramètres et signatures sur les méthodes indiquées pour
rendre le bot opérationnel en réel.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import datetime as dt
import functools
import hashlib
import hmac
import json
import logging
import logging.handlers
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseSettings, Field, PositiveFloat, validator
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from ta.volume import MFIIndicator
from websockets.exceptions import ConnectionClosed

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = lambda *_, **__: None  # type: ignore

# ---------------------------------------------------------------------------
# Chargement des variables d'environnement
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Journalisation
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "aster_scalper.log"

logger = logging.getLogger("aster_scalper")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
if not logger.handlers:
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(_formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# ---------------------------------------------------------------------------
# Configurations Pydantic
# ---------------------------------------------------------------------------


class BotConfig(BaseSettings):
    symbol: str = Field("ASTERUSDT-PERP", description="Symbole perp ASTER")
    context_symbol: str = Field("BTCUSDT-PERP", description="Symbole de contexte")
    leverage: int = Field(50, ge=1, le=100)
    paper: bool = True
    auto_cancel_open: bool = False
    base_url: str = Field(os.getenv("ASTER_BASE_URL", "https://api.asterdex.example"))
    ws_url: str = Field(os.getenv("ASTER_WS_URL", "wss://ws.asterdex.example/stream"))
    api_key: str = Field(os.getenv("ASTER_API_KEY", ""))
    api_secret: str = Field(os.getenv("ASTER_API_SECRET", ""))
    account_mode: str = Field(os.getenv("ASTER_ACCOUNT_MODE", "perp"))
    timezone: str = Field(time.tzname[0])
    tp_bp: PositiveFloat = Field(0.12, description="Take profit en % (basis points * 0.01)")
    sl_bp: PositiveFloat = Field(0.28, description="Stop loss en %")
    tp_atr_mult: PositiveFloat = Field(1.2, description="Multiplicateur ATR pour TP")
    sl_atr_mult: PositiveFloat = Field(1.8, description="Multiplicateur ATR pour SL")
    atr_period: int = 14
    rsi_fast_period: int = 3
    rsi_period: int = 7
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    bb_period: int = 20
    bb_dev: float = 2.0
    trade_notional_pct: float = 0.01
    trade_notional_cap: float = 5.0
    taker_fee: float = 0.0007
    maker_fee: float = -0.00002
    min_notional: float = 1.0
    initial_balance: float = Field(float(os.getenv("ASTER_PAPER_INITIAL_BALANCE", "13")))
    max_trades_hour: int = 40
    max_daily_loss: float = 0.10
    btc_pause_seconds: int = 90
    replay_path: Optional[str] = None
    heartbeat_interval: int = 20
    order_expiry_seconds: int = 25
    price_precision: int = 4
    quantity_precision: int = 3

    class Config:
        env_prefix = "ASTER_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("trade_notional_pct")
    def _pct_between_zero_one(cls, v: float) -> float:
        if not 0 < v <= 0.5:
            raise ValueError("trade_notional_pct doit être entre 0 et 0.5")
        return v


config = BotConfig()

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def format_ts(ts: dt.datetime) -> str:
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def hmac_sha256(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Aster Client (stubs + placeholders)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple rate limiter (token bucket)."""

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()

    async def wait(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= 1:
                self.tokens -= 1
                return
            await asyncio.sleep(0.05)


class AsterClient:
    """Client REST/WS simplifié. Toutes les intégrations API sont centralisées ici."""

    def __init__(self, cfg: BotConfig, session: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.session = session
        self.rate_limiter = RateLimiter(rate=5, capacity=10)
        self._paper_balance = cfg.initial_balance
        self._open_orders: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers HTTP (stubs) à adapter avec la vraie API REST Aster
    # ------------------------------------------------------------------
    async def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        """Effectue une requête REST. Remplacer par requests/httpx en live."""
        await self.rate_limiter.wait()
        # TODO: Insérer ici la logique réelle de signature/headers
        logger.debug("API %s %s params=%s", method, path, params)
        # En paper, renvoyer des données simulées
        if self.cfg.paper:
            if path.endswith("ticker"):
                price = 1.0 + random.uniform(-0.01, 0.01)
                return {
                    "symbol": params.get("symbol"),
                    "last": price,
                    "bid": price - 0.0005,
                    "ask": price + 0.0005,
                    "timestamp": now_utc().isoformat(),
                }
            if path.endswith("balance"):
                return {"balance": self._paper_balance, "timestamp": now_utc().isoformat()}
            if path.endswith("position"):
                return {"symbol": params.get("symbol"), "size": 0, "entryPrice": None}
            if path.endswith("depth"):
                return {
                    "bids": [[0.999, 100], [0.9985, 80]],
                    "asks": [[1.001, 100], [1.0015, 70]],
                    "timestamp": now_utc().isoformat(),
                }
            if path.endswith("order/place"):
                order_id = f"paper-{int(time.time()*1000)}"
                order = {
                    "id": order_id,
                    "symbol": params["symbol"],
                    "price": params["price"],
                    "size": params["size"],
                    "side": params["side"],
                    "status": "open",
                }
                self._open_orders[order_id] = order
                return order
            if path.endswith("order/cancel"):
                order_id = params["id"]
                self._open_orders.pop(order_id, None)
                return {"id": order_id, "status": "cancelled"}
            if path.endswith("order/cancel_all"):
                symbol = params.get("symbol")
                to_cancel = [oid for oid, ord in self._open_orders.items() if ord["symbol"] == symbol]
                for oid in to_cancel:
                    self._open_orders.pop(oid, None)
                return {"cancelled": len(to_cancel)}
        # Par défaut, renvoyer un stub
        return {"status": "ok", "note": "Stub response"}

    # ------------------------------------------------------------------
    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._request("POST", "/perp/leverage/set", {"symbol": symbol, "leverage": leverage})

    async def get_balance(self) -> dict:
        return await self._request("GET", "/perp/account/balance", {})

    async def get_position(self, symbol: str) -> dict:
        return await self._request("GET", "/perp/account/position", {"symbol": symbol})

    async def get_ticker(self, symbol: str) -> dict:
        return await self._request("GET", "/perp/market/ticker", {"symbol": symbol})

    async def get_orderbook(self, symbol: str) -> dict:
        return await self._request("GET", "/perp/market/depth", {"symbol": symbol})

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        post_only: bool = True,
        reduce_only: bool = False,
    ) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "price": round(price, self.cfg.price_precision),
            "size": round(size, self.cfg.quantity_precision),
            "type": "limit",
            "postOnly": post_only,
            "reduceOnly": reduce_only,
        }
        return await self._request("POST", "/perp/order/place", params)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._request("POST", "/perp/order/cancel", {"id": order_id})

    async def cancel_all(self, symbol: str) -> dict:
        return await self._request("POST", "/perp/order/cancel_all", {"symbol": symbol})

    async def listen_trades(self, symbol: str) -> Iterable[dict]:
        """Stub de flux trades. Remplacer par websocket réel."""
        while True:
            await asyncio.sleep(1)
            price = 1 + random.uniform(-0.01, 0.01)
            yield {
                "symbol": symbol,
                "price": price,
                "size": random.uniform(0.1, 0.5),
                "side": random.choice(["buy", "sell"]),
                "timestamp": now_utc().isoformat(),
            }

    async def listen_ticker(self, symbol: str) -> Iterable[dict]:
        while True:
            await asyncio.sleep(1)
            price = 1 + random.uniform(-0.01, 0.01)
            yield {
                "symbol": symbol,
                "last": price,
                "bid": price - 0.0005,
                "ask": price + 0.0005,
                "timestamp": now_utc().isoformat(),
            }


# ---------------------------------------------------------------------------
# Paper Broker (simulation)
# ---------------------------------------------------------------------------


@dataclass
class PaperOrder:
    id: str
    symbol: str
    side: str
    price: float
    size: float
    status: str = "open"
    created_at: dt.datetime = field(default_factory=now_utc)


@dataclass
class PaperTrade:
    ts: dt.datetime
    order_id: str
    side: str
    price: float
    size: float
    fees: float
    pnl: float
    pnl_pct: float
    reason: str
    regime: str


class PaperBroker:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.balance = cfg.initial_balance
        self.equity = cfg.initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        self.side: Optional[str] = None
        self.orders: Dict[str, PaperOrder] = {}
        self.trades: List[PaperTrade] = []
        self.csv_file = Path("paper_trades.csv")
        self.equity_curve_file = Path("equity_curve.csv")
        self._init_csv()

    def _init_csv(self) -> None:
        if not self.csv_file.exists():
            with self.csv_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "order_id", "side", "price", "size", "fees", "pnl", "pnl_pct", "reason", "regime"])
        if not self.equity_curve_file.exists():
            with self.equity_curve_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "equity"])

    def log_equity(self) -> None:
        with self.equity_curve_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([now_utc().isoformat(), self.equity])

    def place_limit(self, side: str, price: float, size: float, regime: str) -> PaperOrder:
        order_id = f"paper-{len(self.orders)+1}-{int(time.time()*1000)}"
        order = PaperOrder(order_id, self.cfg.symbol, side, price, size)
        self.orders[order_id] = order
        logger.info("[PAPER] New limit %s %s @ %.4f size %.4f", side, self.cfg.symbol, price, size)
        return order

    def cancel_order(self, order_id: str) -> None:
        if order_id in self.orders:
            self.orders[order_id].status = "cancelled"
            logger.info("[PAPER] Cancel order %s", order_id)

    def cancel_all(self) -> None:
        for order in self.orders.values():
            order.status = "cancelled"
        logger.info("[PAPER] Cancel all orders")

    def _execute_order(self, order: PaperOrder, price: float, regime: str, reason: str) -> Optional[PaperTrade]:
        if order.status != "open":
            return None
        # Simple slippage model
        slippage = random.uniform(0, 0.0002)
        fill_price = price + (slippage if order.side == "buy" else -slippage)
        fees = abs(fill_price * order.size) * self.cfg.maker_fee
        pnl = 0.0
        pnl_pct = 0.0
        if self.position_size == 0:
            self.side = order.side
            self.position_size = order.size if order.side == "buy" else -order.size
            self.entry_price = fill_price
        else:
            # Closing position
            if (self.position_size > 0 and order.side == "sell") or (self.position_size < 0 and order.side == "buy"):
                pnl = (fill_price - self.entry_price) * self.position_size * self.cfg.leverage
                if self.position_size < 0:
                    pnl = -pnl
                pnl_pct = pnl / self.equity if self.equity else 0.0
                self.balance += pnl + fees
                self.position_size = 0
                self.entry_price = 0
                self.side = None
                self.equity = self.balance
            else:
                logger.warning("[PAPER] Position conflict, ignoring fill")
                return None
        order.status = "filled"
        trade = PaperTrade(now_utc(), order.id, order.side, fill_price, order.size, fees, pnl, pnl_pct, reason, regime)
        self.trades.append(trade)
        with self.csv_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.ts.isoformat(),
                trade.order_id,
                trade.side,
                f"{trade.price:.6f}",
                f"{trade.size:.6f}",
                f"{trade.fees:.6f}",
                f"{trade.pnl:.6f}",
                f"{trade.pnl_pct:.6f}",
                trade.reason,
                trade.regime,
            ])
        self.log_equity()
        logger.info(
            "[PAPER] Filled %s size %.4f @ %.4f (fees %.6f, pnl %.4f)",
            order.side,
            order.size,
            fill_price,
            fees,
            pnl,
        )
        return trade

    def maybe_fill_orders(self, mid_price: float, regime: str, reason: str) -> None:
        for order in list(self.orders.values()):
            if order.status != "open":
                continue
            if order.side == "buy" and order.price >= mid_price:
                self._execute_order(order, order.price, regime, reason)
            elif order.side == "sell" and order.price <= mid_price:
                self._execute_order(order, order.price, regime, reason)


# ---------------------------------------------------------------------------
# Market Data Engine
# ---------------------------------------------------------------------------


class MarketDataEngine:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.prices: List[Tuple[dt.datetime, float]] = []
        self.bars: Dict[str, pd.DataFrame] = {
            "1s": pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            "5s": pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            "15s": pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            "1m": pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
        }

    def _append_bar(self, timeframe: str, ts: dt.datetime, price: float, volume: float) -> None:
        df = self.bars[timeframe]
        period_seconds = {"1s": 1, "5s": 5, "15s": 15, "1m": 60}[timeframe]
        bucket = (ts.timestamp() // period_seconds) * period_seconds
        bucket_ts = dt.datetime.fromtimestamp(bucket, dt.timezone.utc)
        if bucket_ts in df.index:
            bar = df.loc[bucket_ts]
            bar_high = max(bar["high"], price)
            bar_low = min(bar["low"], price)
            bar_volume = bar["volume"] + volume
            df.loc[bucket_ts, ["high", "low", "close", "volume"]] = [bar_high, bar_low, price, bar_volume]
        else:
            df.loc[bucket_ts] = [price, price, price, price, volume]
        self.bars[timeframe] = df.sort_index()

    def add_tick(self, ts: dt.datetime, price: float, volume: float) -> None:
        self.prices.append((ts, price))
        self._append_bar("1s", ts, price, volume)
        if len(self.bars["1s"]) >= 5:
            self._rollup("1s", "5s", 5)
        if len(self.bars["5s"]) >= 3:
            self._rollup("5s", "15s", 3)
        if len(self.bars["15s"]) >= 4:
            self._rollup("15s", "1m", 4)

    def _rollup(self, src_tf: str, dst_tf: str, factor: int) -> None:
        src_df = self.bars[src_tf]
        dst_df = self.bars[dst_tf]
        grouped = src_df.groupby(np.arange(len(src_df)) // factor)
        for _, group in grouped:
            start_ts = group.index[0]
            end_ts = group.index[-1]
            bucket_ts = end_ts
            if bucket_ts in dst_df.index:
                continue
            dst_df.loc[bucket_ts] = [
                group.iloc[0]["open"],
                group["high"].max(),
                group["low"].min(),
                group.iloc[-1]["close"],
                group["volume"].sum(),
            ]
        self.bars[dst_tf] = dst_df.sort_index()

    def get_latest_price(self) -> Optional[float]:
        return self.prices[-1][1] if self.prices else None

    def compute_indicators(self) -> Dict[str, Any]:
        indicators: Dict[str, Any] = {}
        df = self.bars["15s"].copy()
        if len(df) >= self.cfg.bb_period:
            bb = BollingerBands(df["close"], window=self.cfg.bb_period, window_dev=self.cfg.bb_dev)
            indicators["bb_mid"] = bb.bollinger_mavg().iloc[-1]
            indicators["bb_high"] = bb.bollinger_hband().iloc[-1]
            indicators["bb_low"] = bb.bollinger_lband().iloc[-1]
            indicators["bb_width"] = (indicators["bb_high"] - indicators["bb_low"]) / indicators["bb_mid"]
        if len(df) >= self.cfg.atr_period:
            tr = df["high"].combine(df["low"], max) - df["low"].combine(df["high"], min)
            atr = tr.rolling(window=self.cfg.atr_period).mean()
            indicators["atr15"] = atr.iloc[-1]
        if len(df) >= self.cfg.ema_slow_period:
            ema_fast = EMAIndicator(df["close"], self.cfg.ema_fast_period).ema_indicator()
            ema_slow = EMAIndicator(df["close"], self.cfg.ema_slow_period).ema_indicator()
            indicators["ema_fast"] = ema_fast.iloc[-1]
            indicators["ema_slow"] = ema_slow.iloc[-1]
            indicators["ema_fast_slope"] = ema_fast.diff().iloc[-1]
            indicators["ema_slow_slope"] = ema_slow.diff().iloc[-1]
        if len(df) >= self.cfg.rsi_period:
            rsi = RSIIndicator(df["close"], self.cfg.rsi_period).rsi()
            indicators["rsi7"] = rsi.iloc[-1]
        if len(df) >= self.cfg.rsi_fast_period:
            rsi3 = RSIIndicator(df["close"], self.cfg.rsi_fast_period).rsi()
            indicators["rsi3"] = rsi3.iloc[-1]
        if len(df) >= self.cfg.bb_period:
            price = df["close"].iloc[-1]
            weights = np.arange(1, len(df) + 1)
            vwap = np.dot(weights, df["close"].values) / weights.sum()
            indicators["vwap"] = vwap
        return indicators


# ---------------------------------------------------------------------------
# Strategy Range Scalping
# ---------------------------------------------------------------------------


@dataclass
class StrategySignal:
    side: Optional[str]
    reason: str
    regime: str
    price: Optional[float] = None


class StrategyRangeScalp:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.last_signal_side: Optional[str] = None
        self.pause_until: Optional[dt.datetime] = None

    def set_pause(self, seconds: int) -> None:
        self.pause_until = now_utc() + dt.timedelta(seconds=seconds)
        logger.info("Strategy paused for %s seconds", seconds)

    def is_paused(self) -> bool:
        return self.pause_until is not None and now_utc() < self.pause_until

    def evaluate(self, indicators: Dict[str, Any], price: float) -> StrategySignal:
        if self.is_paused():
            return StrategySignal(None, "paused", "paused")
        bb_low = indicators.get("bb_low")
        bb_high = indicators.get("bb_high")
        rsi7 = indicators.get("rsi7")
        rsi3 = indicators.get("rsi3", 50)
        vwap = indicators.get("vwap")
        atr = indicators.get("atr15", 0.0005)
        bb_width = indicators.get("bb_width", 0.01)
        ema_fast_slope = indicators.get("ema_fast_slope", 0.0)
        ema_slow_slope = indicators.get("ema_slow_slope", 0.0)

        regime = "range" if abs(ema_fast_slope) < 0.0001 and abs(ema_slow_slope) < 0.0001 else "trend"
        if bb_width > 0.01:
            regime = "trend"
        if regime != "range":
            return StrategySignal(None, "not in range", regime)

        k_sigma = 0.5
        long_cond = (
            bb_low is not None
            and price <= bb_low * 1.0003
            and rsi7 is not None
            and rsi7 < 30
            and vwap is not None
            and price <= vwap - k_sigma * atr
        )
        short_cond = (
            bb_high is not None
            and price >= bb_high * 0.9997
            and rsi7 is not None
            and rsi7 > 70
            and vwap is not None
            and price >= vwap + k_sigma * atr
        )
        if long_cond and rsi3 > 15:
            self.last_signal_side = "buy"
            return StrategySignal("buy", "bb_low + rsi oversold", regime, price)
        if short_cond and rsi3 < 85:
            self.last_signal_side = "sell"
            return StrategySignal("sell", "bb_high + rsi overbought", regime, price)
        return StrategySignal(None, "no setup", regime)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    tp: Optional[float] = None
    sl: Optional[float] = None


class RiskManager:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.daily_loss_limit = cfg.max_daily_loss
        self.daily_start_equity = cfg.initial_balance
        self.trades_this_hour = 0
        self.last_hour = now_utc().hour
        self.position = PositionState()

    def reset_hour_if_needed(self) -> None:
        hour = now_utc().hour
        if hour != self.last_hour:
            self.trades_this_hour = 0
            self.last_hour = hour

    def can_trade(self, equity: float) -> bool:
        self.reset_hour_if_needed()
        if equity <= self.daily_start_equity * (1 - self.daily_loss_limit):
            logger.warning("Daily loss limit reached: %.2f <= %.2f", equity, self.daily_start_equity * (1 - self.daily_loss_limit))
            return False
        if self.trades_this_hour >= self.cfg.max_trades_hour:
            logger.warning("Max trades/hour reached (%d)", self.cfg.max_trades_hour)
            return False
        if self.position.size != 0:
            logger.debug("Already in position, no new trade")
            return False
        return True

    def update_position(self, side: str, size: float, entry_price: float, atr: float) -> None:
        self.position.side = side
        self.position.size = size if side == "buy" else -size
        self.position.entry_price = entry_price
        tp_offset = min(self.cfg.tp_bp / 100, self.cfg.tp_atr_mult * atr)
        sl_offset = max(self.cfg.sl_bp / 100, self.cfg.sl_atr_mult * atr)
        if side == "buy":
            self.position.tp = entry_price * (1 + tp_offset)
            self.position.sl = entry_price * (1 - sl_offset)
        else:
            self.position.tp = entry_price * (1 - tp_offset)
            self.position.sl = entry_price * (1 + sl_offset)
        self.trades_this_hour += 1
        logger.info("Position updated: %s size %.4f entry %.5f TP %.5f SL %.5f", side, size, entry_price, self.position.tp, self.position.sl)

    def clear_position(self) -> None:
        self.position = PositionState()

    def check_exit(self, price: float) -> Optional[str]:
        if self.position.size == 0:
            return None
        if self.position.side == "buy":
            if price >= (self.position.tp or float("inf")):
                return "tp"
            if price <= (self.position.sl or 0):
                return "sl"
        else:
            if price <= (self.position.tp or 0):
                return "tp"
            if price >= (self.position.sl or float("inf")):
                return "sl"
        return None


# ---------------------------------------------------------------------------
# Order Executor
# ---------------------------------------------------------------------------


class OrderExecutor:
    def __init__(self, cfg: BotConfig, client: AsterClient, broker: Optional[PaperBroker], risk: RiskManager) -> None:
        self.cfg = cfg
        self.client = client
        self.broker = broker
        self.risk = risk
        self.open_orders: Dict[str, dict] = {}

    async def place_order(self, side: str, price: float, size: float, regime: str) -> Optional[str]:
        if self.cfg.paper:
            order = self.broker.place_limit(side, price, size, regime)
            self.open_orders[order.id] = {"side": side, "price": price, "size": size, "created": now_utc()}
            self.risk.update_position(side, size, price, atr=0.0005)
            return order.id
        else:
            order = await self.client.place_limit(self.cfg.symbol, side, price, size, post_only=True)
            order_id = order.get("id")
            self.open_orders[order_id] = {"side": side, "price": price, "size": size, "created": now_utc()}
            self.risk.update_position(side, size, price, atr=0.0005)
            return order_id

    async def cancel_stale_orders(self) -> None:
        to_cancel = [oid for oid, meta in self.open_orders.items() if (now_utc() - meta["created"]).total_seconds() > self.cfg.order_expiry_seconds]
        for oid in to_cancel:
            if self.cfg.paper:
                self.broker.cancel_order(oid)
            else:
                await self.client.cancel_order(oid)
            self.open_orders.pop(oid, None)

    async def cancel_all(self) -> None:
        if self.cfg.paper:
            self.broker.cancel_all()
        else:
            await self.client.cancel_all(self.cfg.symbol)
        self.open_orders.clear()


# ---------------------------------------------------------------------------
# BTC Context Monitor (simple placeholder)
# ---------------------------------------------------------------------------


class BTCContext:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.state = "calme"
        self.pause_until: Optional[dt.datetime] = None

    def update(self, price: float) -> None:
        # Placeholder simple random logic
        if random.random() < 0.01:
            self.state = "alerte pump"
            self.pause_until = now_utc() + dt.timedelta(seconds=self.cfg.btc_pause_seconds)
        elif random.random() < 0.01:
            self.state = "alerte dump"
            self.pause_until = now_utc() + dt.timedelta(seconds=self.cfg.btc_pause_seconds)
        else:
            if self.pause_until and now_utc() < self.pause_until:
                return
            self.state = "calme"
            self.pause_until = None

    def is_paused(self) -> bool:
        return self.pause_until is not None and now_utc() < self.pause_until

    def remaining(self) -> int:
        if not self.pause_until:
            return 0
        return max(0, int((self.pause_until - now_utc()).total_seconds()))


# ---------------------------------------------------------------------------
# TUI (Rich)
# ---------------------------------------------------------------------------


class TradingUI:
    def __init__(self, cfg: BotConfig, broker: Optional[PaperBroker], risk: RiskManager, btc_ctx: BTCContext) -> None:
        self.cfg = cfg
        self.broker = broker
        self.risk = risk
        self.console = Console()
        self.btc_ctx = btc_ctx
        self.status_text = "Running"

    def build_table(self, last_price: float, indicators: Dict[str, Any], signal: StrategySignal) -> Table:
        table = Table(title="Aster Scalper", box=box.SIMPLE_HEAVY)
        table.add_column("Metric")
        table.add_column("Value")
        balance = self.broker.balance if self.broker else 0.0
        equity = self.broker.equity if self.broker else 0.0
        table.add_row("Mode", "PAPER" if self.cfg.paper else "LIVE")
        table.add_row("Balance", f"{balance:.4f}")
        table.add_row("Equity", f"{equity:.4f}")
        table.add_row("Position", f"{self.risk.position.side or 'flat'} {self.risk.position.size:.4f}")
        table.add_row("Entry", f"{self.risk.position.entry_price:.5f}")
        table.add_row("TP/SL", f"{self.risk.position.tp} / {self.risk.position.sl}")
        table.add_row("Last Price", f"{last_price:.5f}" if last_price else "-")
        table.add_row("BB Width", f"{indicators.get('bb_width', float('nan')):.6f}")
        table.add_row("RSI7 / RSI3", f"{indicators.get('rsi7', float('nan')):.2f} / {indicators.get('rsi3', float('nan')):.2f}")
        table.add_row("VWAP", f"{indicators.get('vwap', float('nan')):.5f}")
        table.add_row("ATR15", f"{indicators.get('atr15', float('nan')):.5f}")
        table.add_row("Regime", signal.regime)
        table.add_row("Signal", signal.reason)
        table.add_row("BTC State", f"{self.btc_ctx.state} ({self.btc_ctx.remaining()}s)")
        table.add_row("Status", self.status_text)
        return table


# ---------------------------------------------------------------------------
# Replay Loader
# ---------------------------------------------------------------------------


def load_replay_ticks(path: str) -> List[Tuple[dt.datetime, float, float]]:
    ticks: List[Tuple[dt.datetime, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = dt.datetime.fromisoformat(row.get("ts"))
            price = float(row.get("price"))
            volume = float(row.get("volume", 0.1))
            ticks.append((ts, price, volume))
    return ticks


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------


async def heartbeat(client: AsterClient) -> None:
    while True:
        try:
            await client.get_balance()
            logger.debug("Heartbeat ok")
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("Heartbeat failed: %s", exc)
        await asyncio.sleep(config.heartbeat_interval)


async def run_bot(cfg: BotConfig) -> None:
    client = AsterClient(cfg)
    broker = PaperBroker(cfg) if cfg.paper else None
    risk = RiskManager(cfg)
    market = MarketDataEngine(cfg)
    strategy = StrategyRangeScalp(cfg)
    executor = OrderExecutor(cfg, client, broker, risk)
    btc_ctx = BTCContext(cfg)
    ui = TradingUI(cfg, broker, risk, btc_ctx)

    console = Console()
    console.print("Aster Scalper démarre...", style="bold green")

    if not cfg.paper and (not cfg.api_key or not cfg.api_secret):
        console.print("API key/secret manquants. Abandon.", style="bold red")
        return

    # Appliquer levier
    await client.set_leverage(cfg.symbol, cfg.leverage)

    # Replay mode
    ticks_iter: Iterable[Tuple[dt.datetime, float, float]]
    if cfg.replay_path:
        console.print(f"Mode replay depuis {cfg.replay_path}")
        ticks_iter = load_replay_ticks(cfg.replay_path)
    else:
        ticks_iter = None

    stop_event = asyncio.Event()

    def handle_sigint():
        logger.info("SIGINT reçu, arrêt en cours...")
        ui.status_text = "Stopping"
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_sigint)
        except NotImplementedError:  # Windows fallback
            signal.signal(sig, lambda *_: handle_sigint())

    async def data_feed():
        if ticks_iter is not None:
            for ts, price, volume in ticks_iter:
                market.add_tick(ts, price, volume)
                await asyncio.sleep(0.01)
        else:
            async for tick in client.listen_ticker(cfg.symbol):
                ts = dt.datetime.fromisoformat(tick["timestamp"])
                price = float(tick["last"])
                volume = random.uniform(0.1, 1.0)
                market.add_tick(ts, price, volume)
                if broker:
                    broker.maybe_fill_orders(price, "range", "mid touch")
                if stop_event.is_set():
                    break

    async def strategy_loop():
        with Live(refresh_per_second=2) as live:
            while not stop_event.is_set():
                await asyncio.sleep(1)
                last_price = market.get_latest_price()
                if last_price is None:
                    continue
                indicators = market.compute_indicators()
                btc_ctx.update(last_price * 0.03 + 20000)  # fake context price
                if btc_ctx.is_paused():
                    strategy.set_pause(btc_ctx.remaining())
                signal = strategy.evaluate(indicators, last_price)
                if signal.side and not btc_ctx.is_paused():
                    equity = broker.equity if broker else cfg.initial_balance
                    if risk.can_trade(equity):
                        size = max(cfg.min_notional, equity * cfg.trade_notional_pct)
                        size = min(size, cfg.trade_notional_cap)
                        await executor.place_order(signal.side, signal.price or last_price, size, signal.regime)
                exit_reason = risk.check_exit(last_price)
                if exit_reason and broker:
                    side = "sell" if risk.position.side == "buy" else "buy"
                    broker.place_limit(side, last_price, abs(risk.position.size), signal.regime)
                    risk.clear_position()
                await executor.cancel_stale_orders()
                table = ui.build_table(last_price, indicators, signal)
                live.update(table)

    await asyncio.gather(data_feed(), strategy_loop())

    if cfg.paper:
        broker.cancel_all()
    elif cfg.auto_cancel_open:
        await executor.cancel_all()

    console.print("Bot stoppé.", style="bold yellow")


# ---------------------------------------------------------------------------
# Confirmation live
# ---------------------------------------------------------------------------


def confirm_live() -> bool:
    console = Console()
    console.print("Mode LIVE demandé.", style="bold red")
    console.print("Assurez-vous d'avoir configuré vos clés API et compris les risques.")
    answer = console.input("Tapez OUI pour continuer : ")
    return answer.strip().lower() == "oui"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aster Range Scalper")
    parser.add_argument("--live", action="store_true", help="Active le trading live")
    parser.add_argument("--paper", action="store_true", help="Force le paper trading")
    parser.add_argument("--auto-cancel-open", action="store_true", help="Annuler les ordres ouverts à l'arrêt")
    parser.add_argument("--symbol", type=str, default=config.symbol, help="Symbole principal")
    parser.add_argument("--leverage", type=int, default=config.leverage, help="Levier")
    parser.add_argument("--tp_bp", type=float, default=config.tp_bp, help="Take profit %")
    parser.add_argument("--sl_bp", type=float, default=config.sl_bp, help="Stop loss %")
    parser.add_argument("--max-trades-hour", type=int, default=config.max_trades_hour, help="Max trades par heure")
    parser.add_argument("--replay", type=str, default=None, help="Chemin CSV ticks pour replay")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> BotConfig:
    cfg = BotConfig()
    cfg.paper = not args.live if not args.paper else True
    cfg.auto_cancel_open = args.auto_cancel_open
    cfg.symbol = args.symbol
    cfg.leverage = args.leverage
    cfg.tp_bp = args.tp_bp
    cfg.sl_bp = args.sl_bp
    cfg.max_trades_hour = args.max_trades_hour
    cfg.replay_path = args.replay
    if args.live:
        cfg.paper = False
    return cfg


async def async_main() -> None:
    args = parse_args()
    cfg = build_config(args)
    if not cfg.paper and not confirm_live():
        Console().print("Confirmation manquante, arrêt.", style="bold red")
        return
    await run_bot(cfg)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Arrêt demandé par utilisateur")
