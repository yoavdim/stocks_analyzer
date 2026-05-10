#!/usr/bin/env python3
"""
Drop-in adapter for TASE mutual funds that mimics the YfTickerUSD interface.

Reads price history from the .tase_cache/ directory (populated by tase_fund_fetcher.py).
Prices in the cache are in ILA (agorot); FX conversion to base currency is applied
transparently, matching YfTickerUSD behavior.
"""

import sys
import re
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from fx_converter import fx_converter

_CACHE_DIR = Path(__file__).parent / ".tase_cache"


def is_tase_fund(symbol: str, market: str) -> bool:
    """True if symbol is a numeric TASE fund ID (not a Yahoo ticker)."""
    return market.upper() == "TLV" and symbol.isdigit()


def _parse_period(period: str) -> timedelta:
    """Convert yfinance period strings like '10y', '5d', '1mo' to timedelta."""
    m = re.match(r"(\d+)(d|mo|y)", period)
    if not m:
        return timedelta(days=365 * 10)
    n, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        return timedelta(days=n)
    elif unit == "mo":
        return timedelta(days=n * 30)
    else:
        return timedelta(days=n * 365)


def _load_fund_cache(fund_id: str) -> dict:
    """Load cache, raising if missing or empty."""
    import json
    cache_file = _CACHE_DIR / f"{fund_id}.json"
    if not cache_file.exists():
        raise ValueError(
            f"No TASE cache for fund {fund_id}. "
            f"Run: python tase_fund_fetcher.py {fund_id}"
        )
    data = json.loads(cache_file.read_text())
    if not data.get("history"):
        raise ValueError(f"TASE cache for fund {fund_id} has no history data.")
    return data


def _check_staleness(cache: dict, fund_id: str):
    """Warn if cache is from a previous day."""
    fetched_at = datetime.fromisoformat(cache["fetched_at"])
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if fetched_at < today_start:
        print(
            f"TASE WARNING: cache for fund {fund_id} is stale "
            f"(fetched {fetched_at.date()}). Run: python tase_fund_fetcher.py {fund_id}",
            file=sys.stderr,
        )


class TaseTickerUSD:
    """Drop-in replacement for YfTickerUSD that reads from TASE fund cache."""

    def __init__(self, fund_id: str):
        self.fund_id = fund_id
        self._cache = None
        self._price_currency = "ILA"
        self._financial_currency = "ILA"

    def _ensure_cache(self):
        if self._cache is None:
            self._cache = _load_fund_cache(self.fund_id)
            _check_staleness(self._cache, self.fund_id)
        return self._cache

    def _detect_currencies(self):
        pass  # currencies are fixed

    @property
    def info(self) -> dict:
        cache = self._ensure_cache()
        latest_price = cache["history"][0]["price"]
        rate = fx_converter.get_rate("ILA")
        return {
            "quoteType": "MUTUALFUND",
            "currency": "ILA",
            "shortName": f"TASE Fund {self.fund_id}",
            "longName": f"TASE Mutual Fund {self.fund_id}",
            "regularMarketPrice": latest_price * rate,
            "previousClose": latest_price * rate,
        }

    def history(self, *args, **kwargs) -> pd.DataFrame:
        """Return price history converted to base currency.

        Supports kwargs: period, start, end (matching yfinance interface).
        """
        cache = self._ensure_cache()
        history = cache["history"]

        dates = [pd.Timestamp(entry["date"]) for entry in history]
        prices = [entry["price"] for entry in history]
        df = pd.DataFrame({"Close": prices}, index=pd.DatetimeIndex(dates))
        df = df.sort_index()
        df.index = df.index.normalize()
        df = df[~df.index.duplicated(keep="last")]

        # Filter by period or start/end
        start = kwargs.get("start")
        end = kwargs.get("end")
        period = kwargs.get("period")

        if start is not None:
            start = pd.Timestamp(start).normalize()
            df = df[df.index >= start]
        if end is not None:
            end = pd.Timestamp(end).normalize()
            df = df[df.index <= end]
        if period is not None and start is None and end is None:
            cutoff = pd.Timestamp.now().normalize() - _parse_period(period)
            df = df[df.index >= cutoff]

        if df.empty:
            return df

        # Convert ILA to base currency
        fx_series = fx_converter.get_rate_series("ILA", df.index)
        df["Close"] = df["Close"] * fx_series

        return df

    def __getattr__(self, name):
        # For any other attributes (balance_sheet, etc.), return empty/None
        if name in ("balance_sheet", "quarterly_balance_sheet",
                    "income_stmt", "quarterly_income_stmt",
                    "cash_flow", "quarterly_cash_flow"):
            return pd.DataFrame()
        raise AttributeError(f"TaseTickerUSD has no attribute '{name}'")


class TaseTickersUSD:
    """Mixed Yahoo + TASE wrapper that satisfies the YfTickersUSD interface."""

    def __init__(self, yf_tickers_usd, tase_tickers: dict):
        """
        Args:
            yf_tickers_usd: YfTickersUSD instance for Yahoo symbols (or None)
            tase_tickers: dict mapping full_symbol -> TaseTickerUSD
        """
        self._yahoo = yf_tickers_usd
        self._tase = tase_tickers

    @property
    def tickers(self) -> dict:
        result = {}
        if self._yahoo:
            result.update(self._yahoo.tickers)
        result.update(self._tase)
        return result

    def history(self, *args, **kwargs) -> pd.DataFrame:
        """Return bulk price history combining Yahoo and TASE data.

        Returns a DataFrame with MultiIndex columns (field, symbol) when mixed,
        or simple columns when only "Close" is accessed downstream.
        """
        yahoo_df = None
        if self._yahoo:
            yahoo_df = self._yahoo.history(*args, **kwargs)

        # Build TASE histories
        tase_frames = {}
        for full_sym, ticker in self._tase.items():
            h = ticker.history(*args, **kwargs)
            if not h.empty:
                tase_frames[full_sym] = h["Close"]

        if not tase_frames and yahoo_df is not None:
            return yahoo_df
        if not tase_frames and yahoo_df is None:
            return pd.DataFrame()

        # If Yahoo returned a MultiIndex DataFrame, merge TASE into it
        if yahoo_df is not None and isinstance(yahoo_df.columns, pd.MultiIndex):
            for sym, series in tase_frames.items():
                yahoo_df[("Close", sym)] = series
                for field in ("Open", "High", "Low"):
                    yahoo_df[(field, sym)] = series  # ETF-like: OHLC = Close
                yahoo_df[("Volume", sym)] = 0
            return yahoo_df

        # Simple case: Yahoo has single-level columns or is None
        # Build a combined DataFrame with symbol columns
        parts = []
        if yahoo_df is not None:
            parts.append(yahoo_df)

        if tase_frames:
            tase_df = pd.DataFrame(tase_frames)
            parts.append(tase_df)

        if len(parts) == 1:
            return parts[0]

        return parts[0].join(parts[1], how="outer")

    def __getattr__(self, name):
        if self._yahoo:
            return getattr(self._yahoo, name)
        raise AttributeError(f"TaseTickersUSD has no attribute '{name}'")
