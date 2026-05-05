#!/usr/bin/python3
"""
FX-converting wrappers around yfinance Ticker and Tickers objects.

All prices and monetary values are converted to the configured base currency transparently.
Historical values use the actual exchange rate on that date.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import json

# Load base currency from config
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npv_config.json")
with open(_config_path, "r") as _f:
    _BASE_CURRENCY = json.load(_f).get("portfolio", {}).get("base_currency", "USD").upper()

# Fields in .info that represent prices/monetary values and should be converted
_INFO_PRICE_FIELDS = {
    "regularMarketPrice", "previousClose", "open", "dayHigh", "dayLow",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyDayAverage",
    "twoHundredDayAverage", "regularMarketDayHigh", "regularMarketDayLow",
    "regularMarketOpen", "regularMarketPreviousClose",
    "bookValue", "priceToBook", "currentPrice",
    "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice",
    "totalCash", "totalDebt", "totalRevenue", "grossProfits",
    "ebitda", "operatingCashflow", "freeCashflow", "revenue",
    "marketCap", "enterpriseValue",
}

# Fields that are per-share and should be converted
_INFO_PER_SHARE_FIELDS = {
    "trailingEps", "forwardEps", "dividendRate", "fiveYearAvgDividendYield",
    "revenuePerShare", "totalCashPerShare",
}

# Price columns in history DataFrames
_HISTORY_PRICE_COLS = {"Open", "High", "Low", "Close"}

# Sub-unit currencies: currency code -> (parent currency, divisor)
_SUB_UNIT_CURRENCIES = {
    "ILA": ("ILS", 100.0),  # Israeli agora = ILS / 100
    "GBX": ("GBP", 100.0),  # British pence = GBP / 100
}


class FxConverter:
    """Singleton-style FX rate provider. Fetches and caches historical FX rates.
    Converts any currency to the configured base currency."""

    _TTL = 3600 * 6  # 6 hour cache for rate series

    def __init__(self):
        self._current_rates = {}  # currency -> (rate, timestamp)
        self._history_cache = {}  # currency -> pd.Series (date -> rate)
        self.base_currency = _BASE_CURRENCY

    def _resolve_sub_unit(self, currency: str):
        """Resolve sub-unit currencies (ILA, GBX) to their parent and divisor.
        Returns (lookup_currency, divisor)."""
        currency = currency.upper()
        if currency in _SUB_UNIT_CURRENCIES:
            parent, divisor = _SUB_UNIT_CURRENCIES[currency]
            return parent, divisor
        return currency, 1.0

    def get_rate(self, currency: str) -> float:
        """Get current FX rate: 1 unit of `currency` = X base_currency."""
        currency = currency.upper()
        if currency == self.base_currency:
            return 1.0

        lookup, divisor = self._resolve_sub_unit(currency)
        if divisor != 1.0:
            return self.get_rate(lookup) / divisor

        cached = self._current_rates.get(currency)
        if cached and time.time() - cached[1] < self._TTL:
            return cached[0]

        try:
            rate = self._fetch_current_rate(currency)
            self._current_rates[currency] = (rate, time.time())
            return rate
        except Exception as e:
            print(f"FX: failed to get rate for {currency}: {e}")
            return 1.0

    def _fetch_current_rate(self, currency: str) -> float:
        """Fetch current rate for a standard (non-sub-unit) currency to base."""
        if self.base_currency == "USD":
            return self._get_yf_rate(f"{currency}USD=X")
        # source -> USD -> base
        rate_to_usd = self._get_yf_rate(f"{currency}USD=X")
        rate_usd_to_base = self._get_yf_rate(f"USD{self.base_currency}=X")
        return rate_to_usd * rate_usd_to_base

    @staticmethod
    def _get_yf_rate(pair: str) -> float:
        """Get rate from a yfinance FX pair ticker."""
        fx = yf.Ticker(pair)
        rate = fx.info.get("regularMarketPrice")
        if rate is None or rate <= 0:
            rate = fx.info.get("previousClose", 1.0)
        return rate

    def get_rate_series(self, currency: str, index: pd.DatetimeIndex) -> pd.Series:
        """Get historical FX rates aligned to the given DatetimeIndex."""
        currency = currency.upper()
        if currency == self.base_currency:
            return pd.Series(1.0, index=index)

        lookup, divisor = self._resolve_sub_unit(currency)
        history = self._get_fx_history(lookup)
        if history is None or history.empty:
            return pd.Series(self.get_rate(currency), index=index)

        # Align to requested index using forward-fill, then back-fill leading NaNs
        aligned = history.reindex(index, method="ffill").bfill()
        aligned = aligned.fillna(self.get_rate(lookup))
        return aligned / divisor

    def get_rate_at(self, currency: str, date) -> float:
        """Get FX rate at a specific date."""
        currency = currency.upper()
        if currency == self.base_currency:
            return 1.0

        lookup, divisor = self._resolve_sub_unit(currency)
        history = self._get_fx_history(lookup)
        if history is None or history.empty:
            return self.get_rate(currency)

        ts = pd.Timestamp(date)
        if ts in history.index:
            return history.loc[ts] / divisor

        rate = history.asof(ts)
        if pd.isna(rate):
            return self.get_rate(currency)
        return rate / divisor

    def _get_fx_history(self, currency: str) -> pd.Series:
        """Fetch and cache 10y daily FX history for a standard currency -> base_currency."""
        cached = self._history_cache.get(currency)
        if cached is not None:
            return cached

        try:
            if self.base_currency == "USD":
                hist = self._fetch_pair_history(f"{currency}USD=X")
            else:
                hist_to_usd = self._fetch_pair_history(f"{currency}USD=X")
                hist_usd_to_base = self._fetch_pair_history(f"USD{self.base_currency}=X")
                # Align and multiply
                common_idx = hist_to_usd.index.union(hist_usd_to_base.index)
                hist_to_usd = hist_to_usd.reindex(common_idx, method="ffill")
                hist_usd_to_base = hist_usd_to_base.reindex(common_idx, method="ffill")
                hist = (hist_to_usd * hist_usd_to_base).dropna()

            self._history_cache[currency] = hist
            return hist
        except Exception as e:
            print(f"FX: failed to get history for {currency}->{self.base_currency}: {e}")
            self._history_cache[currency] = pd.Series(dtype=float)
            return self._history_cache[currency]

    @staticmethod
    def _fetch_pair_history(pair: str) -> pd.Series:
        """Fetch 10y daily close for an FX pair, return tz-naive normalized Series."""
        hist = yf.Ticker(pair).history(period="10y")["Close"]
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        hist.index = hist.index.normalize()
        return hist[~hist.index.duplicated(keep='last')]


# Module-level singleton
fx_converter = FxConverter()


class YfTickerUSD:
    """Wrapper around yf.Ticker that converts all values to the configured base currency."""

    def __init__(self, ticker):
        if isinstance(ticker, str):
            self._inner = yf.Ticker(ticker)
        else:
            self._inner = ticker

        try:
            raw_info = self._inner.info
            self._price_currency = (raw_info.get("currency") or _BASE_CURRENCY).upper()
            self._financial_currency = (raw_info.get("financialCurrency") or self._price_currency).upper()
        except Exception:
            self._price_currency = _BASE_CURRENCY
            self._financial_currency = _BASE_CURRENCY

    @property
    def info(self) -> dict:
        """Return .info with price/monetary fields converted to base currency."""
        raw = self._inner.info
        if self._price_currency == _BASE_CURRENCY:
            return raw

        rate = fx_converter.get_rate(self._price_currency)
        converted = dict(raw)
        for field in _INFO_PRICE_FIELDS | _INFO_PER_SHARE_FIELDS:
            if field in converted and converted[field] is not None:
                try:
                    converted[field] = converted[field] * rate
                except (TypeError, ValueError):
                    pass
        converted["currency"] = _BASE_CURRENCY
        return converted

    def history(self, *args, **kwargs) -> pd.DataFrame:
        """Return price history converted to base currency with tz-naive normalized index."""
        df = self._inner.history(*args, **kwargs)
        if df.empty:
            return df

        # Normalize timezone
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index = df.index.normalize()

        # Convert currency
        if self._price_currency != _BASE_CURRENCY:
            fx_series = fx_converter.get_rate_series(self._price_currency, df.index)
            for col in _HISTORY_PRICE_COLS:
                if col in df.columns:
                    df[col] = df[col] * fx_series

        return df

    def _convert_financial_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert a financial statement DataFrame to base currency."""
        if df is None or df.empty or self._financial_currency == _BASE_CURRENCY:
            return df
        df = df.copy()
        for col in df.columns:
            rate = fx_converter.get_rate_at(self._financial_currency, col)
            df[col] = df[col] * rate
        return df

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return self._convert_financial_df(self._inner.balance_sheet)

    @property
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        return self._convert_financial_df(self._inner.quarterly_balance_sheet)

    @property
    def income_stmt(self) -> pd.DataFrame:
        return self._convert_financial_df(self._inner.income_stmt)

    @property
    def quarterly_income_stmt(self) -> pd.DataFrame:
        return self._convert_financial_df(self._inner.quarterly_income_stmt)

    @property
    def cash_flow(self) -> pd.DataFrame:
        return self._convert_financial_df(self._inner.cash_flow)

    @property
    def quarterly_cash_flow(self) -> pd.DataFrame:
        return self._convert_financial_df(self._inner.quarterly_cash_flow)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class YfTickersUSD:
    """Wrapper around yf.Tickers that converts all values to the configured base currency."""

    def __init__(self, tickers):
        if isinstance(tickers, str):
            self._inner = yf.Tickers(tickers)
        else:
            self._inner = tickers

        # Build currency map and wrapped tickers
        self._currencies = {}
        self._wrapped_tickers = {}
        for symbol, ticker in self._inner.tickers.items():
            try:
                currency = (ticker.info.get("currency") or _BASE_CURRENCY).upper()
            except Exception:
                currency = _BASE_CURRENCY
            self._currencies[symbol] = currency
            self._wrapped_tickers[symbol] = YfTickerUSD(ticker)

    @property
    def tickers(self) -> dict:
        return self._wrapped_tickers

    def history(self, *args, **kwargs) -> pd.DataFrame:
        """Return bulk price history converted to base currency with tz-naive normalized index."""
        df = self._inner.history(*args, **kwargs)
        if df.empty:
            return df

        # Normalize timezone
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index = df.index.normalize()

        # Convert per-symbol
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            for symbol, currency in self._currencies.items():
                if currency == _BASE_CURRENCY:
                    continue
                fx_series = fx_converter.get_rate_series(currency, df.index)
                for field in _HISTORY_PRICE_COLS:
                    if (field, symbol) in df.columns:
                        df[(field, symbol)] = df[(field, symbol)] * fx_series
        else:
            df = df.copy()
            for symbol, currency in self._currencies.items():
                if currency == _BASE_CURRENCY or symbol not in df.columns:
                    continue
                fx_series = fx_converter.get_rate_series(currency, df.index)
                df[symbol] = df[symbol] * fx_series

        return df

    def __getattr__(self, name):
        return getattr(self._inner, name)
