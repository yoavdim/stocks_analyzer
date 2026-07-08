#!/usr/bin/python3

import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import os
import json
from pypfopt.risk_models import fix_nonpositive_semidefinite, CovarianceShrinkage
from fx_converter import YfTickerUSD, YfTickersUSD
from tase_adapter import is_tase_fund, TaseTickerUSD, TaseTickersUSD

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npv_config.json")
with open(_config_path, "r") as _f:
    _PORTFOLIO_CONFIG = json.load(_f).get("portfolio", {})

# "Now" is decided once at app start and held fixed, so every price fetch/slice
# is consistent (the app behaves as if frozen in time) instead of resampling.
NOW_DATE = datetime.date.today()


class YfinanceException(Exception):
    """Exceptions that are thrown from the yfinance class. The class is
    responsible for fetching the financial data from Yahoo. This includes mostly
    stock data such as price."""
    pass


def is_stock(yf_ticker) -> bool:
    return yf_ticker.info.get("quoteType", "") == "EQUITY"


market_to_yf_market = {
        "NASDAQ"    : None,  # None value will leave the symbol intact
        "NYSE"      : None,
        "AMEX"      : None,
        "TPE"       : "TW",  # Taiwan
        "TYO"       : "T",   # Japan
        "LON"       : "L",   # UK
        "SWX"       : "SW",  # Switzerland
        "AMS"       : "AS",  # Holland
        "STO"       : "ST",  # Sweden
        "TLV"       : "TA",  # Israel
        "KRX"       : "KS",  # Korea
        "SHE"       : "SZ",
        "TSE"       : "TO",  # Toronto
        "ASX"       : "AX",  # Australia
    }

def get_ticker_from_standard_symbols(symbol:str, market:str):
    full_symbol = symbol.replace('.', '-').replace(' ', '-')  # for tickers like "brk.b"

    if market not in market_to_yf_market.keys():
        raise YfinanceException("unrecognised market")

    market_endian = market_to_yf_market[market]
    if market_endian is not None:
        full_symbol = full_symbol + "." + market_endian
    else:
        full_symbol = full_symbol
    return full_symbol, market_endian


class YahooInfo:

    def get_stock_price_now(self):
        """Get the current stock price in USD, or, if the market is closed, the
        closing price. Cached for the lifetime of this object (one fetch per
        session); not pickled, so a new process refetches a fresh price."""
        if getattr(self, "_price_now", None) is None:
            todays_data = self.yf_ticker.history(period='5d')
            self._price_now = todays_data['Close'].dropna().iloc[-1]
        return self._price_now

    def prefetch_price_range(self, from_date):
        """Fetch the daily Close series [from_date, now] ONCE into a transient
        in-memory cache (_range_series). Every price lookup (per-date and graph
        slices) is served from this single fetch. Not pickled.

        The range always runs to today, so its last point is the current price;
        it is pinned to the frozen _price_now (decided once) so the app stays
        consistent even as the market keeps moving."""
        if getattr(self, "_range_series", None) is not None:
            return  # already fetched this session (first report date -> now)
        if isinstance(from_date, datetime.datetime):
            from_date = from_date.date()

        close = self.yf_ticker.history(start=from_date, end=NOW_DATE + datetime.timedelta(days=1))["Close"].dropna()
        self._range_series = close
        
        if getattr(self, "_price_now", None) is None:
            # first observation of 'now' -> freeze it from the range's last close
            self._price_now = float(close.iloc[-1])
        else:
            # freeze in time: force the last point to the already-decided price
            close.iloc[-1] = self._price_now

    def get_stock_price_at_date(self, day, month, year):
        """Stock price in USD at the given date (or the closest trading day
        after it), served from the prefetched full-range series. Call
        prefetch_price_range first; for a one-off lookup without a prefetched
        range use fetch_price_at_date_uncached."""
        date_str = "{year}-{month}-{day}".format(day=str(day).zfill(2), month=str(month).zfill(2), year=year)
        if date_str in self.stock_prices:
            return self.stock_prices[date_str]

        series = getattr(self, "_range_series", None)
        # nearest trading day >= target (bfill); NaN if target is past the range
        price = series.reindex([pd.Timestamp(year=year, month=month, day=day)], method="bfill").iloc[0]
        self.stock_prices[date_str] = price
        return price

    def fetch_price_at_date_uncached(self, day, month, year):
        """One-off uncached price lookup: does its own small windowed fetch and
        does NOT touch the shared price cache or the prefetched range. Use for a
        single past date when no range prefetch is warranted."""
        # fetch from a day *after* the requested one because yahoo returns the
        # stock one day earlier than requested
        start = datetime.date(day=day, month=month, year=year) + datetime.timedelta(days=1)
        stocks_data = self.yf_ticker.history(start=start, end=start + datetime.timedelta(days=10))
        return stocks_data["Close"].iloc[0] if len(stocks_data) else float("nan")

    def get_stock_price_in_range(self, from_date, to_date):
        # Served from the shared daily range series (one fetch, first report
        # date -> now). Returns only the requested [from_date, to_date] slice as
        # (DatetimeIndex, Close Series), so the price graph and the per-date
        # lookups all draw from a single network round-trip.
        # @return date & price vectors
        self.prefetch_price_range(from_date)
        series = self._range_series if self._range_series is not None else pd.Series(dtype=float)
        sliced = series.loc[pd.Timestamp(from_date):pd.Timestamp(to_date)]
        return sliced.index, sliced

    def pre_pickle(self, short_term):
        self.yf_ticker = None
        # transient full-range daily price series, only alive during a build.
        # removing also in short term to save on memory for the entire ticker list TODO: this will cause a refetch in npv_calculator price plot
        self._range_series = None
        # a short-term pickle (e.g. across the process pool) keeps the freshly
        # fetched price; the long-term disk cache must not persist a stale price
        if not short_term:
            self._price_now = None

    def post_pickle(self, yf_ticker=None, price_now=None):
        self.yf_ticker = yf_ticker if yf_ticker else YfTickerUSD(self.full_symbol)
        # restore the live price the same way as yf_ticker (None if not provided)
        self._price_now = price_now
        self._range_series = None  # transient, never persisted
        # seed currencies from the pickled info so the fresh wrapper doesn't
        # re-fetch .info just to detect currency
        self.yf_ticker.set_currencies(self.info.get("currency"), self.info.get("financialCurrency"))

    def __init__(self, symbol, market, *, yf_info = None):
        self.full_symbol, self.market_endian = get_ticker_from_standard_symbols(symbol, market)
        if yf_info:
            self.yf_ticker = yf_info
        elif is_tase_fund(symbol, market):
            self.yf_ticker = TaseTickerUSD(symbol)
        else:
            self.yf_ticker = YfTickerUSD(self.full_symbol)
        try:
            self.info = self.yf_ticker.info
            self.stock_prices = dict()
            self._price_now = None  # lifetime cache for get_stock_price_now
            # full-range daily price series  - stripped during pickle.
            self._range_series = None
        except:
            raise YfinanceException("Failed to create yf symbol {} or fetch its info".format(self.full_symbol))



class YahooGroup:
    """ Get prices synchronised, all converted to USD """
    def __init__(self, symbols: list, markets: list):
        self.symbols = symbols
        self.markets = markets
        self.history = None
        self.full_symbols = list()
        self.valid_full_symbols = None

        yahoo_full_symbols = []
        tase_tickers = {}

        for i in range(len(symbols)):
            full_sym = get_ticker_from_standard_symbols(symbols[i], markets[i])[0]
            self.full_symbols.append(full_sym)
            if is_tase_fund(symbols[i], markets[i]):
                tase_tickers[full_sym] = TaseTickerUSD(symbols[i])
            else:
                yahoo_full_symbols.append(full_sym)

        if tase_tickers:
            inner_yf = YfTickersUSD(yf.Tickers(" ".join(yahoo_full_symbols))) if yahoo_full_symbols else None
            self.yf_ticker = TaseTickersUSD(inner_yf, tase_tickers)
        else:
            self.yf_ticker = YfTickersUSD(yf.Tickers(" ".join(self.full_symbols)))

    def calculate_correlation(self):
        self.get_monthly_prices()
        self.get_cov()

    def get_monthly_prices(self) -> None:
        """
        TODO: return vectors of dates, weight(price), growth(divided by price) - will be used by portfolio volatility analysis
        should cache the result?
        should take into account splits and dividends(will be added to the monthly growth)
        """
        if self.history is None:
            self.history = self.yf_ticker.history(period="10y")["Close"].iloc[::30]  # todo better implement period & interval (more years and real months?, maybe add overlaps)
            self.history = self.history.ffill(limit=4)  # fill short gaps (holidays, exchange mismatches) but preserve real missing data
            # Valid if ≥80% non-NaN in the last 6 years, or ≥60% over the full 10 years
            cutoff = self.history.index[-1] - pd.DateOffset(years=6)
            recent = self.history.loc[self.history.index >= cutoff]
            self.valid_full_symbols = [
                col for col in self.history.columns
                if recent[col].notna().mean() >= 0.8 and self.history[col].notna().mean() >= 0.6
            ]

    def get_monthly_growths(self):
        p1 = self.history[1:]
        p0 = self.history[:-1]
        p0.index = p1.index
        return (p1 - p0) / p0

    def get_past_annual_performance(self, symbol, market, is_yahoo=False):
        full_symbol = symbol if is_yahoo else get_ticker_from_standard_symbols(symbol,market)[0]
        monthly = self.get_monthly_growths()[full_symbol].dropna().mean()
        return (1 + monthly) ** 12 - 1

    def get_cov(self):
        monthly = self.get_monthly_growths()
        valid_monthly = monthly[self.valid_full_symbols]

        if _PORTFOLIO_CONFIG.get("use_shrinkage", True):
            cs = CovarianceShrinkage(valid_monthly, returns_data=True, frequency=12)
            cov = cs.ledoit_wolf()
        else:
            cov = valid_monthly.cov() * 12  # multiply by 12 to convert from monthly to annual variance
            cov = fix_nonpositive_semidefinite(cov)

        self.cov = cov.values if hasattr(cov, 'values') else cov

    # -----------------------------------------------------------------------------

    def get_stock_prices_now(self):
        """Get the current stock price, or, if the market is closed, the closing price,
        without caching, as the price continue to change"""
        todays_data = self.yf_ticker.history(period='5d')
        prices = todays_data['Close'].ffill().iloc[-1]
        # yfinance returns columns sorted alphabetically; reindex 
        return prices.reindex(self.full_symbols)

    def get_market_caps(self, existing_tickers: dict = None) -> list:
        """Return market caps aligned with self.full_symbols.

        Reads cached Ticker.statistics["market_cap"] when available
        (existing_tickers maps (symbol, market) -> Ticker), falling back to
        yf_ticker.tickers[fsym].info.get("marketCap"). Missing values are None.
        """
        existing_tickers = existing_tickers or {}
        caps = []
        for sym, mkt, fsym in zip(self.symbols, self.markets, self.full_symbols):
            mc = None
            ticker = existing_tickers.get((sym, mkt))
            if ticker is not None:
                v = ticker.statistics.get("market_cap")
                if v and not (isinstance(v, float) and np.isnan(v)):
                    mc = v
            if mc is None:
                try:
                    v = self.yf_ticker.tickers[fsym].info.get("marketCap")
                    if v and not (isinstance(v, float) and np.isnan(v)):
                        mc = v
                except Exception as e:
                    print(f"{fsym}: error reading marketCap ({e})")
            caps.append(mc)
        return caps




if __name__ == '__main__':  # test index
    import matplotlib.pyplot as plt
    y = YahooInfo('%5EGSPC', 'NYSE')  # S&P500
    y2 = YahooInfo("MSFT","NASDAQ")
    y3 = YahooGroup(["MSFT","AAPL"],["NASDAQ", "NASDAQ"])
    fig = plt.figure()
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=(365.25*4))
    date_vector, price_vector = y.get_stock_price_in_range(start_date, end_date)
    plt.plot(date_vector, price_vector, '-')
    plt.show()

