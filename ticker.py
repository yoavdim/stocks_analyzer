#!/usr/bin/env python3
import warnings
import json
from copy import deepcopy

from pypfopt import EfficientFrontier, plotting

from reports import Reports
from yahoo_reports import YReports
from yfinance_info import YahooInfo, YahooGroup, is_stock
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
import pickle
from pprint import pformat
import datetime
import os.path
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.widgets
import sys

# Load NPV assumptions from config
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npv_config.json")
with open(_config_path, "r") as _f:
    _config = json.load(_f)
    NPV_COMMON = _config["npv_common"]
    NPV_ASSUMPTIONS = _config["npv_assumptions"]
    LINEAR_IRR_CONFIG = _config["linear_irr"]
    PORTFOLIO_CONFIG = _config["portfolio"]
    TLDR_FIELDS = _config.get("tldr_fields", [])
    QUICK_FILTERS = _config.get("quick_filters", {})

# Define:
tickers_dir = "./tickers_cache"
cache_file_name = "{symbol}-{market}.pkl"

forcast_growth_field = PORTFOLIO_CONFIG["forecast_growth_field"]


# Forecast policies for the efficient-frontier expected return of each ticker.
# id -> human-readable label (used by the portfolio_analyzer startup dialog).
# Non-stocks (indices/ETFs) always use past growth regardless of policy.
FORECAST_POLICIES = {
    "past":                  "Past growth",
    "irr_past_if_nan":       "IRR (past if NaN)",
    "irr_past_if_no_model":  "IRR (past if no model)",
    "irr_filter_if_nan":     "IRR (filter if NaN)",
    "irr_filter_if_no_model":"IRR (filter if no model)",
}


# --- DCF model persistence ---
_DCF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcf_models")

def save_dcf_model(symbol: str, market: str, **kwargs):
    """Save DCF parameters for a ticker to dcf_models/.
    Note: discount_rate_percent is saved for UI pre-fill only — it is NOT used
    in IRR/intrinsic value calculations (those pass the rate explicitly)."""
    os.makedirs(_DCF_DIR, exist_ok=True)
    data = {"symbol": symbol.upper(), "market": market.upper(),
            "saved_at": datetime.date.today().isoformat(), **kwargs}
    path = os.path.join(_DCF_DIR, f"{symbol.upper()}-{market.upper()}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_dcf_model(symbol: str, market: str) -> dict | None:
    """Load saved DCF parameters. Returns None if not found."""
    path = os.path.join(_DCF_DIR, f"{symbol.upper()}-{market.upper()}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def dcf_remaining_growth_years(model: dict) -> float:
    end = datetime.date.fromisoformat(model["growth_phase_end"])
    return max((end - datetime.date.today()).days / 365.25, 0)


def compute_avg_fcf(plot_data: dict, basis: str) -> tuple[float, float]:
    """Compute average free cash flow per share based on chosen basis.
    basis: '4-year avg' (mean of last 4 annual reports, excluding TTM) or 'TTM' (most recent TTM value).
    Returns (avg_fcf, years_behind_last_report) — the offset to roll forward to last_annual_date."""
    fcf_series = plot_data.get("free_cf_ps", plot_data["free_cf"])
    if basis == "TTM":
        return fcf_series[-1], 0.0
    # Exclude the TTM (last element) and take at most 4 annual reports
    annual = fcf_series[:-1] if len(fcf_series) > 1 else fcf_series
    annual = annual[-4:]  # at most 4 years
    n = len(annual)
    midpoint_offset = (n - 1) / 2
    return np.mean(annual), midpoint_offset


def resolve_add_value(plot_data: dict, add_mode: str) -> float:
    """Resolve the per-unit balance-sheet amount to add to a DCF intrinsic value.

    add_mode: "none" -> 0, "book_value" -> latest BV/unit, "cash" -> latest cash/unit.
    Returns 0.0 for "none" or when the underlying series is missing/NaN.
    """
    if add_mode in (None, "none"):
        return 0.0
    key = {"book_value": "bv", "cash": "cash_ps"}.get(add_mode)
    if key is None:
        raise ValueError(f"Unknown add_mode: {add_mode}")
    series = plot_data.get(key)
    if series is None or len(series) == 0:
        return 0.0
    value = series[-1]
    return 0.0 if (value is None or np.isnan(value)) else float(value)


class MarketDataCache:
    """Caches risk-free rate (^TNX) and S&P500 1yr return with a 1-hour TTL."""
    _TTL = 3600

    def __init__(self):
        self._cache = {"rfr": (None, 0), "mkt": (None, 0), "mkt_std": (None, 0), "mkt_monthly": (None, 0)}

    def _get(self, key, fetch_fn):
        import time
        value, ts = self._cache[key]
        if value is None or time.time() - ts > self._TTL:
            value = fetch_fn()
            self._cache[key] = (value, time.time())
        return value

    def get_risk_free_rate(self) -> float:
        return self._get("rfr", lambda: YahooInfo("%5ETNX", "NASDAQ").get_stock_price_now() / 100)

    def get_market_return(self) -> float:
        import datetime
        def fetch():
            spx = YahooInfo("%5EGSPC", "NASDAQ")
            d = datetime.date.today() - datetime.timedelta(days=365)
            old = spx.fetch_price_at_date_uncached(d.day, d.month, d.year)
            return (spx.get_stock_price_now() - old) / old
        return self._get("mkt", fetch)

    def get_market_monthly_returns(self) -> pd.Series:
        def fetch():
            from fx_converter import YfTickerUSD
            market_ticker = YfTickerUSD("^GSPC")
            market_hist = market_ticker.history(period="10y")["Close"].iloc[::30]
            return market_hist.pct_change().dropna()
        return self._get("mkt_monthly", fetch)

    def get_market_std(self) -> float:
        def fetch():
            market_returns = self.get_market_monthly_returns()
            return np.sqrt(market_returns.var() * 12)
        return self._get("mkt_std", fetch)


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta as Cov(R_asset, R_market) / Var(R_market)."""
    common = asset_returns.dropna().index.intersection(market_returns.index)
    if len(common) < 3:
        return np.nan
    market_var = market_returns.loc[common].var()
    if market_var == 0:
        return np.nan
    return asset_returns.loc[common].cov(market_returns.loc[common]) / market_var


def calculate_beta_from_history(yf_ticker, market_returns: pd.Series) -> float:
    """Calculate beta for a single yfinance ticker object from its price history.
    Used as a fallback when yfinance .info doesn't provide beta."""
    try:
        hist = yf_ticker.history(period="10y")["Close"].iloc[::30]
        asset_returns = hist.pct_change().dropna()
        return calculate_beta(asset_returns, market_returns)
    except Exception as e:
        print(f"Beta fallback calculation failed: {e}")
        return np.nan


market_data = MarketDataCache()


class StatisticsException(Exception):
    """Exceptions that are thrown during the Ticker statistics calculation.
        This will happen mostly due to a bug"""
    pass


def get_exception_line():
    """use inside except"""
    frame = sys.exc_info()[2]
    while frame.tb_next:
        frame = frame.tb_next
    return frame.tb_lineno


def search_growth(npv_function, price, min_growth,
                  max_growth=NPV_ASSUMPTIONS["irr_search_max"],
                  delta_growth=NPV_ASSUMPTIONS["irr_search_step_percent"]/100,
                  monotone=False):
    """
    Find the IRR/growth rate where npv_function(rate) == price.
    :param npv_function: function(growth_rate) -> npv value
    :param price: target value to match
    :param min_growth: lower bound of search range
    :param max_growth: upper bound of search range
    :param delta_growth: step size for linear scan / tolerance for bisection
    :param monotone: if True, use binary search (assumes npv is monotonically decreasing with rate)
    :return: IRR in percent
    """
    if monotone:
        irr = _bisect_search(npv_function, price, min_growth, max_growth, delta_growth)
        if not np.isnan(irr):
            return irr
        # Fallback to linear scan if bisection failed

    return _linear_search(npv_function, price, min_growth, max_growth, delta_growth)


def _bisect_search(npv_function, price, lo, hi, tol):
    """Find the rate where npv_function(rate) == price using scipy's Brent method."""
    from scipy.optimize import brentq

    def f(rate):
        with np.errstate(all='ignore'):
            v = npv_function(rate)
        if v is None or np.isnan(v):
            raise ValueError
        if not np.isfinite(v):
            v = np.sign(v) * 1e18
        return v - price

    try:
        return brentq(f, lo, hi, xtol=tol) * 100
    except (ValueError, RuntimeError):
        return np.nan


def _linear_search(npv_function, price, min_growth, max_growth, delta_growth):
    """Original linear scan search."""
    best_result = None
    best_growth = np.nan
    for growth in range(0, int(1 + (max_growth - min_growth) / delta_growth)):
        growth = delta_growth * growth + min_growth
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("error")
                with np.errstate(all='ignore'):
                    npv = npv_function(growth)
        except Exception:
            npv = np.nan
        if npv is None or np.isnan(npv) or not np.isfinite(npv):
            continue
        error = np.abs(npv - price)
        if (best_result is None) or error < best_result:
            best_result = error
            best_growth = growth

    irr = best_growth * 100
    if best_result is not None and price > 0:
        relative_error = best_result / price
        if relative_error > 0.05:
            print(f"IRR search: poor fit (error {relative_error*100:.1f}% of price)")
    elif best_result is None:
        print("IRR search: no valid NPV found in search range")
    return irr


def build_dcf_model(average_annual_free_cash_flow, add_value, diluted_shares, growth, last_annual_date,
                    forward_to_present=NPV_ASSUMPTIONS["forward_to_present"],
                    short_term_is_linear=NPV_ASSUMPTIONS["short_term_is_linear"],
                    long_term_growth_duration=NPV_ASSUMPTIONS["long_term_growth_duration"],
                    short_term_growth_duration=NPV_ASSUMPTIONS["short_term_growth_duration"],
                    maximal_long_term_growth_rate=NPV_ASSUMPTIONS["maximal_long_term_growth_rate_percent"]/100):
    """
    Build a DCF valuation model. Returns calc_npv(discount_rate) -> intrinsic value per unit.

    Parameters
    ----------
    average_annual_free_cash_flow : float — average annual free cash flow
    add_value : float — a last-report balance-sheet amount added to intrinsic value per
        unit (e.g. book value per share, or cash per share). Pass 0 for none.
    diluted_shares : float — number of units to divide by (shares for Ticker, 1 for Portfolio)
    growth : float — short-term growth. Interpretation depends on short_term_is_linear:
        - False: percent compounding rate (e.g. 10 means 10%/yr)
        - True:  absolute FCF increment per year (same units as average_annual_free_cash_flow)
    last_annual_date : datetime — date of last annual report (for forward_to_present)
    """
    if short_term_is_linear:
        forcasted_short_term_growth_rate = 0  # unused in the linear short-term branch
        linear_growth = growth
        forcasted_long_term_growth_rate = maximal_long_term_growth_rate  # independent of short-term
    else:
        forcasted_short_term_growth_rate = growth / 100
        linear_growth = 0
        forcasted_long_term_growth_rate = np.min([maximal_long_term_growth_rate, forcasted_short_term_growth_rate])

    def calc_npv(discount_rate):
        assert discount_rate > -100E-2, "discontinuity at -1"
        # we calculate the q of the geometric series
        short_term_q = (1 + forcasted_short_term_growth_rate) / (1 + discount_rate)
        long_term_q = (1 + forcasted_long_term_growth_rate) / (1 + discount_rate)

        # sum over the short term
        if not short_term_is_linear:
            first_short_term = average_annual_free_cash_flow * short_term_q
            last_short_term = average_annual_free_cash_flow * (short_term_q ** short_term_growth_duration)
            if short_term_q == 1:  # discontinuity point
                sum_discounted_fcf_short_term = first_short_term * short_term_growth_duration
            else:
                sum_discounted_fcf_short_term = (first_short_term - last_short_term * short_term_q) / (1 - short_term_q)
        else:
            # there is no easy formula for *discounted* constant addition, we will calculate explicitly
            terms = np.arange(1, short_term_growth_duration + 1)
            fcf = average_annual_free_cash_flow + linear_growth * terms
            dfcf = fcf / (1 + discount_rate) ** terms
            sum_discounted_fcf_short_term = np.sum(dfcf)
            last_short_term = dfcf[-1]

        # and from its ending to eternity
        first_long_term = last_short_term * long_term_q
        if long_term_growth_duration < 0:
            if long_term_q >= 1:  # edge cases
                sum_discounted_fcf_long_term = np.sign(first_long_term) * np.inf
            elif long_term_q <= -1:
                sum_discounted_fcf_long_term = np.nan
            else:
                sum_discounted_fcf_long_term = first_long_term / (1 - long_term_q)
        else:  # (a1-an*q)/(1-q)
            last_long_term_times_q = last_short_term * long_term_q ** (long_term_growth_duration + 1)
            if long_term_q == 1:  # discontinuity point
                sum_discounted_fcf_long_term = first_long_term * long_term_growth_duration
            else:
                sum_discounted_fcf_long_term = (first_long_term - last_long_term_times_q) / (1 - long_term_q)

        intrinsic_value_dcf = (sum_discounted_fcf_short_term + sum_discounted_fcf_long_term) / diluted_shares
        intrinsic_value_dcf += add_value
        if forward_to_present:
            years = FundamentalMixin._calculate_time_forward(last_annual_date)
            intrinsic_value_dcf *= (discount_rate + 1) ** years
        return intrinsic_value_dcf

    return calc_npv


class FundamentalMixin:
    """
    Mixin providing shared financial analysis methods for both Ticker and Portfolio.
    This class represent an object which we know the financial data of.
    Subclasses must implement:
        _get_plot_data() -> dict   (time series of BV, EPS, FCF, prices)
        get_current_price() -> float  (current total price: share price for Ticker, portfolio value for Portfolio)
    """

    @staticmethod
    def _calculate_time_forward(last_annual_date):
        return (datetime.datetime.now() - last_annual_date).days / 365.25

    def get_growth_rate(self, field="eps", linear=False):
        """Annualized growth rate via log-linear regression on the given field (eps, bv, revenue_ps).
        linear=True: linear regression slope (absolute annual change)."""
        data = self._get_plot_data()
        if data is None or field not in data:
            return np.nan
        values = data[field]
        times = data["times"]
        if len(values) < 3 or (not linear and (values <= 0).any()):
            return np.nan
        first_date = times[0]
        years = np.array([(d - first_date).days / 365.25 for d in times])
        try:
            if linear:
                b, _ = np.polyfit(years, values, 1)
                return b
            else:
                b, _ = np.polyfit(years, np.log(values), 1)
                return (np.exp(b) - 1) * 100
        except Exception:
            return np.nan

    def get_projected_pe(self):
        """PE using linear extrapolation of earnings to today."""
        from numpy.polynomial.polynomial import Polynomial
        data = self._get_plot_data()
        if data is None:
            return np.nan
        eps = data["eps"]
        times = data["times"]
        if len(eps) < 2:
            return np.nan
        days = np.array([(t - times[0]).days for t in times])
        poly_fit = Polynomial.fit(days, eps, deg=1)
        earnings_fit = poly_fit.convert().coef
        now_days = (datetime.datetime.now() - times[0]).days
        forecasted_eps = earnings_fit[0] + now_days * earnings_fit[1]
        if forecasted_eps <= 0:
            return np.nan
        return self.get_current_price() / forecasted_eps

    def _build_dcf_from_plot_data(self, growth=None, avg_fcf=None, add_mode=None, **kwargs):
        """Build a DCF model from _get_plot_data. Returns (calc_npv, price) or (None, None).
        If a saved DCF model exists and no explicit parameters are passed, uses saved parameters.
        `growth` interpretation matches build_dcf_model: percent in exponential mode,
        absolute FCF/share/yr in linear mode (kwargs['short_term_is_linear']).
        `add_mode` in {"none","book_value","cash"} — balance-sheet amount added post-roll."""
        data = self._get_plot_data()
        if data is None:
            return None, None

        # Apply saved DCF model only when no explicit parameters are passed.
        # Callers with explicit args (e.g. linear IRR, NPV calculator) bypass this.
        if growth is None and avg_fcf is None and add_mode is None and not kwargs:
            model = getattr(self, 'dcf_model', None)
            if model:
                remaining_years = dcf_remaining_growth_years(model)
                if remaining_years <= 0:
                    print(f"Warning: {getattr(self, 'symbol', '?')} saved DCF model growth phase has ended")
                days_since_save = (datetime.date.today() - datetime.date.fromisoformat(model["saved_at"])).days
                if days_since_save > 180:
                    print(f"Warning: {getattr(self, 'symbol', '?')} DCF model is {days_since_save} days old, consider updating")
                is_linear = (model["growth_trend"] == "Linear")
                growth = model["linear_growth"] if is_linear else model["growth_rate_percent"]
                avg_fcf, fcf_offset = compute_avg_fcf(data, model["fcf_basis"])
                # Roll avg_fcf forward to last_annual_date using the appropriate growth model
                if fcf_offset > 0:
                    if is_linear:
                        avg_fcf += growth * fcf_offset
                    else:
                        avg_fcf *= (1 + growth / 100) ** fcf_offset
                add_mode = model["add_mode"]
                kwargs = {
                    "short_term_is_linear": is_linear,
                    "short_term_growth_duration": remaining_years,
                    "long_term_growth_duration": 0 if model["terminal_model"] == "Nothing" else -1,
                    "maximal_long_term_growth_rate": model["terminal_growth_percent"] / 100 if model["terminal_model"] == "Slow Exponent" else 0,
                }

        gr = growth if growth is not None else self.get_growth_rate("eps")
        if np.isnan(gr):
            return None, None
        price = self.get_current_price()
        if price <= 0:
            return None, None

        if avg_fcf is None:
            avg_fcf = np.mean(data.get("free_cf_ps", data["free_cf"]))

        calc_npv = build_dcf_model(
            average_annual_free_cash_flow=avg_fcf,
            add_value=resolve_add_value(data, add_mode),
            diluted_shares=1,
            growth=gr,
            last_annual_date=data["times"][-1] if data["times"] else datetime.datetime.now(),
            forward_to_present=True,
            **kwargs,
        )
        return calc_npv, price

    def get_irr(self, linear=False):
        """DCF-based IRR. If linear=True, uses linear growth model."""
        try:
            data = self._get_plot_data()
            if data is None:
                return np.nan
            # Use saved fcf basis if available, otherwise 4-year avg
            model = getattr(self, 'dcf_model', None)
            fcf_basis = model.get("fcf_basis", "4-year avg") if model else "4-year avg"
            avg_fcf, _ = compute_avg_fcf(data, fcf_basis)
            # Note: offset correction is handled inside _build_dcf_from_plot_data for the saved model path

            if linear:
                avg_eps = np.mean(data["eps"])
                eps_slope = self.get_growth_rate("eps", linear=True)
                linear_growth = eps_slope * avg_fcf / avg_eps if avg_eps != 0 else 0
                calc_npv, price = self._build_dcf_from_plot_data(
                    avg_fcf=avg_fcf,
                    short_term_is_linear=True,
                    growth=linear_growth,
                    long_term_growth_duration=LINEAR_IRR_CONFIG["long_term_growth_duration"],
                    short_term_growth_duration=LINEAR_IRR_CONFIG["short_term_growth_duration"],
                )
                # Monotone if FCF positive and linear growth is non-decreasing
                is_monotone = avg_fcf > 0 and linear_growth >= 0
            else:
                calc_npv, price = self._build_dcf_from_plot_data()
                # Monotone if FCF positive (exponential growth preserves sign)
                is_monotone = avg_fcf > 0
            if calc_npv is None:
                return np.nan
            return search_growth(calc_npv, price, min_growth=NPV_ASSUMPTIONS["irr_search_min"],
                                 monotone=is_monotone)
        except Exception as e:
            print(f"IRR calculation failed: {e}")
            return np.nan

    def get_capm_discount(self):
        """CAPM discount ratio: how much the DCF intrinsic value differs from market price using CAPM rate."""
        try:
            calc_npv, price = self._build_dcf_from_plot_data()
            if calc_npv is None:
                return np.nan
            beta = getattr(self, 'portfolio_beta', None) or (self.statistics.get("beta") if hasattr(self, 'statistics') else None)
            if beta is None or np.isnan(beta):
                return np.nan
            rfr = market_data.get_risk_free_rate()
            mkt = market_data.get_market_return()
            capm_npv = calc_npv(rfr + beta * (mkt - rfr))
            return 100 * (capm_npv - price) / price
        except Exception:
            return np.nan

    def get_intrinsic_value(self, discount_rate=NPV_COMMON["discount_rate_percent"]/100, **kwargs):
        """DCF intrinsic value at the given discount rate."""
        calc_npv, _ = self._build_dcf_from_plot_data(**kwargs)
        if calc_npv is None:
            return np.nan
        return calc_npv(discount_rate)

    def get_title(self):
        irr = self.get_irr()
        linear_irr = self.get_irr(linear=True)
        pe = self.get_projected_pe()
        price = self.get_current_price()
        name = self.get_name() if hasattr(self, 'get_name') else ""
        return "{} ({:,.2f}) — IRR {:.1f}% | Linear {:.1f}% | PE {:.1f}".format(name, price, irr, linear_irr, pe)

    def plot_me(self, show=True):
        data = self._get_plot_data()
        if data is None:
            print("Cannot plot: no data available")
            return

        price_data = self._get_plot_price_data()
        fig = plt.figure()
        gs = fig.add_gridspec(4, 2)

        times = data["times"]
        bv = data["bv"]
        eps = data["eps"]
        prices = data["prices"]

        # 00 - Book Value
        ax = fig.add_subplot(gs[0, 0])
        format_axis(ax)
        ax.plot(times, bv, '-', label="book value")
        ax.legend(framealpha=0.4)

        # 01 - EPS
        ax = fig.add_subplot(gs[0, 1])
        format_axis(ax)
        ax.plot(times, eps, '-', label="eps")
        ax.legend(framealpha=0.4)

        # 10 - Cash Flow
        ax = fig.add_subplot(gs[1, 0])
        format_axis(ax)
        ax.plot(times, data["operating_cf"], '-', label="operating")
        ax.plot(times, data["free_cf"], '-', label="free")
        ax.legend(framealpha=0.4)

        # 11 - Price
        ax = fig.add_subplot(gs[1, 1])
        format_axis(ax)
        ax.plot(times, prices, '-', label="price")
        ax.legend(framealpha=0.4)

        # 21 - PE & EPS Growth
        ax = fig.add_subplot(gs[2, 1])
        format_axis(ax)
        pe_ratios = prices / eps
        line1 = ax.plot(times, pe_ratios, '-', label="PE")
        ax.set_ylabel("PE")

        ax2 = ax.twinx()
        times_arr = np.array(times)
        dt_days = np.array(times_arr[1:] - times_arr[:-1], dtype='timedelta64[D]')
        dt_years = dt_days / np.timedelta64(1, 'D') / 365.25
        median_spacing = np.median(dt_years)

        if median_spacing < 0.5:
            eps_series = pd.Series(eps, index=times)
            eps_shifted = eps_series.shift(12)
            yoy = (eps_series / eps_shifted - 1) * 100
            yoy = yoy.dropna()
            line2 = ax2.plot(yoy.index, yoy.values, label="EPS Growth (YoY)", color="C1")
        else:
            time_for_deltas = times_arr[1:]
            de = eps[1:] / eps[:-1]
            growths = de ** (1 / dt_years)
            growths = (growths - 1) * 100
            line2 = ax2.plot(time_for_deltas, growths, label="EPS Growth", color="C1")

        ax2.set_ylabel("Growth")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, framealpha=0.4)

        # wide price graph with interactive selector
        ax = fig.add_subplot(gs[-1, :])
        format_axis(ax)
        ax.plot(price_data["price_times"], price_data["price_values"], '-')
        if price_data.get("new_price_times") is not None and len(price_data["new_price_times"]) > 0:
            ax.plot(price_data["new_price_times"], price_data["new_price_values"])

        self._price_series = price_data["price_series"]
        ax.set_xlim((self._price_series.index[0], self._price_series.index[-1]))

        rectprops = dict(facecolor='cyan', alpha=0.15)
        self._widget = matplotlib.widgets.SpanSelector(ax,
                                                       lambda f, t: self.show_delta(
                                                           mdates.num2date(f).replace(tzinfo=None),
                                                           mdates.num2date(t).replace(tzinfo=None)),
                                                       'horizontal', props=rectprops, useblit=True)

        fig.set_layout_engine('tight')
        fig.suptitle(self.get_title())

        if show:
            plt.show()
        else:
            return fig

    def show_delta(self, from_date, to_date):
        """Print price growth over a selected range on the price graph."""
        index = self._price_series.index.unique()
        timezone = index.tzinfo
        from_date = from_date.replace(tzinfo=timezone)
        to_date = to_date.replace(tzinfo=timezone)
        start_price = self._price_series.loc[index[index.get_indexer([from_date], method="nearest")]].iloc[0]
        end_price = self._price_series.loc[index[index.get_indexer([to_date], method="nearest")]].iloc[0]
        change = (end_price - start_price) / start_price
        days = (to_date - from_date).days
        if days == 0:
            print("price at %s: %.2f" % (from_date.date(), start_price))
            return
        years = days / 365.25
        yoy_change = (change + 1) ** (1 / years) - 1
        print("")
        print("during %s days (%.1f years):" % (days, years))
        print("price growth: " + "%.2f%%" % (change * 100))
        print("yearly growth: " + "%.2f%%" % (yoy_change * 100))


class Ticker(FundamentalMixin):

    def __calculate_stats(self):
        statistics = self.statistics
        use_ttm = self.reports.has_full_ttm()
        if not use_ttm:
            self.warnings.append("incomplete TTM")
        all_yearly_income_statements = self.reports.get_reports_ascending("annual", "income_statement", use_ttm)
        all_yearly_balance_sheets = self.reports.get_reports_ascending("annual", "balance_sheet", use_ttm)
        all_yearly_cash_flows = self.reports.get_reports_ascending("annual", "cash_flow", use_ttm)
        last_yearly_balance_sheet = all_yearly_balance_sheets[-1]
        last_yearly_income_statement = all_yearly_income_statements[-1]
        last_yearly_cash_flow = all_yearly_cash_flows[-1]
        statistics["TTM"] = use_ttm

        # for the tickers with an incomplete ttm, we will still take the most updated values from the quarterly reports
        last_quarterly_balance_sheet = self.reports.get_last_report("quarterly", "balance_sheet")
        last_quarterly_income_statement = self.reports.get_last_report("quarterly", "income_statement")

        annual_dates = self.reports.get_reports_dates("annual", use_ttm)
        statistics["updated at"] = annual_dates[-1]

        yahoo_info = self.yahoo_info

        # one bulk price fetch covering every date this computation will look up
        # (first report date -> today)
        self.yahoo_info.prefetch_price_range(min(annual_dates).date())

        statistics["price on update"] = self.yahoo_info.get_stock_price_at_date(annual_dates[-1].day,
                                                                                annual_dates[-1].month,
                                                                                annual_dates[-1].year)

        # calculate eps
        earnings = last_yearly_income_statement["Net Income"]
        shares_outstanding = last_yearly_income_statement["Diluted Weighted Average Shares"]  # Diluted eps
        statistics["eps"] = earnings / shares_outstanding  # assuming no preferred dividends

        # operating cash-flow per share
        operating_cash_flow = last_yearly_cash_flow["Cash Flow from Operating Activities"]
        statistics["operating_cfps"] = operating_cash_flow / shares_outstanding

        # calculate non-operating(financing and investing) cash-flow per share
        total_cash_flow = last_yearly_cash_flow["Change in Cash"]
        non_operating_cash_flow = total_cash_flow - operating_cash_flow
        statistics["non_operating_cfps"] = operating_cash_flow / shares_outstanding

        # owners earnings  - aka free cash flow
        owners_earnings = operating_cash_flow - last_yearly_cash_flow["Purchase/Sale of Prop,Plant,Equip: Net"]
        statistics["owners_earnings"] = owners_earnings / shares_outstanding

        # book value
        total_equity = last_quarterly_balance_sheet["Total Equity"]
        shares_outstanding = last_quarterly_income_statement["Diluted Weighted Average Shares"]  # Diluted
        statistics["book_value"] = total_equity / shares_outstanding
        statistics["shares (diluted)"] = shares_outstanding

        # cash & equivalents per share (for the equity-DCF cash add-back; no debt subtraction)
        statistics["cash_per_share"] = last_quarterly_balance_sheet.get("Cash and Equivalents", float('nan')) / shares_outstanding

        # dividends. Negate the "Common Stock Dividends Paid" field since it indicates lost money for the company
        # take the non-diluted number of shares since only real stocks receives dividends
        dividends = (-last_yearly_cash_flow["Common Stock Dividends Paid"]) / last_yearly_balance_sheet[
            "Ordinary Shares Outstanding"]
        statistics["dividends"] = dividends
        if np.isnan(dividends): dividends = 0

        # delta_book_value
        indicies = (-3, -2) if use_ttm else (-2, -1)  # we ignore the ttm
        yearly_total_equity = all_yearly_balance_sheets[indicies[0]]["Total Equity"]
        yearly_shares_outstanding = all_yearly_income_statements[indicies[0]]["Diluted Weighted Average Shares"]
        old_bv = yearly_total_equity / yearly_shares_outstanding
        yearly_total_equity = all_yearly_balance_sheets[indicies[1]]["Total Equity"]
        yearly_shares_outstanding = all_yearly_income_statements[indicies[1]]["Diluted Weighted Average Shares"]
        new_bv = yearly_total_equity / yearly_shares_outstanding
        delta_book_value = (new_bv - old_bv) / ((annual_dates[indicies[1]] - annual_dates[indicies[0]]).days / 365.25)

        # actual owners earnings
        statistics["actual_earnings"] = delta_book_value + dividends

        # price to book
        balance_sheet_date = last_quarterly_balance_sheet["Period End Date"]
        stock_price = yahoo_info.get_stock_price_at_date(**balance_sheet_date)
        book_value = statistics["book_value"]
        statistics["price_to_book"] = stock_price / book_value

        # pe ratio
        # NOTE: uses the last quarter price, but if without ttm, earnings of last year, true for all of our ratios
        eps = statistics["eps"]
        statistics["pe_ratio"] = stock_price / eps

        # ep ratio
        pe_ratio = statistics["pe_ratio"]
        statistics["ep_ratio[%]"] = 100 / pe_ratio

        # pe * bv
        price_to_book_value = statistics["price_to_book"]
        statistics["pe*bv"] = max(pe_ratio, 0) * price_to_book_value

        # ROE
        return_on_equity = 100 * eps / book_value
        statistics["roe[%]"] = return_on_equity

        # ROA
        return_on_assets = 100 * earnings / last_quarterly_balance_sheet["Total Assets"]
        statistics["roa[%]"] = return_on_assets

        # price_to_operating_cf_ratio
        statistics["pocf_ratio"] = stock_price / statistics["operating_cfps"]

        # current_ratio
        current_assets = last_quarterly_balance_sheet["Total Current Assets"]
        current_liabilities = last_quarterly_balance_sheet["Total Current Liabilities"]
        statistics["current_ratio"] = current_assets / current_liabilities  # todo check if need to use current debt or current liabilties

        # debt_to_equity
        total_debt = last_quarterly_balance_sheet["Current Debt"] + last_quarterly_balance_sheet["Long Term Debt"]
        statistics["debt_to_equity"] = total_debt / total_equity

        # market cap
        statistics["market_cap"] = stock_price * shares_outstanding  # take the most updated number (quarterly)

        # naive time to profit
        statistics["naive_time_to_profit"] = (stock_price - book_value) / eps if eps > 0 else np.nan

        self._calculate_trends(all_yearly_income_statements,
                               all_yearly_balance_sheets,
                               all_yearly_cash_flows,
                               annual_dates)
        self._calculate_intrinsic_values(shares_outstanding,
                                         stock_price,
                                         annual_dates[-1])
        self._calculate_quick_filter()
        self._round()

    def _calculate_trends(self, all_yearly_income_statements,
                          all_yearly_balance_sheets, all_yearly_cash_flows, annual_dates):
        """ Calculate 1st order trends from the financial reports statements.
        The trends are:
            - Net income trend
            - Equity trend
            - Operating cash flow (earned money by company's oprations)
            - Non operating cash flow (investing + financing activities)
        The function also calculates the min/max of some of these fields

        @all_yearly_income_statements   - the 'Income Statement' reports for all
                                          years
        @all_yearly_balance_sheets      - the 'Balance Sheet' reports for all years
        @all_yearly_cash_flows          - the 'Cash flow' statements for all years
        """

        statistics = self.statistics
        years = Ticker.__calculate_year_diff(annual_dates)

        # share count per year (used to compute per-share series)
        shares = np.array([s["Diluted Weighted Average Shares"] for s in all_yearly_income_statements])

        # earnings trend & growth (per-share — EPS)
        yearly_eps = np.array([s["Net Income"] for s in all_yearly_income_statements]) / shares
        poly_fit = Polynomial.fit(years, yearly_eps, deg=1)
        statistics["earnings_yearly_trend"] = poly_fit.convert().coef[1]

        try:
            poly_fit = Polynomial.fit(years, np.log(yearly_eps), deg=1)
            growth_rate = (np.exp(poly_fit.convert().coef[1]) - 1) * 100
            statistics["growth_rate"] = growth_rate
            statistics["peg_ratio"] = statistics["pe_ratio"] / growth_rate
        except RuntimeWarning as warn:
            if yearly_eps[0] > 0 and yearly_eps[-1] > 0:
                self.warnings.append("Failed to calculate log growth_rate. Growth rate fallback calculation")
                growth_rate = (yearly_eps[-1] / yearly_eps[0]) ** (1 / years[-1])
                growth_rate = (growth_rate - 1) * 100
                statistics["growth_rate"] = growth_rate
                statistics["peg_ratio"] = statistics["pe_ratio"] / growth_rate
            else:
                self.warnings.append("Failed to calculate log growth_rate. Warning: {}".format(warn))
                statistics["growth_rate"] = float('NaN')
                statistics["peg_ratio"] = float('NaN')

        # revenue trend & growth (per-share)
        yearly_revenue_ps = np.array([s["Total Revenue"] for s in all_yearly_income_statements]) / shares
        poly_fit = Polynomial.fit(years, yearly_revenue_ps, deg=1)
        statistics["revenues_yearly_trend"] = poly_fit.convert().coef[1]

        try:
            poly_fit = Polynomial.fit(years, np.log(yearly_revenue_ps), deg=1)
            revenue_growth_rate = (np.exp(poly_fit.convert().coef[1]) - 1) * 100
            statistics["revenue_growth_rate"] = revenue_growth_rate
            statistics["peg_ratio"] = statistics["pe_ratio"] / revenue_growth_rate
        except RuntimeWarning as warn:
            if yearly_revenue_ps[0] > 0 and yearly_revenue_ps[-1] > 0:
                self.warnings.append("Failed to calculate log revenue_growth_rate. Growth rate fallback calculation")
                revenue_growth_rate = (yearly_revenue_ps[-1] / yearly_revenue_ps[0]) ** (1 / years[-1])
                revenue_growth_rate = (revenue_growth_rate - 1) * 100
                statistics["revenue_growth_rate"] = revenue_growth_rate
            else:
                self.warnings.append("Failed to calculate log revenue_growth_rate. Warning: {}".format(warn))
                statistics["revenue_growth_rate"] = float('NaN')

        # equity_trend (per-share — book value per share)
        yearly_bv = np.array([s["Total Equity"] for s in all_yearly_balance_sheets]) / shares
        poly_fit = Polynomial.fit(years, yearly_bv, deg=1)
        statistics["equity_yearly_trend"] = poly_fit.convert().coef[1]

        # bv growth rate
        try:
            equity_ln = np.log(yearly_bv)  # might throw
            poly_fit = Polynomial.fit(years, equity_ln, deg=1)
            equity_ln_fit = poly_fit.convert().coef
            bv_growth_rate = (np.exp(equity_ln_fit[1]) - 1) * 100
            statistics["bv_growth_rate"] = bv_growth_rate
        except RuntimeWarning as warn:
            if yearly_bv[0] > 0 and yearly_bv[-1] > 0:
                self.warnings.append("Failed to calculate log bv_growth_rate. Growth rate fallback calculation")
                bv_growth_rate = (yearly_bv[-1] / yearly_bv[0]) ** (1 / years[-1])
                bv_growth_rate = (bv_growth_rate - 1) * 100
                statistics["bv_growth_rate"] = bv_growth_rate
            else:
                self.warnings.append("Failed to calculate log bv_growth_rate. Warning: {}".format(warn))
                statistics["bv_growth_rate"] = float('NaN')

        # operating cash flow trend (per-share)
        yearly_ocf_ps = np.array(
            [flow["Cash Flow from Operating Activities"] for flow in all_yearly_cash_flows]) / shares
        poly_fit = Polynomial.fit(years, yearly_ocf_ps, deg=1)
        statistics["operating_cf_yearly_trend"] = poly_fit.convert().coef[1]

        # minimal operating cf (per-share)
        statistics["minimal_operating_cf"] = np.min(yearly_ocf_ps)

        # free cash flow trend & growth (per-share)
        yearly_capex = np.array(
            [flow["Purchase/Sale of Prop,Plant,Equip: Net"] for flow in all_yearly_cash_flows]) / shares
        yearly_fcf_ps = yearly_ocf_ps + yearly_capex  # capex is negative
        poly_fit = Polynomial.fit(years, yearly_fcf_ps, deg=1)
        statistics["fcf_yearly_trend"] = poly_fit.convert().coef[1]
        try:
            poly_fit = Polynomial.fit(years, np.log(yearly_fcf_ps), deg=1)
            statistics["fcf_growth_rate"] = (np.exp(poly_fit.convert().coef[1]) - 1) * 100
        except (RuntimeWarning, FloatingPointError):
            self.warnings.append("Failed to calculate log fcf_growth_rate (negative FCF in series)")
            statistics["fcf_growth_rate"] = float('NaN')

        # non operating cash flow trend (per-share)
        yearly_total_cf_ps = np.array([flow["Change in Cash"] for flow in all_yearly_cash_flows]) / shares
        yearly_non_op_cf_ps = yearly_total_cf_ps - yearly_ocf_ps
        poly_fit = Polynomial.fit(years, yearly_non_op_cf_ps, deg=1)
        statistics["non_operating_cf_yearly_trend"] = poly_fit.convert().coef[1]

        # maximal non operating cf (per-share)
        statistics["maximal_non_operating_cf"] = np.max(yearly_non_op_cf_ps)

    @staticmethod
    def __calculate_free_cash_flow(cashflow_statement):
        # from online search, there are a few different ways of calculating free cash flow
        #   until farther research, I provide the course definition as owners earnings
        return cashflow_statement["Cash Flow from Operating Activities"] + \
               cashflow_statement["Purchase/Sale of Prop,Plant,Equip: Net"]

    @staticmethod
    def __calculate_year_diff(annual_dates):
        first_date = annual_dates[0]
        return [(date - first_date).days / 365.25 for date in annual_dates]

    def get_current_price(self):
        return self.yahoo_info.get_stock_price_now()

    def get_growth_rate(self, field="eps", linear=False):
        """Return pre-computed growth rate from statistics if available."""
        if not linear:
            field_map = {
                "eps": "growth_rate",
                "bv": "bv_growth_rate",
                "revenue_ps": "revenue_growth_rate",
                "free_cf_ps": "fcf_growth_rate",
            }
            stat_key = field_map.get(field)
            if stat_key and hasattr(self, 'statistics') and stat_key in self.statistics:
                return self.statistics[stat_key]
        return super().get_growth_rate(field, linear=linear)

    def _calculate_intrinsic_values(self, diluted_shares, stock_price, last_annual_date):
        statistics = self.statistics
        discount_rate = NPV_COMMON["discount_rate_percent"] / 100

        # --- basic_discount_value ---
        #   current book_value plus the summary of the discounted eps till the end of time
        #   assumes fixed eps and a pre-selected discount ratio
        book_value = statistics["book_value"]
        eps = statistics["eps"]
        statistics["basic_discount_value"] = book_value + eps * ((1 + discount_rate) / discount_rate)
        discount_value = statistics["basic_discount_value"]
        statistics["basic_discount_ratio"] = 100 * (discount_value - stock_price) / stock_price

        # --- dcf model ---
        statistics["intrinsic_value_dcf"] = self.get_intrinsic_value(discount_rate)
        statistics["irr[%]"] = self.get_irr()
        statistics["dcf_discount_ratio"] = 100 * (statistics["intrinsic_value_dcf"] - stock_price) / stock_price

        # --- capm ---
        beta = statistics.get("beta")
        if beta is None or np.isnan(beta):
            beta = calculate_beta_from_history(self.yahoo_info.yf_ticker, market_data.get_market_monthly_returns())
            if not np.isnan(beta):
                statistics["beta"] = beta
        if beta is not None and not np.isnan(beta):
            rfr = market_data.get_risk_free_rate()
            mkt = market_data.get_market_return()
            capm_rate = rfr + beta * (mkt - rfr)
            statistics["capm_interest"] = capm_rate * 100
            statistics["capm_npv"] = self.get_intrinsic_value(capm_rate)
            statistics["capm_discount_ratio"] = 100 * (statistics["capm_npv"] - stock_price) / stock_price

    def _calculate_quick_filter(self):
        """Evaluate quick filters from config. Each filter is a set of conditions
        combined with 'and' or 'or' logic."""
        import operator
        ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge,
               "<=": operator.le, "==": operator.eq, "!=": operator.ne}

        for filter_name, filter_def in QUICK_FILTERS.items():
            logic = filter_def["logic"]  # "and" or "or"
            results = []
            for stat_name, op_str, value in filter_def["conditions"]:
                stat_val = self.statistics.get(stat_name)
                if stat_val is None:
                    results.append(False if logic == "and" else False)
                    continue
                try:
                    results.append(ops[op_str](stat_val, value))
                except (TypeError, ValueError):
                    results.append(False)

            if logic == "and":
                self.statistics[filter_name] = all(results)
            else:
                self.statistics[filter_name] = any(results)


    @staticmethod
    def get_cache(symbol, market, yf_ticker=None):
        symbol = symbol.upper()
        market = market.upper()
        symbol_file_name = cache_file_name.format(symbol=symbol, market=market)
        cache_file = os.path.join(tickers_dir, symbol_file_name)

        def get_seconds_from_now(filename):
            sec_from_epoch = os.path.getmtime(filename)
            return datetime.datetime.now().timestamp() - sec_from_epoch

        # if old, ignore cache
        if not os.path.isfile(cache_file) or get_seconds_from_now(cache_file) > 3600 * 24 * 30:  # 30 days
            return Ticker(symbol, market, yf_info=yf_ticker)
        try:
            with open(cache_file, 'rb') as file:
                return pickle.load(file).post_pickle(yf_ticker=yf_ticker)
        except FileNotFoundError:
            return Ticker(symbol, market, yf_info=yf_ticker)


    def pre_pickle(self, short_term):
        self.reports.pre_pickle(short_term)
        self.yahoo_info.pre_pickle(short_term)

    def post_pickle(self, yf_ticker=None, price_now=None):
        self.yahoo_info.post_pickle(yf_ticker, price_now)
        self.reports.post_pickle(self.yahoo_info.yf_ticker)
        self.dcf_model = load_dcf_model(self.symbol, self.market)
        return self

    def save_cache(self):
        try:
            symbol_file_name = cache_file_name.format(symbol=self.symbol, market=self.market)
            cache_file = os.path.join(tickers_dir, symbol_file_name)
            os.makedirs(tickers_dir, exist_ok=True)
            with open(cache_file, 'wb') as file:
                yf_ticker = self.yahoo_info.yf_ticker
                price_now = self.yahoo_info._price_now
                self.pre_pickle(short_term=False)
                pickle.dump(self, file)
                self.post_pickle(yf_ticker, price_now)
                
            
        except TypeError:
            print("Ticker.py: warnning: failed to save cache, probably yf_ticker object")

    def __str__(self):
        result = "Ticker of %s:%s\n{\n" % (self.symbol, self.market)
        result += "Statistics:\n%s,\n" % pformat(self.statistics)


    def __init__(self, symbol, market, *, yf_info = None):

        self.symbol = symbol.upper()
        self.market = market.upper()

        self.yahoo_info = YahooInfo(self.symbol, self.market, yf_info = yf_info)

        # This would throw an exception if it fails
        self.reports = YReports(symbol, market, self.yahoo_info.yf_ticker)
        #self.reports = Reports(self.symbol, self.market)

        # allow __calculate_stats to log warning in this file
        self.warnings = list()
        self.dcf_model = load_dcf_model(self.symbol, self.market)

        self.statistics = {
            # the order here is the order in the csv
            "name": self.yahoo_info.info.get("shortName"),

            "price_to_book": None,
            "pe_ratio": None,
            "ep_ratio[%]": None,
            "pe*bv": None,
            "roe[%]": None,
            "roa[%]": None,
            "peg_ratio": None,
            "operating_cfps": None,
            "non_operating_cfps": None,
            "pocf_ratio": None,
            "basic_discount_value": None,
            "basic_discount_ratio": None,
            "intrinsic_value_dcf": None,
            "dcf_discount_ratio": None,
            "irr[%]": None,
            "current_ratio": None,
            "debt_to_equity": None,
            "market_cap": None,
            "naive_time_to_profit": None,  # in years
            "minimal_operating_cf": None,
            "maximal_non_operating_cf": None,
            "earnings_yearly_trend": None,
            "equity_yearly_trend": None,
            "operating_cf_yearly_trend": None,
            "non_operating_cf_yearly_trend": None,
            "revenues_yearly_trend": None,
            "growth_rate": None,
            "bv_growth_rate": None,
            "revenue_growth_rate": None,
            "fcf_growth_rate": None,
            "fcf_yearly_trend": None,
            "dividends": None,
            "owners_earnings": None,
            "actual_earnings": None,
            "shares (diluted)": None,

            "net_income": self.reports.get_last_report("annual", "income_statement")["Net Income"],
            "healthy": None,
            "overvalued": None,
            "leveraged": None,
            "sector": self.yahoo_info.info.get("sector"),
            "industry": self.yahoo_info.info.get("industry"),
            "beta": self.yahoo_info.info.get("beta"),  # code calculates locally if this field is missing
            "capm_interest": None,
            "capm_npv": None,
            "capm_discount_ratio": None,
            "price on update": None,
            "eps": None,
            "book_value": None,
            "updated at": None,
            "TTM": None,
            "dcf_model_date": self.dcf_model.get("saved_at") if self.dcf_model else None
        }

        try:
            self.__calculate_stats()
        except Exception as err:
            line = get_exception_line()
            raise StatisticsException(str(err) + " in line: " + str(line))

        self.save_cache()

    def get_forecasted_annual_growth(self):
        forcast = self.statistics[forcast_growth_field]/100  # assume field is annual and in percents
        forcast = forcast if forcast and forcast > 0 else 0
        forcast = min(forcast, 1)
        return forcast

    # Plotting:

    def get_price_graph(self, term, add_ttm=False):
        dates = self.reports.get_reports_dates(term, add_ttm)
        start_date = dates[0]
        end_date = dates[-1]
        date_vector, price_vector = self.yahoo_info.get_stock_price_in_range(start_date, end_date)
        return date_vector, price_vector

    def get_price_graph_after_report(self, term, add_ttm=False):
        dates = self.reports.get_reports_dates(term, add_ttm)
        start_date = dates[-1]
        end_date = datetime.datetime.now()
        date_vector, price_vector = self.yahoo_info.get_stock_price_in_range(start_date, end_date)
        return date_vector, price_vector

    def get_price_at_report_dates(self, term, add_ttm=False):
        reports_ordered = self.reports.get_reports_ascending(term, 'balance_sheet', add_ttm)
        dates = [report["Period End Date"] for report in reports_ordered]
        # one bulk fetch covering all report dates -> today, so each date below
        # is served from memory instead of its own windowed network request
        if dates:
            earliest = min(datetime.date(d["year"], d["month"], d["day"]) for d in dates)
            self.yahoo_info.prefetch_price_range(earliest)
        prices = [self.yahoo_info.get_stock_price_at_date(date["day"], date["month"], date["year"]) for date in dates]
        return prices

    def get_name(self):
        return self.symbol

    def _get_plot_data(self):
        """Extract all time series needed for plot_me."""
        if hasattr(self, '_plot_data_cache'):
            return self._plot_data_cache
        add_ttm = True
        times = self.reports.get_reports_dates("annual", add_ttm=add_ttm)

        equity = np.array(self.reports.get_field_as_list("balance_sheet", "annual", "Total Equity", add_ttm=add_ttm))
        shares = np.array(self.reports.get_field_as_list("income_statement", "annual",
                                                          "Diluted Weighted Average Shares", add_ttm=add_ttm))
        bv = equity / shares

        try:
            cash = np.array(self.reports.get_field_as_list("balance_sheet", "annual", "Cash and Equivalents", add_ttm=add_ttm))
            cash_ps = cash / shares
        except Exception:
            cash_ps = np.full_like(bv, np.nan)

        earnings = np.array(self.reports.get_field_as_list("income_statement", "annual", "Net Income", add_ttm=add_ttm))
        eps = earnings / shares

        try:
            revenue = np.array(self.reports.get_field_as_list("income_statement", "annual", "Revenue", add_ttm=add_ttm))
            revenue_ps = revenue / shares
        except Exception:
            revenue_ps = np.full_like(eps, np.nan)

        operating = np.array(self.reports.get_field_as_list("cash_flow", "annual",
                                                             "Cash Flow from Operating Activities", add_ttm=add_ttm))
        capex = np.array(self.reports.get_field_as_list("cash_flow", "annual",
                                                         "Purchase/Sale of Prop,Plant,Equip: Net", add_ttm=add_ttm))
        free_cf = operating + capex

        # per-share versions for portfolio aggregation
        operating_ps = operating / shares
        free_cf_ps = free_cf / shares

        prices = np.array(self.get_price_at_report_dates('annual', add_ttm=add_ttm))

        self._plot_data_cache = {
            "times": times,
            "bv": bv,
            "cash_ps": cash_ps,
            "eps": eps,
            "revenue_ps": revenue_ps,
            "operating_cf": operating,
            "free_cf": free_cf,
            "operating_cf_ps": operating_ps,
            "free_cf_ps": free_cf_ps,
            "prices": prices,
        }
        return self._plot_data_cache

    def _get_plot_price_data(self):
        """Daily price history for the bottom price graph. Cached separately."""
        if hasattr(self, '_plot_price_data_cache'):
            return self._plot_price_data_cache
        has_ttm = self.reports.has_full_ttm()
        price_times, price_values = self.get_price_graph('annual', add_ttm=has_ttm)
        new_price_times, new_price_values = self.get_price_graph_after_report('annual', add_ttm=has_ttm)
        self._plot_price_data_cache = {
            "price_times": price_times,
            "price_values": price_values,
            "new_price_times": new_price_times,
            "new_price_values": new_price_values,
            "price_series": pd.concat([price_values, new_price_values]),
        }
        return self._plot_price_data_cache

    def get_current_pe(self):
        """
        Since the pe ratio in the statistics was calculated for the time of the last yearly report
        this function calculate the pe ratio according to the current price as a quick replacement for the websites
        and not as a screening parameter

        right now we generate multiple versions of this ratio, for experimenting with them
        e.g.:   Ticker('MSFT', 'NASDAQ').get_current_pe()
        """

        # calculate the diluted eps per quarter:
        last_quarterly_income_statement = self.reports.get_last_report("quarterly", "income_statement")
        last_quarterly_balance_sheet = self.reports.get_last_report("quarterly", "balance_sheet")
        shares_outstanding = last_quarterly_income_statement["Diluted Weighted Average Shares"]
        earnings = 4 * last_quarterly_income_statement["Net Income"]
        quarterly_eps = earnings / shares_outstanding

        yearly_eps = self.statistics["eps"]

        # find the prices, now, at the quarter & at the year:

        # Quarter:
        balance_sheet_date = last_quarterly_balance_sheet["Period End Date"]
        quarterly_price = self.yahoo_info.get_stock_price_at_date(**balance_sheet_date)

        # Year:
        last_yearly_balance_sheet = self.reports.get_last_report("annual", "balance_sheet")
        balance_sheet_date = last_yearly_balance_sheet["Period End Date"]
        yearly_price = self.yahoo_info.get_stock_price_at_date(**balance_sheet_date)

        # Now:
        real_price = self.yahoo_info.get_stock_price_now()

        # and finally, the pe ratios:
        yearly_pe_ratio = real_price / yearly_eps
        quarterly_pe_ratio = real_price / quarterly_eps
        old_yearly_pe_ratio = yearly_price / yearly_eps
        old_quarterly_pe_ratio = quarterly_price / quarterly_eps

        print("yearly_pe_ratio:        " + str(yearly_pe_ratio))
        print("quarterly_pe_ratio:     " + str(quarterly_pe_ratio))
        print("")
        print("old_yearly_pe_ratio:    " + str(old_yearly_pe_ratio))
        print("old_quarterly_pe_ratio: " + str(old_quarterly_pe_ratio))

    def _round(self, digits=2):
        for key, value in self.statistics.items():
            if np.issubdtype(type(value), np.floating):
                self.statistics[key] = np.round(value, digits)


def format_axis(ax):
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # # round to nearest years.
    # datemin = np.datetime64(data['date'][0], 'Y')
    # datemax = np.datetime64(data['date'][-1], 'Y') + np.timedelta64(1, 'Y')
    # ax.set_xlim(datemin, datemax)
    #
    # # rotates and right aligns the x labels, and moves the bottom of the
    # # axes up to make room for them
    # fig.autofmt_xdate()


""" --- Portfolios: --- """
class TickerGroup(YahooGroup):
    def __init__(self, symbols:list, markets:list, *,
                 risk_free_rate=None, existing_tickers:dict = dict(), forecast_policy):
        symbols = [s.upper() for s in symbols]
        markets = [m.upper() for m in markets]
        super().__init__(symbols, markets)
        self.risk_free_rate = risk_free_rate if risk_free_rate else market_data.get_risk_free_rate()
        self.market_std = market_data.get_market_std()
        self.portfolio_std = np.nan
        self.tickers_dictionary = existing_tickers  # dict[(symbol,market)] will hold the ticker, will be used for get_forcasted_monthly_growth(), otherwise use past growth
        if forecast_policy not in FORECAST_POLICIES:
            raise ValueError(f"Unknown forecast_policy: {forecast_policy}")
        self.forecast_policy = forecast_policy
        self.annual_growth_forecasts = list()
        self.beta_dictionary = {}
        self.efficient_frontier = None

        # set in find_tangency_portfolio:  TODO: group together and remove duplication with min var
        self.tangency_portfolio = None
        self.return_tangent = np.nan
        self.std_tangent = np.nan
        self.beta_tangent = np.nan

        # set in find_min_variance_portfolio:
        self.min_var_portfolio = None
        self.return_min_var = np.nan
        self.std_min_var = np.nan
        self.beta_min_var = np.nan

    def to_df(self) -> pd.DataFrame:
        tickers = list(self.tickers_dictionary.values())
        if not tickers:
            return pd.DataFrame()
        columns = list(tickers[0].statistics.keys())
        d = {f"{t.symbol}:{t.market}": [t.statistics.get(k) for k in columns] for t in tickers}
        return pd.DataFrame.from_dict(d, orient='index', columns=columns)

    def calculate_correlation(self):
        # Note: we call get_monthly_prices() directly rather than super().calculate_correlation()
        # So cov is computed only over the final valid set
        self.get_monthly_prices()
        self.calculate_growth_forecast()
        self.build_beta_dictionary()
        if len(self.valid_full_symbols) < 2:
            print(f"Warning: only {len(self.valid_full_symbols)} valid ticker(s) after forecast policy '{self.forecast_policy}'; skipping efficient frontier")
            self.efficient_frontier = None
            return
        # ef:
        self.get_cov()
        self.create_frontier()
        self.find_tangency_portfolio()
        self.find_min_variance_portfolio()

    def calculate_growth_forecast(self):
        """Compute the per-ticker expected annual return used by the efficient frontier,
        according to self.forecast_policy. Tickers are always (re)created so their
        statistics are computed/cached; the policy only affects the EF expected-return
        input (and thus valid_full_symbols membership).

        Policy decomposition:
          - "past": always past performance.
          - "irr_*": use the DCF/IRR forecast; when unavailable, either fall back to
            past growth ("*_past_*") or drop the ticker from the frontier ("*_filter_*").
          - unavailable = IRR is NaN ("*_if_nan"), or (no saved DCF model OR IRR NaN)
            for the "*_if_no_model" variants.
        Non-stocks (indices/ETFs) always use past growth.
        """
        print("recreating tickers and calculating growth")  # todo optimize runtime
        policy = self.forecast_policy
        use_irr = policy != "past"
        is_filter = policy in ("irr_filter_if_nan", "irr_filter_if_no_model")
        require_model = policy in ("irr_past_if_no_model", "irr_filter_if_no_model")

        for symbol, market, full_symbol in zip(self.symbols, self.markets, self.full_symbols):
            # For indices/ETFs, always use past growth and skip ticker creation
            if not is_stock(self.yf_ticker.tickers[full_symbol]):
                self.annual_growth_forecasts.append(self.get_past_annual_performance(symbol, market))
                continue
            
            # Create ticker if not already in dictionary
            if (symbol, market) not in self.tickers_dictionary:
                try:
                    self.tickers_dictionary[(symbol, market)] = Ticker.get_cache(symbol, market, yf_ticker=self.yf_ticker.tickers[full_symbol])
                except Exception as e:
                    print(f"Warning: failed to create ticker for {symbol}:{market}: {e}")

            ticker = self.tickers_dictionary.get((symbol, market))

            if not use_irr:  # "past"
                self.annual_growth_forecasts.append(self.get_past_annual_performance(symbol, market))
                continue

            # IRR-based policies -----------------------------------------------
            raw_irr = ticker.statistics.get("irr[%]") if ticker else None
            has_model = bool(ticker and getattr(ticker, "dcf_model", None))
            irr_nan = raw_irr is None or (isinstance(raw_irr, float) and np.isnan(raw_irr))
            unavailable = irr_nan or (require_model and not has_model)

            if not unavailable:
                growth = ticker.get_forecasted_annual_growth()
            elif is_filter:
                growth = float('nan')  # dropped from the frontier below
            else:  # past-fallback variants
                growth = self.get_past_annual_performance(symbol, market)

            self.annual_growth_forecasts.append(growth)

        # Remove symbols with NaN forecasts from valid set
        self.valid_full_symbols = [f for f, g in zip(self.full_symbols, self.annual_growth_forecasts)
                                   if f in self.valid_full_symbols and not np.isnan(g)]

    def build_beta_dictionary(self):
        """Build a dictionary mapping full_symbol to beta value for all symbols."""
        self.beta_dictionary = {}
        monthly = self.get_monthly_growths()
        market_returns = market_data.get_market_monthly_returns()
        
        for symbol, market, full_symbol in zip(self.symbols, self.markets, self.full_symbols):
            beta = np.nan
            if (symbol, market) in self.tickers_dictionary:
                beta = self.tickers_dictionary[(symbol, market)].statistics.get("beta")
            else:
                try:
                    beta = self.yf_ticker.tickers[full_symbol].info.get("beta")
                except Exception as e:
                    print(f"{full_symbol}: error getting beta ({e})")
            
            # Fallback: calculate from already-fetched portfolio returns (preferred),
            # or from individual ticker history via the shared helper
            if beta is None or (isinstance(beta, float) and np.isnan(beta)):
                if full_symbol in monthly.columns:
                    beta = calculate_beta(monthly[full_symbol], market_returns)
                else:
                    yf_single = self.yf_ticker.tickers[full_symbol]
                    beta = calculate_beta_from_history(yf_single, market_returns)
                if np.isnan(beta):
                    print(f"{full_symbol}: could not calculate beta")
            
            self.beta_dictionary[full_symbol] = beta if beta is not None else np.nan

    def create_frontier(self):
        print("EF")
        named_growth = pd.Series(data=self.annual_growth_forecasts, index=self.full_symbols)
        named_growth = named_growth[self.valid_full_symbols].fillna(0)
        self.efficient_frontier = EfficientFrontier(named_growth, self.cov, verbose=False, solver="ECOS")  # todo: understand why this solver works when the default failed

    def _solve_portfolio(self, ef_method, **kwargs):
        """Solve an efficient frontier optimization and return (weights, return, std, beta)."""
        if self.efficient_frontier is None:
            raise ValueError("no efficient frontier (too few valid tickers)")
        ef_copy = self.efficient_frontier.deepcopy()
        weights = ef_method(ef_copy, **kwargs)
        ret, std, _ = ef_copy.portfolio_performance(risk_free_rate=self.risk_free_rate)
        w_arr = np.array([weights.get(f, 0) for f in self.full_symbols])
        b_arr = np.array([self.beta_dictionary.get(f, np.nan) for f in self.full_symbols])
        valid = ~np.isnan(b_arr) & (w_arr > 0)
        beta = w_arr[valid] @ b_arr[valid] / w_arr[valid].sum() if valid.any() else np.nan
        return weights, ret, std, beta

    def _print_weights(self, title, weights, ret, std, beta):
        """Print portfolio weights to terminal."""
        print(f"\n{title}:")
        print(f"  Return: {ret*100:.1f}%  Std: {std*100:.1f}%  Beta: {beta:.2f}")
        for symbol, weight in sorted(weights.items(), key=lambda x: -x[1]):
            if weight > 0.001:
                print(f"  {symbol}: {weight*100:.1f}%")

    def find_tangency_portfolio(self):
        """Calculate the optimal (tangency) portfolio using max Sharpe ratio"""
        try:
            self.tangency_portfolio, self.return_tangent, self.std_tangent, self.beta_tangent = \
                self._solve_portfolio(lambda ef: ef.max_sharpe(risk_free_rate=self.risk_free_rate))
            self._print_weights("Optimal Portfolio (Max Sharpe)", self.tangency_portfolio,
                                self.return_tangent, self.std_tangent, self.beta_tangent)
        except Exception as e:
            print(f"Warning: Could not calculate optimal portfolio: {e}")

    def find_min_variance_portfolio(self):
        """Calculate the minimum variance portfolio"""
        try:
            self.min_var_portfolio, self.return_min_var, self.std_min_var, self.beta_min_var = \
                self._solve_portfolio(lambda ef: ef.min_volatility())
            self._print_weights("Min Variance Portfolio", self.min_var_portfolio,
                                self.return_min_var, self.std_min_var, self.beta_min_var)
        except Exception as e:
            print(f"Warning: Could not calculate min variance portfolio: {e}")

    def plot_frontier(self, ax=None):
        if not ax:
            _, ax = plt.subplots()
        if self.efficient_frontier is None:
            ax.text(0.5, 0.5, "No efficient frontier\n(too few valid tickers for this forecast policy)",
                    ha='center', va='center', transform=ax.transAxes)
            return ax
        plotting.plot_efficient_frontier(self.efficient_frontier.deepcopy(), ax=ax, ef_param="return", show_assets=True, show_tickers=True)
        
        # Normalize X axis by market std
        # Use FuncFormatter to relabel the x-axis ticks
        from matplotlib.ticker import FuncFormatter
        
        def format_func(value, tick_number):
            return f'{value / self.market_std:.2f}'
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.set_xlabel("Volatility (std)")
        
        # Add risk-free rate reference point
        rfr = self.risk_free_rate
        # Plot at x=0 (zero volatility) with normalized x-axis if applicable
        ax.plot(0, rfr, 'go', markersize=10, label=f'Risk-Free Rate ({rfr*100:.1f}%)', zorder=5)

        # Add market index reference point
        mkt_return = market_data.get_market_return()
        # Plot at market std (x=market_std, y=market_return)
        if hasattr(self, 'market_std') and self.market_std > 0:
            ax.plot(self.market_std, mkt_return, 'go', markersize=10, label=f'Market (S&P 500: {mkt_return*100:.1f}%)', zorder=5)

        # Add optimal portfolio and CAL
        self.plot_optimal_and_cal(ax, rfr)
    
        ax.set_title("Efficient Frontier")
        ax.legend(fontsize=8, markerscale=0.6)
        ax.get_figure().set_layout_engine('tight')
        return ax

    def plot_optimal_and_cal(self, ax, rfr):
        """Plot optimal portfolio, min variance portfolio, and Capital Allocation Line on the efficient frontier"""
        ax.plot(self.std_tangent, self.return_tangent, 'mo', markersize=10, 
               label=f'Optimal ({self.return_tangent*100:.1f}%)', zorder=5)
        
        # Draw Capital Allocation Line (CAL) from risk-free rate to optimal portfolio
        ax.plot([0, self.std_tangent], [rfr, self.return_tangent], 'g--', 
               linewidth=1, alpha=0.7, label='Capital Allocation Line', zorder=4)

        # Min variance portfolio
        if not np.isnan(self.std_min_var):
            ax.plot(self.std_min_var, self.return_min_var, 'co', markersize=10,
                   label=f'Min Variance ({self.return_min_var*100:.1f}%)', zorder=5)

    def plot_capm(self, ax=None):
        """Plot CAPM graph: beta (x-axis) vs growth (y-axis)"""
        if not ax:
            _, ax = plt.subplots()
        
        betas = []
        growths = []
        labels = []
        
        for symbol, market, growth, full_symbol in zip(self.symbols, self.markets, self.annual_growth_forecasts, self.full_symbols):
            beta = self.beta_dictionary.get(full_symbol, np.nan)
            
            if not np.isnan(beta) and not np.isnan(growth):
                betas.append(beta)
                growths.append(growth * 100)  # convert to percentage
                labels.append(symbol)
        
        if betas:
            ax.scatter(betas, growths, alpha=0.6, s=100, label='Individual Assets')
            for i, label in enumerate(labels):
                ax.annotate(label, (betas[i], growths[i]), fontsize=9, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points')
            ax.set_xlabel("Beta")
            ax.set_ylabel("Expected Return (%)")
            ax.set_title("CAPM")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=1, color='r', linestyle='--', linewidth=0.5, alpha=0.5, label='Market Beta=1')
            
            # Add risk-free rate reference
            rfr = market_data.get_risk_free_rate() * 100
            ax.plot(0, rfr, 'go', markersize=12, label=f'Risk-Free Rate ({rfr:.1f}%)', zorder=4)
            
            # Add market index reference
            mkt_return = market_data.get_market_return() * 100
            ax.plot(1, mkt_return, 'go', markersize=12, label=f'Market (S&P 500: {mkt_return:.1f}%)', zorder=4)
            
            # Draw Security Market Line (SML) from risk-free rate to market
            ax.plot([0, 1], [rfr, mkt_return], 'g--', 
                   linewidth=1, alpha=0.7, label='Security Market Line', zorder=3)

            # Tangency portfolio
            if hasattr(self, 'beta_tangent') and not np.isnan(self.beta_tangent):
                ax.plot(self.beta_tangent, self.return_tangent * 100, 'mo', markersize=10,
                       label='Optimal', zorder=5)

            # Min variance portfolio
            if hasattr(self, 'beta_min_var') and not np.isnan(self.beta_min_var):
                ax.plot(self.beta_min_var, self.return_min_var * 100, 'co', markersize=10,
                       label='Min Variance', zorder=5)
        else:
            ax.text(0.5, 0.5, 'No beta data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        return ax


if __name__ == '__main__':
    ticker_name = input("Ticker Name: ")
    stock_exchange = input("Stock Exchange: ")

    Ticker.get_cache(ticker_name, stock_exchange).plot_me()