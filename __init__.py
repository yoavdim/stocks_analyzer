"""
Stock portfolio analyzer package.

Provides tools for stock analysis, portfolio optimization,
financial report fetching, and GUI-based visualization.
"""

# Data fetching layer
from .yfinance_info import YahooInfo, YahooGroup, YfinanceException
from .reports import Reports, BaseReport, MsnReportsException
from .yahoo_reports import YReports

# Core analysis
from .ticker import Ticker, TickerGroup, MarketDataCache, market_data, search_growth, StatisticsException
from .portfolio import Portfolio, HistoricPortfolio
from .bonds import calc_yield_to_maturity

# Batch analysis
from .stocks_analyzer import create_tickers_from_file, create_tickers_from_symbol_names, group_tickers
