#!/usr/bin/env python3

from matplotlib import pyplot as plt
from fx_converter import fx_converter
from yfinance_info import YahooInfo
from ticker import TickerGroup, search_growth, Ticker, PORTFOLIO_CONFIG
from gui.forecast_policy_dialog import ask_forecast_policy
from portfolio import Portfolio, PortfolioGui
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sys import stdout
import sys

"""
Calculate your average portfolio's growth, and compare it to an index
*   supports indices inside the table (with yahoo's unintuitive names)
*   does not include currency conversion/taxes/etc...
*   file can have blank lines & commented-out ones

Table Format (tab separated):
Ticker	Market	Date	Amount	Action	Cost
"""

sep  = "\t"
comment = "#"
frmt = "%Y-%m-%d"  # T%H:%M:%S"

# Convert Cost to base currency using FX rate at transaction date
market_to_currency = {
    "NASDAQ": "USD", "NYSE": "USD", "AMEX": "USD",
    "TLV": "ILA",  # Israeli stocks trade in agora
    "TSE": "CAD",
    "LON": "GBX",  # London trades in pence
    "ASX": "AUD",
    "TPE": "TWD", "TYO": "JPY", "SWX": "CHF",
    "AMS": "EUR", "STO": "SEK", "KRX": "KRW", "SHE": "CNY",
}

def read_tsv(file):
    import io
    with open(file) as f:
        lines = [l for l in f if not l.lstrip().startswith('#') and l.strip()]
    table = pd.read_csv(io.StringIO("".join(lines)), sep=sep)
    if "Cost" in table.columns and "Market" in table.columns and "Date" in table.columns:
        # Market -> price currency (what the cost is denominated in)
        dates = pd.to_datetime(table["Date"], format=frmt)
        for i, row in table.iterrows():
            currency = market_to_currency.get(row["Market"], "USD")
            if currency != fx_converter.base_currency:
                rate = fx_converter.get_rate_at(currency, dates.iloc[i])
                table.at[i, "Cost"] = row["Cost"] * rate

    return table


def get_buy_amount(df):
    keys = df.index if isinstance(df, pd.Series) else df.columns
    if "Action" in keys:
        action = df["Action"]
        if isinstance(action, str):
            buy_sign = {"BUY": 1, "SELL": -1, "TRACK": 0}[action.upper()]
        else:
            buy_sign = action.str.upper().map({"BUY": 1, "SELL": -1, "TRACK": 0})
    else:
        buy_sign = 1
    buy_amount = buy_sign * (df["Amount"] if "Amount" in keys else 0)
    return buy_amount


def create_portfolio(table, forecast_policy="past"):
    # remove repetitions:
    symbols = []
    markets = []
    ids = []
    amounts  = []
    for _, row in table.iterrows():
        symbol = row["Ticker"]
        market = row["Market"]
        buy_amount = get_buy_amount(row)
        if (symbol, market) in ids:
            old_index = ids.index((symbol, market))
            amounts[old_index] += buy_amount
        else:
            symbols.append(symbol)
            markets.append(market)
            ids.append((symbol, market))
            amounts.append(buy_amount)
    return Portfolio(symbols, markets, amounts, forecast_policy=forecast_policy)


def get_get_npv(table):  # will return the lambda function
    # time which had passed for each transaction
    duration = (pd.Timestamp.now().normalize() - pd.to_datetime(table["Date"], format=frmt)) / np.timedelta64(365, "D")
    # current price
    market_value = np.array([YahooInfo(row[1]["Ticker"], row[1]["Market"]).get_stock_price_now() for row in table.iterrows()])
    # buy\sell sign
    buy_amount = get_buy_amount(table)
    # calculate once, use in every get_npv
    portfolio_value = (market_value * buy_amount).sum()
    money_invested = buy_amount * table["Cost"]
    # net_present_value of each transaction including opportunity cost
    get_npv = lambda rate: portfolio_value - (money_invested * ((1 + rate) ** duration)).sum()
    # more stats
    profitable = (portfolio_value - money_invested.sum()) >= 0  # will be used to limit the search range
    return profitable, portfolio_value, money_invested.sum(), get_npv


def get_get_future_npv(table):
    pass  # todo will call all ticker.get_calc_npv()


def get_performance(table, verbose=True):
    # todo check if can be assumed to be monotonic, I think not
    profitable, portfolio_value, money_invested, get_npv = get_get_npv(table)
    if verbose: print("Money invested: %.2f -> %.2f" % (money_invested, portfolio_value))
    return search_growth(
        npv_function=get_npv,
        price=0,
        min_growth=0 if profitable else -1,
        max_growth=4 if profitable else 0,
        delta_growth=0.1 / 100
    )
    

def performance_per_ticker(table, portfolio:Portfolio) -> pd.DataFrame:
    # each ticker need: portfolio_weight, growth
    Histories = {name[0]:group for name, group in table.groupby(["Ticker"])}
    Performance = dict()
    for name, history in Histories.items():
        market = history.iloc[0]["Market"]
#        ticker = portfolio.tickers_dictionary[(name, market)]
        Performance[name] = dict()
        Performance[name]["Annualized Price Growth"] = get_performance(history, verbose=False)
        Performance[name]["Weight[%]"] = portfolio.get_weight(name, market)
#        # more data at a glance:
#        Performance[name]["pe_ratio"]    = ticker.statistics["pe_ratio"]
#        Performance[name]["healthy"]     = ticker.statistics["healthy"]
#        Performance[name]["overvalued"]  = ticker.statistics["overvalued"]
    return pd.DataFrame.from_dict(Performance, orient="index")


def get_index(table):
    """ a similiar table for investing the money in an index instead """
    index_name = "NASDAQ-100"  # informative name
    index_market = "NASDAQ"
    index_yahoo_name = "^IXIC".replace("^", "%5E")

    idx = YahooInfo(index_yahoo_name, index_market)
    table = table.copy()
    dates = pd.to_datetime(table["Date"], format=frmt)
    # one bulk fetch covering all transaction dates -> now
    idx.prefetch_price_range(dates.min().date())
    idx_prices = [idx.get_stock_price_at_date(date.day, date.month, date.year) for date in dates]
    adjusted_amount = table["Amount"] * table["Cost"] / idx_prices
    table["Cost"] = idx_prices
    table["Amount"] = adjusted_amount
    table["Ticker"] = index_yahoo_name
    table["Market"] = index_market
    return index_name, table


class HistoricPortfolioGui(PortfolioGui):
    def __init__(self, portfolio, perf_df, portfolio_irr, money_invested, portfolio_value,
                 index_name, index_irr, index_value, show_frontier=False):
        super().__init__(portfolio, show_frontier)
        summary = (
            "Portfolio:  {:.2f} -> {:.2f}\n"
            "            {:.2f}% / yr\n\n"
            "{}:  {:.2f} -> {:.2f}\n"
            "            {:.2f}% / yr"
        ).format(money_invested, portfolio_value, portfolio_irr,
                 index_name, money_invested, index_value, index_irr)
        
        # Merge performance data with portfolio table
        portfolio_table = portfolio.get_portfolio_table_df()
        merged_df = perf_df[["Annualized Price Growth"]].join(portfolio_table, how='left')
        merged_df = merged_df[["Weight (%)", "Annualized Price Growth", "Portfolio Beta"]]
        self.set_summary(summary, merged_df)

# Future estimate:



def main():
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from qt_material import apply_stylesheet
    import os
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_red.xml')


    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
    file, _ = QFileDialog.getOpenFileName(
        None, "Select Portfolio TSV", default_dir, "TSV Files (*.tsv);;All Files (*)"
    )
    if not file:
        sys.exit(0)

    forecast_policy = ask_forecast_policy(PORTFOLIO_CONFIG.get("forecast_policy", "past"))

    run_portfolio_optimization = True

    table = read_tsv(file)
    portfolio = create_portfolio(table, forecast_policy=forecast_policy)

    do_historic_analysis = "Date" in table.columns
    if do_historic_analysis:
        portfolio_irr = get_performance(table)
        _, portfolio_value, money_invested, _ = get_get_npv(table)
        index_name, index_table = get_index(table)
        index_irr = get_performance(index_table)
        _, index_value, _, _ = get_get_npv(index_table)
        perf_df = performance_per_ticker(table, portfolio)

    if run_portfolio_optimization:
        portfolio.calculate_correlation()

    if do_historic_analysis:
        gui = HistoricPortfolioGui(portfolio, perf_df, portfolio_irr, money_invested, portfolio_value,
                       index_name, index_irr, index_value, run_portfolio_optimization)
    else:
        gui = PortfolioGui(portfolio, run_portfolio_optimization)
    
    portfolio_name = os.path.splitext(os.path.basename(file))[0]
    gui.setWindowTitle(f"Portfolio Analyzer - {portfolio_name}")
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



