#!/usr/bin/env python3

from ticker import Ticker, StatisticsException, TickerGroup, NPV_ASSUMPTIONS, TLDR_FIELDS
from yfinance_info import YfinanceException
from reports import MsnReportsException
from PyQt5.QtWidgets import QApplication, QFileDialog
import pandas as pd
import warnings
import sys
from time import sleep
from os.path import basename, join
import os

OUTPUT_DIR = "outputs"

import multiprocessing as mp

# to have colored output
from colorama import Fore, Back, Style

# make warning throw exceptions only inside worker processes (not globally)
# this is important when using Threads to avoid having writing to
# stderr in parallel. The warnings would need to be caught explicitly so
# that they wouldn't be missed


class tickerWorkerStatus():
    STATUS_SUCCESS = 0
    STATUS_WARNINGS = 1
    STATUS_FAILED = 2

    LONGEST_SYM_MAR_STR_LEN = 30

    def __init__(self, symbol, market):
        self.symbol = symbol
        self.market = market
        self.ticker = None
        self.status = self.STATUS_SUCCESS
        self.msg = None

    def setWarning(self, msg):
        self.status = self.STATUS_WARNINGS
        self.msg = msg

    def setFailure(self, msg):
        self.status = self.STATUS_FAILED
        self.msg = msg

    def setTicker(self, ticker):
        self.ticker = ticker

    def getTicker(self):
        return self.ticker

    def isFailed(self):
        return self.status == self.STATUS_FAILED

    def __str__(self):
        if self.status == self.STATUS_SUCCESS:
            status_msg = Fore.GREEN + "SUCCESS" + Fore.RESET
        elif self.status == self.STATUS_WARNINGS:
            status_msg = Fore.LIGHTMAGENTA_EX + "WARNING" + Fore.RESET + " ({})".format(self.msg)
        else:
            status_msg = Fore.RED + "FAILURE" + Fore.RESET + " ({})".format(self.msg)

        symbol_market_str = "{}:{}".format(self.symbol, self.market)
        padded_symbol_market_str = symbol_market_str.ljust(self.LONGEST_SYM_MAR_STR_LEN)
        return Fore.BLUE + padded_symbol_market_str + Fore.RESET + "-\t\t" + status_msg

# TODO: replace all silly prints with proper log functions that indicate an
#   error
def create_ticker_worker(ticker_queue_tuple):
    warnings.simplefilter('error')  # scoped to this worker process only

    # the multiproccesing will hide any uncatched exception, so wrapping the whole function in a general try statement
    try:
        symbol, market = ticker_queue_tuple["ticker_tuple"]
        status_queue = ticker_queue_tuple["queue"]

        status = tickerWorkerStatus(symbol, market)

        try:
            # print("Fetching data for {symbol}:{market}\n".format(symbol = symbol,
                # market = market))
            ticker = Ticker.get_cache(symbol, market)

            if ticker.warnings:
                status.setWarning("Ticker {}:{} has warnings: {}".format(symbol, market, ", ".join(ticker.warnings)))

            status.setTicker(ticker)
        except YfinanceException as err:
            status.setFailure("Failed in yfinance. error: {}".format(err))
        except MsnReportsException as err:
            status.setFailure("Failed in reports. error: {}".format(err))
        except StatisticsException as err:
            status.setFailure("Failed in statistics. error: {}".format(err))
        except Exception as err:
            status.setFailure("Failed in an unknown location. error: {}".format(err))

        #todo if inner fields are needed, call post_pickle at the other side
        if status.ticker:
            status.ticker.pre_pickle(short_term=True)

        try:
            status_queue.put(status)
        except Exception as err:
            print(f"Inserting the ticker {symbol}:{market} object into the queue failed.")
            print("This probably means that some of its objects cannot be serialized")
            print("")
    except Exception as err:
        print(Fore.RED + "ERROR: " + Fore.RESET + str(err))


def create_tickers_from_symbol_names(symbol_list):
    """ Get a list of tuples that contain symbol ticker name and its
        stock exchange. Return a list of class Ticker that represent each of the
        stocks"""
    # pre-fetch market data in the main process so forked workers inherit the cache
    from ticker import market_data
    market_data.get_risk_free_rate()
    market_data.get_market_return()

    symbols_nr = len(symbol_list)
    symbol_ix   = 0
    LONGEST_PROGRESS_STRING = len("99.99% [{} / {}]".format(symbols_nr, symbols_nr)) +\
                              len(Fore.CYAN) * 6

    manager = mp.Manager()
    queue   = manager.Queue(len(symbol_list))
    ticker_queue_tuple = [ { "ticker_tuple" : ticker, "queue" : queue }
            for ticker in symbol_list ]

    tickers_list = list()
    with mp.Pool(processes=None) as pool:
        result = pool.map_async(create_ticker_worker, ticker_queue_tuple)
        while not result.ready():
            while not queue.empty():
                status_item = queue.get()
                if not status_item.isFailed():
                    tickers_list.append(status_item.getTicker())

                symbol_ix = symbol_ix + 1
                percent = round((symbol_ix / symbols_nr) * 100, 2)

                # create the progress string (multiline to add it color)
                progress = Fore.CYAN + f"{percent}%" + Fore.RESET
                progress = progress + " [" + Fore.MAGENTA + str(symbol_ix) + Fore.RESET
                progress = progress + " / " + Fore.MAGENTA + str(symbols_nr) + Fore.RESET
                progress = progress + "]"

                progress = progress.ljust(LONGEST_PROGRESS_STRING)

                print(progress, status_item)

    return tickers_list


def sort_stocks_by_fields(tickers_list, fields):
    """Sort stocks in @tickers_list by the @fields. The stocks
       are sorted by the first item in @fields_name, then by the second and so
       forth. @fields is a list containing field name and True/False value. True
       would sort the value in ascending order which False would do the opposite"""

    def get_ticker_fields(ticker):
        return tuple(ticker.statistics[field[0]] * (field[1] - 0.5) for field in fields)

    return sorted(tickers_list, key = get_ticker_fields)


def filter_stocks_by_fields(ticker, fields):
    """Check stock @ticker by the elements in @fields which have the
    format [ field, value, less_than ]. If field element is a string, then
    the field == value is checked. Otherwise, field <= value or field >= value
    is checked depending on less_than value.
    All checks are AND'd"""

    passes=True
    for check in fields:
        fname, fvalue = check[0:2]
        field = ticker.statistics[fname]

        if type(field) == type("string"):
            passes &= field == fvalue
        else:
            less_than = check[2] if len(check) == 3 else True
            if less_than:
                passes &= field <= fvalue
            else:
                passes &= field >= fvalue
    return passes


def extract_statistics(ticker):
    return ticker.statistics

def ticker_list_to_df(tickers_list : list) -> pd.DataFrame:
    d = {f"{ticker.symbol}:{ticker.market}" : extract_statistics(ticker).values() for ticker in tickers_list}
    df = pd.DataFrame.from_dict(d, orient='index', columns=tickers_list[0].statistics.keys())
    return df

# This would output a csv file containing the statistics for each of the tickers
def stocks_list_to_csv(tickers_list, out_path, show_fields=None, ignore_fields=None):
    if len(tickers_list) == 0:
        print("No Tickers To Save. ignored.")
        return
    df = ticker_list_to_df(tickers_list)

    if show_fields is not None:
        df = df[show_fields]
    if ignore_fields is not None:
        df = df.drop(ignore_fields)

    # save to file
    try:
        df.to_csv(out_path)
    except PermissionError:
        print("CSV file is opened.\nPlease close the Excel app...\nThis is the last chance")
        sleep(6)
        df.to_csv(out_path)
    return df


def create_tickers_from_file(file_path : str) -> list:
    """The function receives a path to a file containing entries of the form
    'TICKER MARKET' (e.g. 'AAPL NASDAQ') and returns a Ticker list"""

    symbol_list = []
    with open(file_path, "r") as f:
        print("Querying file " + file_path)
        for line in f:
            if line.startswith("#") or len(line) <= 1:  # allow comments
                continue
            line_attr = line.split()
            line_attr = (" ".join(line_attr[:-1]), line_attr[-1])  # allow spaces in ticker name
            symbol_list.append(line_attr)

    return create_tickers_from_symbol_names(symbol_list)


def group_tickers(tickers:list, filter_non_healthy:bool) -> TickerGroup:
    """ group the tickers as TickerGroup for portfolio optimization """
    # filter out tickers without positive price growth prediction
    symbols = list()
    markets = list()
    existing_tickers = dict()
    for ticker in tickers:  # take only successful ones, to avoid error handling
        if (0 < ticker.statistics["irr[%]"] < 95) and (not filter_non_healthy or ticker.statistics["healthy"] == True):
            symbols.append(ticker.symbol)
            markets.append(ticker.market)
            existing_tickers[(ticker.symbol, ticker.market)] = ticker
    
    from gui.forecast_policy_dialog import ask_forecast_policy
    forecast_policy = ask_forecast_policy()
    ticker_group = TickerGroup(symbols, markets, existing_tickers=existing_tickers,
                               forecast_policy=forecast_policy)
    return ticker_group


tldr_statistics = TLDR_FIELDS if TLDR_FIELDS else [  # a less overwhelming set of statistics
            "pe_ratio",
            "price_to_book",
            "peg_ratio",
            "irr[%]",
            "dcf_discount_ratio",
            "current_ratio",
            "growth_rate",
            "bv_growth_rate",
            "healthy",
            "overvalued",
            "leveraged",
        ]

def select_stocks_file(parent_widget=None) -> str :
    my_stocks_file, _ = QFileDialog.getOpenFileName(
        parent_widget,
        "Select Stocks File",
        "inputs",
        "Followed Stocks (*.txt)"
    )
    if not my_stocks_file:
        raise Exception("NoInputFile") # todo better raise

    return my_stocks_file

def main():
    app = QApplication(sys.argv)

    # 1) Create 'Ticker' variable for every symbol
    # 2) store result in some list
    # 3) Do something with the Ticker's list
    # 4) Save in a csv file
    # 5) optionally, run portfolio correlation analysis
    run_optimize = False
    filter_non_healthy = True
    risk_free_rate = NPV_ASSUMPTIONS["portfolio_risk_free_rate_percent"] / 100
    my_stocks_file = select_stocks_file()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = ".".join(basename(my_stocks_file).split(".")[:-1])
    csv_path  = join(OUTPUT_DIR, stem + "_statistics.csv")
    tldr_path = join(OUTPUT_DIR, stem + "_statistics_tldr.csv")
    try:  # alert the user while still have time
        with open(csv_path, 'a+'):
            pass
        with open(tldr_path, 'a+'):
            pass
    except PermissionError:
        print("Close Excel!")
        sleep(3)

    tickers = create_tickers_from_file(my_stocks_file)

    df = stocks_list_to_csv(tickers, csv_path)
    stocks_list_to_csv(tickers, tldr_path, show_fields=tldr_statistics)

    # optimizer:
    if run_optimize:
        import matplotlib.pyplot as plt
        print("starting our optimizer")
        portfolio_optimizer = group_tickers(tickers, filter_non_healthy)
        portfolio_optimizer.risk_free_rate = risk_free_rate
        print("plotting")
        portfolio_optimizer.plot_frontier()
        plt.show()



if __name__ == '__main__':
    main()
