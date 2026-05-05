#!/usr/bin/python3

import yfinance as yf
from yfinance_info import get_ticker_from_standard_symbols
from reports import BaseReport
import numpy as np
import pandas as pd

report_yahoo2msn = {
    "balance_sheet": {
        #"???" : "Period End Date",  -- todo set manually from the column header
        "Current Assets" : "Total Current Assets",
        "Total Assets" : "Total Assets",
        "Current Liabilities" : "Total Current Liabilities",
        "Total Debt" : "Total Liabilities",
        "Current Debt" : "Current Debt",  # not all stocks have
        "Long Term Debt" : "Long Term Debt",
        "Common Stock Equity" : "Total Equity",
        "Goodwill And Other Intangible Assets" : "Goodwill and Other Intangible Assets",
        "Ordinary Shares Number" : "Ordinary Shares Outstanding",
        #"???" : "Currency Code"
    },
    "income_statement": {
        #"???" : "Period End Date",
        "Net Income" : "Net Income",
        "Total Revenue" : "Total Revenue",
        "Diluted Average Shares" : "Diluted Weighted Average Shares"
        # todo add profit margins
    },
    "cash_flow": {
        #"???" : "Period End Date",
        "Operating Cash Flow" : "Cash Flow from Operating Activities",
        "Changes In Cash" : "Change in Cash",
        "Cash Dividends Paid" : "Common Stock Dividends Paid",  # not really the same but let's assume no preffered stocks
        "Capital Expenditure" : "Purchase/Sale of Prop,Plant,Equip: Net"  # Capital Expenditures
    }
}


report_name_convert = {
    ("balance_sheet"   , "annual"   ): "balance_sheet",
    ("income_statement", "annual"   ): "income_stmt",
    ("cash_flow"       , "annual"   ): "cash_flow",
    ("balance_sheet"   , "quarterly"): "quarterly_balance_sheet",
    ("income_statement", "quarterly"): "quarterly_income_stmt",
    ("cash_flow"       , "quarterly"): "quarterly_cash_flow",
}

class YReports(BaseReport):
    """ a temporary wraper class to yfinance to emulate the old msn reports class """
    def __init__(self, symbol:str, market:str, yf_ticker:yf.Ticker = None):
        super().__init__(symbol, market)
        self.full_symbol, self.market_endian = get_ticker_from_standard_symbols(symbol, market)
        self.yf_ticker = None
        self.post_pickle(yf_ticker=yf_ticker)
        self.parse_and_save_reports()
        self.finish_init()

    def pre_pickle(self):
        self.yf_ticker = None

    def post_pickle(self, yf_ticker = None):
        if yf_ticker is not None:
            self.yf_ticker = yf_ticker
        else:
            from fx_converter import YfTickerUSD
            self.yf_ticker = YfTickerUSD(self.full_symbol)

    def parse_and_save_reports(self):
        # todo finish
        for (report_name, term), yf_ticker_field in report_name_convert.items():
            self.parse_report(report_name, term, getattr(self.yf_ticker, yf_ticker_field))
        # todo save?
        # ...

    def parse_report(self, report_name, term, report_table):
        report_dict = getattr(self, report_name)
        term_dict = dict()
        report_dict[term] = term_dict
        report_table = report_table[report_table.keys()[:4]]  # sometime a 5th year is added but its full of NaN (but a single cell)
        periods = report_table.keys()
        for period in periods:
            period_dict = dict()
            term_dict[period] = period_dict
            period_dict["Period End Date"] = {'year': period.year, 'month': period.month, 'day': period.day}
            period_column = report_table[period]
            for yahoo_name, name in report_yahoo2msn[report_name].items():
                period_dict[name] = period_column.get(yahoo_name, np.nan)



