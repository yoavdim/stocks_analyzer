#!/usr/bin/env python3

import requests
from htmldom import htmldom
import time
import re
import datetime
from pprint import pformat


class MsnReportsException(Exception):
    """Exceptions that are thrown from the Reports class. The class is
    responsible for fetching the financial data from MSN. This includes mostly
    financial reports."""
    pass


site_format_init = "https://www.msn.com/en-us/money/stockdetailsvnext/financials"
site_format_dict = {
    "NAS": site_format_init + "/{report_name}/{term}/fi-126.1.{symbol}.{market}",
    "NYS": site_format_init + "/{report_name}/{term}/fi-126.1.{symbol}.{market}",
    "ASE": site_format_init + "/{report_name}/{term}/fi-126.1.{symbol}.{market}",
    "TAI": site_format_init + "/{report_name}/{term}/fi-144.1.{symbol}.{market}",
    "TKS": site_format_init + "/{report_name}/{term}/fi-133.1.{symbol}.{market}",
    "LON": site_format_init + "/{report_name}/{term}/fi-151.1.{symbol}.{market}",
    "SWX": site_format_init + "/{report_name}/{term}/fi-182.1.{symbol}.{market}",
    "AMS": site_format_init + "/{report_name}/{term}/fi-202.1.{symbol}.{market}",
    "STO": site_format_init + "/{report_name}/{term}/fi-170.1.{symbol}.{market}",
    "TAE": site_format_init + "/{report_name}/{term}/fi-292.1.IS-{symbol}.{market}.{symbol}",
    "KRX": site_format_init + "/{report_name}/{term}/fi-141.1.A{symbol}.{market}.{symbol}",
    "SHE": site_format_init + "/{report_name}/{term}/fi-137.1.{symbol}.{market}",
}
site_for_ticker_with_dot = lambda site: site.format(symbol="{tempered_symbol}.{market}", market="{symbol}",
                                                    report_name="{report_name}", term="{term}")  # will work only on standart countries
report_dir = "./msn_reports"
file_format = "{symbol}-{market}-{report_name}-{term}.html"
cache_file_name = "{symbol}-cached-reports.json"

market_to_msn_market = {
    "NASDAQ": "NAS",
    "NYSE": "NYS",
    "AMEX": "ASE",  # bought by nyse
    "TPE": "TAI",  # Taiwan
    "TYO": "TKS",  # Japan
    "LON": "LON",  # UK
    "SWX": "SWX",  # Switzerland
    "AMS": "AMS",  # Holland
    "STO": "STO",  # Sweden
    "TLV": "TAE",  # Israel
    "KRX": "KRX",  # Korea
    "SHE": "SHE",  # Shenzen
}

num_of_fields = {
    "annual": 4,
    "quarterly": 4
}
fields = {
    "balance_sheet": [
        "Period End Date",
        "Total Current Assets",
        "Total Assets",
        "Total Current Liabilities",
        "Total Liabilities",
        "Current Debt",
        "Long Term Debt",
        "Total Equity",
        "Cash and Equivalents",
        "Goodwill and Other Intangible Assets",
        "Ordinary Shares Outstanding",
       # "Currency Code"
    ],
    "income_statement": [
        "Period End Date",
        "Net Income",
        "Total Revenue",
        "Diluted Weighted Average Shares"
    ],
    "cash_flow": [
        "Period End Date",
        "Cash Flow from Operating Activities",
        # "Cash Flow from Investing Activities",  # Warnning! in msft this field is '-', but the graph is still viewable
        # "Cash Flow from Financing Activities",
        "Change in Cash",  # Im using this field minus the operating as a more stable replacement
        # to the sum of investing + financing
        "Common Stock Dividends Paid",
        "Purchase/Sale of Prop,Plant,Equip: Net"  # Capital Expenditures
    ]
}

# when calculating TTM, those fields should take the last quarter value, and not the sum of the 4
non_additive_fields = [
    *fields["balance_sheet"],  # includes "Period End Date" for all of the reports
    "Diluted Weighted Average Shares"
]

brk_a2b_ratio = 1500


def store_process_value(term_dict, key, str_value):
    """Receive a value parsed from the html of a form, and store
        its value in the dictionary in the correct type"""

    if key == "Period End Date":
        m = re.match(r"(?P<month>\d+)/(?P<day>\d+)/(?P<year>\d+)", str_value)
        term_dict[key] = {key: int(value) for key, value in m.groupdict().items()}
    elif key in ("Currency Code",):
        term_dict[key] = str_value
    elif str_value == "-":
        term_dict[key] = float('NaN')
    else:
        value = float(str_value.replace(',', ''))
        value = value * 10 ** 6
        term_dict[key] = value


def get_number_of_fields(document, searched_field):
    """ Get an htmlDom object and count how many instances of a tag exist in it
    @document: the document to search in
    @searched_field: the field to search
    """

    num = 0
    while document.find(searched_field)[num + 1]:
        num += 1

    return num


class BaseReport:
    def __init__(self, symbol, market):
        self.symbol = symbol
        self.market = market

        self.balance_sheet = dict()
        self.income_statement = dict()
        self.cash_flow = dict()
        """
        self.sheet[annual/quartely][quarter_name][field_name] == float_number; 
        """

        self.__cached_ttm = dict()

        # self.parse_and_save_reports()

    def finish_init(self):
        self._ffill_diluted_shares()
        self.get_ttm("balance_sheet")
        self.get_ttm("income_statement")
        self.get_ttm("cash_flow")

    def pre_pickle(self, short_term):
        pass  # no live state to drop

    def post_pickle(self, *args, **kwargs):
        pass  # subclasses restore live refs (e.g. yf_ticker)

    def _ffill_diluted_shares(self):
        """Forward-fill 'Diluted Weighted Average Shares' across quarterly income statements."""
        import numpy as np
        quarterly = self.income_statement.get("quarterly", {})
        if not quarterly:
            return

        last_valid = None
        for key in sorted(quarterly.keys()):
            report = quarterly[key]
            shares = report.get("Diluted Weighted Average Shares")
            if shares is not None and not np.isnan(shares):
                last_valid = shares
            elif last_valid is not None:
                report["Diluted Weighted Average Shares"] = last_valid
                print(f"Warning: {self.symbol} quarterly diluted shares missing, forward-filled")

    def parse_and_save_reports(self):
        """ fill the dictionaries """
        raise Exception("Unimplemented method, override in base class and call in __init__")

    def get_reports_ascending(self, term, report_name, add_ttm=False):
        report_dict = getattr(self, report_name)
        term_dict = report_dict[term]

        ordered_terms = sorted(term_dict.keys())
        ordered_reports = [term_dict[t] for t in ordered_terms]
        if term == "annual" and add_ttm:
            ordered_reports.append(self.get_ttm(report_name))
        return ordered_reports

    def get_last_report(self, term, report_name):
        ordered_reports = self.get_reports_ascending(term, report_name)
        try:
            return ordered_reports[-1]
        except:
            print("Ticker {}:{}. Parameters: {}, {}".format(self.symbol, self.market,
                term, report_name))
            raise

    def get_reports_dates(self, term, add_ttm=False):
        # it doesnt really matter if we take the dates from a balance_sheet or income_statement:
        reports_ordered = self.get_reports_ascending(term, 'balance_sheet', add_ttm)
        dates = [report["Period End Date"] for report in reports_ordered]
        dates = [datetime.datetime(date["year"], date["month"], date["day"]) for date in dates]
        return dates

    def get_field_as_list(self, report, term, field, add_ttm=False):
        return [r[field] for r in self.get_reports_ascending(term, report, add_ttm)]

    def has_full_ttm(self) -> bool:
        quarters = self.balance_sheet['quarterly'].keys()
        return len(quarters) >= 4

    def get_ttm(self, report: str) -> dict:
        """ get the trailing twelve months of data (or less if quarters are missing in the reports) """
        if hasattr(self.__cached_ttm, report):
            return self.__cached_ttm[report]

        result = dict()
        reports = self.get_reports_ascending("quarterly", report)
        num_of_quarters = len(reports)
        if num_of_quarters > 4:
            reports = reports[-4:]
            num_of_quarters = 4
        if num_of_quarters < 4:
            print("unreliable TTM for %s, missing quarters" % self.symbol)

        for field in fields[report]:
            if field in non_additive_fields:
                result[field] = reports[-1][field]
            else:
                # if we are missing reports, we still want the numbers to be annual-like
                result[field] = sum([r[field] for r in reports]) * 4 / num_of_quarters

        self.__cached_ttm[report] = result
        return result

    def __str__(self):
        result = "{\n"
        result += "balance_sheet:\n%s,\n" % pformat(self.balance_sheet, indent=4)
        result += "income_statement:\n%s,\n" % pformat(self.income_statement, indent=4)
        result += "cash_flow:\n%s,\n" % pformat(self.cash_flow, indent=4)
        result += "}"
        return result





# TODO: catch specific exceptions and not just assume what they are
class Reports(BaseReport):

    def __parse_fields(self, term, report_name, response_text):
        """ parse all fields defined in self.fields and insert them into a
        a dictionary """
        document = htmldom.HtmlDom()
        document.createDom(response_text)

        report_dict = getattr(self, report_name)
        report_dict[term] = dict()
        term_dict = report_dict[term]
        report_fields = fields[report_name]

        # year or quarter columns
        periods_number = get_number_of_fields(document, "div.column-heading")

        for i in range(periods_number):
            columns = document.find("div.column-heading")[i + 1]
            quarter_name = columns.find("p").attr("title")

            # initialize the quarter column
            term_dict[quarter_name] = dict()
            for key in report_fields:
                words = key.split()
                selector = "".join("[title~={}]".format(word) for word in words)
                for ul in document.find("ul").has(selector):
                    p = ul.find(selector)
                    if p.attr("title") == key:
                        try:
                            str_value = ul.find("li")[i + 1].find('p').attr("title")
                            store_process_value(term_dict[quarter_name], key, str_value)

                            # small fix for brk.b:
                            if self.symbol == "BRK.B":
                                if key in ["Ordinary Shares Outstanding", "Diluted Weighted Average Shares"]:
                                    term_dict[quarter_name][key] = term_dict[quarter_name][key] * brk_a2b_ratio
                                # divide dividends?

                        except IndexError:
                            print("Failed to fetch field %s of stock %s. skipping" % (key, self.symbol))
                            store_process_value(term_dict[quarter_name], key, '-')

    def __fetch_url(self, site_url):
        response = requests.request("GET", site_url)
        time.sleep(0.5)

        if len(response.text) < 70:
            raise MsnReportsException("MSN Server Error: url {}".format(site_url))
        return response.text

    def __parse_and_save_report(self, term, report_name):
        # fixes msn url quirks
        if "." in self.symbol or " " in self.symbol:
            site_symbol = self.symbol.replace(".", "%7CSLA%7C").replace(" ", "%7CSLA%7C")
            site_format = site_for_ticker_with_dot(site_format_dict[self.msn_market])
            site_url = site_format.format(report_name=report_name, term=term, symbol=self.symbol,
                                                       market=self.msn_market, tempered_symbol=site_symbol)
        else:
            site_url = site_format_dict[self.msn_market].format(report_name=report_name, term=term, symbol=self.symbol,
                                                                market=self.msn_market)
        try:
            response_text = self.__fetch_url(site_url)
        except:
            raise MsnReportsException(
                "Failed to fetch site symbol: {} market: {} msn market: {}".format(self.symbol, self.market,
                                                                                   self.msn_market))
        self.__parse_fields(term, report_name, response_text)

    def parse_and_save_reports(self):  # overrides base
        self.__parse_and_save_report("quarterly", "balance_sheet")
        self.__parse_and_save_report("quarterly", "income_statement")
        self.__parse_and_save_report("quarterly", "cash_flow")

        self.__parse_and_save_report("annual", "balance_sheet")
        self.__parse_and_save_report("annual", "income_statement")
        self.__parse_and_save_report("annual", "cash_flow")

    def __init__(self, symbol, market):
        super().__init__(symbol, market)
        try:
            self.msn_market = market_to_msn_market[market]
        except:
            raise MsnReportsException("market {} is not supported for symbol {}".format(market, symbol))
        self.parse_and_save_reports()
        self.finish_init()

