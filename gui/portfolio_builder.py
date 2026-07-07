#!/usr/bin/env python3
"""
Dialog for building a new portfolio from a list of tickers.
Used by both PortfolioGui and tickers_gui.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QCheckBox, QPushButton, QHeaderView, QLineEdit, QMessageBox, QLabel,
    QRadioButton, QButtonGroup, QWidget
)
from PyQt5.QtCore import Qt


# Weighting modes
MODE_MANUAL = "manual"
MODE_EQUAL = "equal"
MODE_MARKET_CAP = "market_cap"


def _format_market_cap(mc: float) -> str:
    """Format a market cap as e.g. '1.23T', '456.7B', '12.3M'."""
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if mc >= scale:
            return f"{mc / scale:.2f}{suffix}"
    return f"{mc:.0f}"


class PortfolioBuilderDialog(QDialog):
    """
    Popup dialog that lets the user select tickers and optionally enter amounts,
    then creates and opens a new PortfolioGui.

    Three weighting modes (gated by the "Include amounts" checkbox):
      - Manual:     user types per-ticker share quantities
      - Equal:      1/N over the checked tickers, given a portfolio dollar value
      - Market-cap: weights proportional to market cap, given a portfolio dollar value

    Parameters
    ----------
    ticker_data : list of (symbol, market, price)
        Available tickers to choose from.
    existing_tickers : dict, optional
        Dict of (symbol, market) -> Ticker objects to reuse.
    forecast_policy : str
        Passed through to Portfolio constructor (EF expected-return source).
    parent : QWidget, optional
    """

    COL_CHECK = 0
    COL_SYMBOL = 1
    COL_MARKET = 2
    COL_PRICE = 3
    COL_AMOUNT = 4
    COL_MARKET_CAP = 5

    def __init__(self, ticker_data, existing_tickers=None, forecast_policy=None, amounts=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Portfolio")
        self.setMinimumSize(500, 400)
        self._ticker_data = ticker_data
        self._existing_tickers = existing_tickers or {}
        self._forecast_policy = forecast_policy
        self._portfolio_windows = []
        has_amounts = amounts is not None

        layout = QVBoxLayout(self)

        # "Include amounts" checkbox — gates the radios + value box
        self._amounts_cb = QCheckBox("Include amounts")
        self._amounts_cb.stateChanged.connect(self._refresh_mode_ui)
        layout.addWidget(self._amounts_cb)

        # Mode radios (Manual / Equal / Market-cap)
        radio_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._radio_manual = QRadioButton("Manual")
        self._radio_equal = QRadioButton("Equal-weighted")
        self._radio_mktcap = QRadioButton("Market-cap weighted")
        self._radio_manual.setChecked(True)
        for btn in (self._radio_manual, self._radio_equal, self._radio_mktcap):
            self._mode_group.addButton(btn)
            radio_row.addWidget(btn)
            btn.toggled.connect(self._refresh_mode_ui)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        # Portfolio-value row (only visible in Equal/Market-cap modes)
        self._value_row = QWidget()
        value_layout = QHBoxLayout(self._value_row)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.addWidget(QLabel("Portfolio value:"))
        self._value_edit = QLineEdit()
        self._value_edit.setPlaceholderText("e.g. 100000")
        value_layout.addWidget(self._value_edit)
        value_layout.addStretch()
        layout.addWidget(self._value_row)

        # Table
        self._table = QTableWidget(len(ticker_data), 6)
        self._table.setHorizontalHeaderLabels(["", "Symbol", "Market", "Price", "Amount", "Market Cap"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(self.COL_CHECK, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)

        self._checkboxes = []
        self._amount_edits = []
        self._market_cap_items = []  # QTableWidgetItem per row, populated lazily
        self._market_caps_loaded = False

        for row, (symbol, market, price) in enumerate(ticker_data):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            self._checkboxes.append(cb)
            self._table.setCellWidget(row, self.COL_CHECK, cb)

            # Symbol
            item = QTableWidgetItem(symbol)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, self.COL_SYMBOL, item)

            # Market
            item = QTableWidgetItem(market)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, self.COL_MARKET, item)

            # Price
            price_str = f"{price:.2f}" if price and price > 0 else "N/A"
            item = QTableWidgetItem(price_str)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, self.COL_PRICE, item)

            # Amount — prefill if provided
            amount_str = str(int(amounts[row]) if amounts and amounts[row] == int(amounts[row]) else amounts[row]) if amounts else "1"
            edit = QLineEdit(amount_str)
            self._amount_edits.append(edit)
            self._table.setCellWidget(row, self.COL_AMOUNT, edit)

            # Market Cap — placeholder, populated lazily on first switch to Market-cap mode
            mc_item = QTableWidgetItem("")
            mc_item.setFlags(mc_item.flags() & ~Qt.ItemIsEditable)
            self._market_cap_items.append(mc_item)
            self._table.setItem(row, self.COL_MARKET_CAP, mc_item)

        # If amounts were provided, activate the checkbox and prefill the
        # portfolio-value box from sum(amounts * prices).
        if has_amounts:
            self._amounts_cb.setChecked(True)
            total = sum((amounts[r] or 0) * (price or 0)
                        for r, (_, _, price) in enumerate(ticker_data))
            if total > 0:
                self._value_edit.setText(f"{total:.2f}")
        layout.addWidget(self._table)

        self._refresh_mode_ui()

        # Open button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._open_btn = QPushButton("Open")
        self._open_btn.clicked.connect(self._on_open)
        btn_layout.addWidget(self._open_btn)
        layout.addLayout(btn_layout)

    def _current_mode(self) -> str:
        if self._radio_equal.isChecked():
            return MODE_EQUAL
        if self._radio_mktcap.isChecked():
            return MODE_MARKET_CAP
        return MODE_MANUAL

    def _refresh_mode_ui(self):
        """Update widget enabled/visible state based on the master checkbox + mode."""
        on = self._amounts_cb.isChecked()
        for btn in (self._radio_manual, self._radio_equal, self._radio_mktcap):
            btn.setEnabled(on)
        mode = self._current_mode() if on else None

        manual_active = on and mode == MODE_MANUAL
        self._table.setColumnHidden(self.COL_AMOUNT, not manual_active)
        for edit in self._amount_edits:
            edit.setEnabled(manual_active)

        # Market Cap column: visible in Market-cap mode (load values on first show)
        mktcap_active = on and mode == MODE_MARKET_CAP
        self._table.setColumnHidden(self.COL_MARKET_CAP, not mktcap_active)
        if mktcap_active and not self._market_caps_loaded:
            self._load_market_caps()

        self._value_row.setVisible(on and mode in (MODE_EQUAL, MODE_MARKET_CAP))

    def _load_market_caps(self):
        """Fetch market caps for all rows and populate the Market Cap column.
        Cached for the lifetime of the dialog — values stored on each
        QTableWidgetItem via Qt.UserRole and read back at allocation time."""
        from yfinance_info import YahooGroup

        symbols = [s for s, _, _ in self._ticker_data]
        markets = [m for _, m, _ in self._ticker_data]
        try:
            group = YahooGroup(symbols, markets)
            caps = group.get_market_caps(self._existing_tickers)
            for row, mc in enumerate(caps):
                item = self._market_cap_items[row]
                if mc:
                    item.setText(_format_market_cap(mc))
                    item.setData(Qt.UserRole, mc)
                else:
                    item.setText("N/A")
                    item.setData(Qt.UserRole, None)
        except Exception as e:
            print(f"Warning: failed to load market caps: {e}")
        self._market_caps_loaded = True

    def _on_open(self):
        master_on = self._amounts_cb.isChecked()
        mode = self._current_mode() if master_on else MODE_MANUAL

        # Collect selected tickers (and amounts in Manual mode)
        selected = []
        for row, (symbol, market, price) in enumerate(self._ticker_data):
            if not self._checkboxes[row].isChecked():
                continue

            if master_on and mode == MODE_MANUAL:
                text = self._amount_edits[row].text().strip()
                try:
                    amount = float(text)
                    if amount <= 0:
                        raise ValueError
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input",
                                        f"Invalid amount for {symbol}:{market} — enter a positive number.")
                    return
            else:
                amount = 0

            selected.append((symbol, market, price, amount))

        if len(selected) < 2:
            QMessageBox.warning(self, "Not Enough Tickers",
                                "Select at least 2 tickers to build a portfolio.")
            return

        symbols = [s for s, m, p, a in selected]
        markets = [m for s, m, p, a in selected]
        prices = [p for s, m, p, a in selected]
        amounts = [a for s, m, p, a in selected]

        if mode == MODE_MANUAL:
            self._build_and_show(symbols, markets, amounts, title="Custom Portfolio")
        else:
            self._open_auto(symbols, markets, prices, mode)

    def _open_auto(self, symbols, markets, prices, mode):
        """Equal / market-cap weighted portfolios.
        Uses allocate_portfolio (computes weights + DiscreteAllocation) on data
        already loaded into the table — no additional yfinance calls."""
        try:
            total_value = float(self._value_edit.text().strip())
            if total_value <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Enter a positive portfolio value.")
            return

        # For market-cap mode we need the cap values. They're loaded into the
        # column when the user switches modes; load them now if missing.
        if mode == MODE_MARKET_CAP and not self._market_caps_loaded:
            self._load_market_caps()

        # Pull market caps from the table for the *checked* rows. Drop rows
        # without a usable cap (only relevant in market-cap mode).
        market_caps = []
        if mode == MODE_MARKET_CAP:
            kept_symbols, kept_markets, kept_prices = [], [], []
            for s, m, p in zip(symbols, markets, prices):
                row = next(r for r, (rs, rm, _) in enumerate(self._ticker_data)
                           if rs == s and rm == m)
                mc = self._market_cap_items[row].data(Qt.UserRole)
                if mc:
                    kept_symbols.append(s)
                    kept_markets.append(m)
                    kept_prices.append(p)
                    market_caps.append(mc)
                else:
                    print(f"Warning: skipping {s}:{m} — no market cap")
            symbols, markets, prices = kept_symbols, kept_markets, kept_prices

        if len(symbols) < 2:
            QMessageBox.warning(self, "Not Enough Tickers",
                                "Need at least 2 tickers with valid data.")
            return

        try:
            from portfolio import allocate_portfolio
            result = allocate_portfolio(mode, symbols, markets, prices,
                                        market_caps if market_caps else None,
                                        total_value)
            if result is None:
                QMessageBox.warning(self, "Allocation Failed",
                                    "Could not allocate shares.")
                return
            out_symbols, out_markets, out_quantities = result
            title = {MODE_EQUAL: "Equal-Weighted Portfolio",
                     MODE_MARKET_CAP: "Market-Cap Weighted Portfolio"}[mode]
            self._build_and_show(out_symbols, out_markets, out_quantities, title=title)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create portfolio:\n{e}")
            import traceback
            traceback.print_exc()

    def _build_and_show(self, symbols, markets, amounts, title):
        """Instantiate Portfolio with the given quantities and open a PortfolioGui."""
        selected_keys = set(zip(symbols, markets))
        filtered_tickers = {k: v for k, v in self._existing_tickers.items() if k in selected_keys}

        # Inherit the originating portfolio's policy if given, else the session's cached choice
        if self._forecast_policy is None:
            from gui.forecast_policy_dialog import get_forecast_policy
            self._forecast_policy = get_forecast_policy()

        try:
            from portfolio import Portfolio, PortfolioGui
            portfolio = Portfolio(
                symbols, markets, amounts,
                existing_tickers=filtered_tickers,
                forecast_policy=self._forecast_policy
            )
            portfolio.calculate_correlation()

            gui = PortfolioGui(portfolio, show_frontier=True)
            gui.setWindowTitle(title)
            self._portfolio_windows.append(gui)
            gui.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create portfolio:\n{e}")
            import traceback
            traceback.print_exc()
