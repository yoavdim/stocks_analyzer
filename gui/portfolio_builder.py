#!/usr/bin/env python3
"""
Dialog for building a new portfolio from a list of tickers.
Used by both PortfolioGui and tickers_gui.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QCheckBox, QPushButton, QHeaderView, QLineEdit, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt


class PortfolioBuilderDialog(QDialog):
    """
    Popup dialog that lets the user select tickers and optionally enter amounts,
    then creates and opens a new PortfolioGui.

    Parameters
    ----------
    ticker_data : list of (symbol, market, price)
        Available tickers to choose from.
    existing_tickers : dict, optional
        Dict of (symbol, market) -> Ticker objects to reuse.
    use_past_growth : bool
        Passed through to Portfolio constructor.
    parent : QWidget, optional
    """

    COL_CHECK = 0
    COL_SYMBOL = 1
    COL_MARKET = 2
    COL_PRICE = 3
    COL_AMOUNT = 4

    def __init__(self, ticker_data, existing_tickers=None, use_past_growth=True, amounts=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Portfolio")
        self.setMinimumSize(500, 400)
        self._ticker_data = ticker_data
        self._existing_tickers = existing_tickers or {}
        self._use_past_growth = use_past_growth
        self._portfolio_windows = []
        has_amounts = amounts is not None

        layout = QVBoxLayout(self)

        # "Include amounts" checkbox
        self._amounts_cb = QCheckBox("Include amounts")
        self._amounts_cb.stateChanged.connect(self._toggle_amounts)
        layout.addWidget(self._amounts_cb)

        # Table
        self._table = QTableWidget(len(ticker_data), 5)
        self._table.setHorizontalHeaderLabels(["", "Symbol", "Market", "Price", "Amount"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(self.COL_CHECK, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)

        self._checkboxes = []
        self._amount_edits = []

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
            edit.setEnabled(has_amounts)
            self._amount_edits.append(edit)
            self._table.setCellWidget(row, self.COL_AMOUNT, edit)

        # If amounts were provided, activate the column by default
        if has_amounts:
            self._amounts_cb.setChecked(True)
        else:
            self._table.setColumnHidden(self.COL_AMOUNT, True)
        layout.addWidget(self._table)

        # Open button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._open_btn = QPushButton("Open")
        self._open_btn.clicked.connect(self._on_open)
        btn_layout.addWidget(self._open_btn)
        layout.addLayout(btn_layout)

    def _toggle_amounts(self, state):
        show = state == Qt.Checked
        self._table.setColumnHidden(self.COL_AMOUNT, not show)
        for edit in self._amount_edits:
            edit.setEnabled(show)

    def _on_open(self):
        # Collect selected tickers
        selected = []
        for row, (symbol, market, price) in enumerate(self._ticker_data):
            if not self._checkboxes[row].isChecked():
                continue

            if self._amounts_cb.isChecked():
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

            selected.append((symbol, market, amount))

        if len(selected) < 2:
            QMessageBox.warning(self, "Not Enough Tickers",
                                "Select at least 2 tickers to build a portfolio.")
            return

        # Create portfolio
        symbols = [s for s, m, a in selected]
        markets = [m for s, m, a in selected]
        amounts = [a for s, m, a in selected]

        # Filter existing_tickers to only include selected symbols
        selected_keys = set(zip(symbols, markets))
        filtered_tickers = {k: v for k, v in self._existing_tickers.items() if k in selected_keys}

        try:
            from portfolio import Portfolio, PortfolioGui
            portfolio = Portfolio(
                symbols, markets, amounts,
                existing_tickers=filtered_tickers,
                use_past_growth=self._use_past_growth
            )
            portfolio.calculate_correlation()

            gui = PortfolioGui(portfolio, show_frontier=True)
            gui.setWindowTitle("Custom Portfolio")
            self._portfolio_windows.append(gui)
            gui.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create portfolio:\n{e}")
            import traceback
            traceback.print_exc()
