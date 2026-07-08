#!/usr/bin/env python3

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget, QCheckBox, QPushButton, QFileDialog)

from gui.ticker_table import TickersTableView, TickersTableModel

import pandas as pd
import os
import sys

import stocks_analyzer
from qt_material import apply_stylesheet


class TickerFilterProxyModel(QSortFilterProxyModel):
    """Supports healthy/overvalued row filtering and numeric column sorting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._healthy_only = False
        self._hide_overvalued = False

    def set_healthy_only(self, value):
        self._healthy_only = value
        self.invalidateFilter()

    def set_hide_overvalued(self, value):
        self._hide_overvalued = value
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        cols = model._data.columns
        row = model._data.iloc[source_row]
        if self._healthy_only and not row.get("healthy", True):
            return False
        if self._hide_overvalued and row.get("overvalued", False):
            return False
        return True

    def lessThan(self, left, right):
        # numeric-aware sort
        l_data = self.sourceModel().data(left, Qt.DisplayRole)
        r_data = self.sourceModel().data(right, Qt.DisplayRole)
        try:
            return float(l_data) < float(r_data)
        except (ValueError, TypeError):
            return str(l_data) < str(r_data)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # map vertical header through to source so colors still work
        if orientation == Qt.Vertical:
            source_row = self.mapToSource(self.index(section, 0)).row()
            return self.sourceModel().headerData(source_row, Qt.Vertical, role)
        return super().headerData(section, orientation, role)


class tickers_gui(QWidget):

    def __init__(self, tickers_stats):
        super().__init__()
        self.full_df = tickers_stats
        self.tldr_df = self.full_df[stocks_analyzer.tldr_statistics]
        self.setWindowTitle("Stocks analyzer")
        self.setGeometry(50, 200, 1500, 1000)
        self.place_tickers(self.tldr_df)
        self.show()

    def place_tickers(self, tickers_stats):
        vlayout = QVBoxLayout()

        # --- table ---
        self.tickers_table = TickersTableView(self)
        self.source_model = TickersTableModel(tickers_stats)

        self.proxy_model = TickerFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.source_model)

        self.tickers_table.setModel(self.proxy_model)
        self.tickers_table.setSortingEnabled(False)
        self.tickers_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tickers_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        header = self.tickers_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        header.sectionClicked.connect(self._on_header_clicked)
        self._sort_column = -1
        self._sort_order = Qt.AscendingOrder

        # --- checkboxes ---
        check_layout = QHBoxLayout()
        self.healthy_cb = QCheckBox("Healthy Only")
        self.overvalued_cb = QCheckBox("Hide Overvalued")
        check_layout.addWidget(self.healthy_cb)
        check_layout.addWidget(self.overvalued_cb)
        check_layout.addStretch()

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        check_layout.addWidget(self.export_btn)

        self.build_btn = QPushButton("Build Portfolio")
        self.build_btn.clicked.connect(self._open_portfolio_builder)
        check_layout.addWidget(self.build_btn)

        self.healthy_cb.stateChanged.connect(self._apply_filters)
        self.overvalued_cb.stateChanged.connect(self._apply_filters)
        vlayout.addLayout(check_layout)

        vlayout.addWidget(self.tickers_table)
        self.setLayout(vlayout)

    def _apply_filters(self):
        self.proxy_model.set_healthy_only(self.healthy_cb.isChecked())
        self.proxy_model.set_hide_overvalued(self.overvalued_cb.isChecked())

    def _on_header_clicked(self, col):
        if self._sort_column != col:
            # new column: sort ascending
            self._sort_column = col
            self._sort_order = Qt.AscendingOrder
        elif self._sort_order == Qt.AscendingOrder:
            self._sort_order = Qt.DescendingOrder
        else:
            # third click: reset to original order
            self._sort_column = -1
            self.proxy_model.sort(-1)
            self.tickers_table.horizontalHeader().setSortIndicatorShown(False)
            return
        self.proxy_model.sort(self._sort_column, self._sort_order)
        self.tickers_table.horizontalHeader().setSortIndicatorShown(True)

    def _export_csv(self):
        os.makedirs(stocks_analyzer.OUTPUT_DIR, exist_ok=True)
        default_path = os.path.join(os.getcwd(), stocks_analyzer.OUTPUT_DIR, "statistics.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", default_path,
            "CSV Files (*.csv)"
        )
        if not path:
            return
        base, ext = os.path.splitext(path)
        self.full_df.to_csv(base + ext)
        self.tldr_df.to_csv(base + "_tldr" + ext)

    def _open_portfolio_builder(self):
        from gui.portfolio_builder import PortfolioBuilderDialog

        # If rows are selected, use only those; otherwise use all
        selected_proxy_rows = self.tickers_table.selectionModel().selectedRows()
        if len(selected_proxy_rows) > 1:
            source_rows = sorted(set(
                self.proxy_model.mapToSource(idx).row() for idx in selected_proxy_rows
            ))
            indices = [self.full_df.index[r] for r in source_rows]
        else:
            indices = list(self.full_df.index)

        ticker_data = []
        for idx in indices:
            parts = idx.split(":", 1)
            if len(parts) == 2:
                symbol, market = parts
            else:
                continue
            price = self.full_df.loc[idx].get("price on update", 0)
            if price is None or (isinstance(price, float) and price != price):
                price = 0
            ticker_data.append((symbol, market, price))
        self._builder_dialog = PortfolioBuilderDialog(
            ticker_data, parent=self
        )
        self._builder_dialog.show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape or e.text() == 'q':
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_red.xml')

    input_file = stocks_analyzer.select_stocks_file()
    tickers = stocks_analyzer.create_tickers_from_file(input_file)
    df = stocks_analyzer.ticker_list_to_df(tickers)

    window = tickers_gui(df)
    sys.exit(app.exec_())
