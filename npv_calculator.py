#!/usr/bin/env python3

import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QLineEdit, QLabel, QPushButton, QButtonGroup, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from ticker import (
    Ticker, market_data, compute_avg_fcf, dcf_remaining_growth_years,
    save_dcf_model, load_dcf_model, search_growth, NPV_ASSUMPTIONS,
)
import json, os
import numpy as np

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npv_config.json")
with open(_config_path, "r") as _f:
    NPV_CALCULATOR_CONFIG = json.load(_f)["npv_calculator"]

from PyQt5 import QtWidgets, QtCore
from qt_material import apply_stylesheet  # import after the appropriate qtwidgets


# -----------------------------------------------------------------------------------------------------

class GrowthApp(QWidget):
    def __init__(self, ticker=None):
        super().__init__()
        self.ticker = ticker or Ticker.get_cache("QCOM", "NASDAQ")  # todo select
        self.initUI()

    def _init_radio(self, names: list, text_box: QLineEdit = None, horizontal: bool = False, prefix: str = None) -> QButtonGroup:
        radio_layout = QHBoxLayout() if horizontal else QVBoxLayout()
        radio_group = QButtonGroup(self)

        if prefix is not None:
            radio_layout.addWidget(QLabel(prefix))

        for id, name in enumerate(names):
            radio_btn = QRadioButton(name)
            radio_group.addButton(radio_btn)
            if (text_box is not None) and (id == len(names) - 1):
                custom_layout = QHBoxLayout()
                custom_layout.addWidget(radio_btn)
                custom_layout.addWidget(text_box)
                radio_layout.addLayout(custom_layout)
            else:
                radio_layout.addWidget(radio_btn)
        self.controls_layout.addLayout(radio_layout)
        return radio_group

    def initUI(self):
        self.setWindowTitle(f"{self.ticker.symbol}:{self.ticker.market}")

        # top-level horizontal split: plot on left, controls on right
        root_layout = QHBoxLayout()
        self.setLayout(root_layout)

        # --- left: matplotlib figure ---
        fig = self.ticker.plot_me(show=False)
        canvas = FigureCanvas(fig)
        canvas.setMinimumWidth(800)
        toolbar = NavigationToolbar(canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(canvas)
        plot_layout.addWidget(toolbar)
        root_layout.addLayout(plot_layout, stretch=3)

        # --- right: controls ---
        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout()
        controls_widget.setLayout(self.controls_layout)
        root_layout.addWidget(controls_widget, stretch=1)

        # Growth Trend Section
        self.trend_group = self._init_radio(prefix="Growth Trend:", names=["Linear", "Exponential"], horizontal=True)
        self.trend_group.buttons()[1].setChecked(True)  # Exponential default

        # Growth Time Section
        growth_time_layout = QHBoxLayout()
        growth_time_label = QLabel("Growth Time:")
        self.growth_time_input = QLineEdit(str(NPV_CALCULATOR_CONFIG["default_growth_time"]))
        growth_time_layout.addWidget(growth_time_label)
        growth_time_layout.addWidget(self.growth_time_input)
        self.controls_layout.addLayout(growth_time_layout)

        # FCF Basis — choose between 4-year average or TTM
        from fx_converter import fx_converter
        try:
            data = self.ticker._get_plot_data()
            avg_4y, _ = compute_avg_fcf(data, "4-year avg")
            ttm, _ = compute_avg_fcf(data, "TTM")
            currency = fx_converter.base_currency
            fcf_names = [f"4-year avg ({avg_4y:.2f} {currency}/share)", f"TTM ({ttm:.2f} {currency}/share)"]
        except Exception:
            fcf_names = ["4-year avg", "TTM"]
        self.fcf_basis_group = self._init_radio(prefix="FCF Basis:", names=fcf_names, horizontal=True)
        self.fcf_basis_group.buttons()[0].setChecked(True)  # 4-year avg default

        # Growth Benchmark Section — labels are mode-dependent (see _refresh_benchmark_labels)
        self.custom_growth_input = QLineEdit()
        self.growth_benchmark_group = self._init_radio(prefix="Growth Benchmark:", text_box=self.custom_growth_input,
                                                       names=["Earnings", "Book Value", "Revenue", "FCF", "Custom"])
        self.growth_benchmark_group.buttons()[0].setChecked(True)
        # Refresh labels now that all controls exist
        self._refresh_benchmark_labels()
        # React to trend changes (Linear/Exponential)
        self.trend_group.buttonClicked.connect(self._refresh_benchmark_labels)

        # Perpetuity Growth Section
        self.perpetuity_growth_input = QLineEdit(str(NPV_CALCULATOR_CONFIG["default_perpetuity_growth_percent"]))
        self.perpetuity_group = self._init_radio(prefix="Perpetuity Growth:", text_box=self.perpetuity_growth_input,
                                                 names=["Nothing", "Constant", "Slow Exponent"])
        self.perpetuity_group.buttons()[2].setChecked(True)  # Slow Exponent default

        # Discount Rate Section
        discount_layout = QHBoxLayout()
        discount_layout.addWidget(QLabel("Discount Rate (%):"))
        self.discount_rate_input = QLineEdit(str(NPV_CALCULATOR_CONFIG["default_discount_rate_percent"]))
        discount_layout.addWidget(self.discount_rate_input)
        self.controls_layout.addLayout(discount_layout)

        # CAPM info
        try:
            rfr = market_data.get_risk_free_rate() * 100
            mkt = market_data.get_market_return() * 100
            beta = self.ticker.statistics.get("beta")
            beta_str = f"{beta:.2f}" if beta and not np.isnan(beta) else "N/A"
            capm = self.ticker.statistics.get("capm_interest")
            capm_str = f"{capm:.1f}%" if capm and not np.isnan(capm) else "N/A"
            capm_lbl = QLabel(f"CAPM: {capm_str}  |  β: {beta_str}  |  RFR: {rfr:.1f}%  |  Mkt: {mkt:.1f}%")
        except Exception as e:
            capm_lbl = QLabel(f"CAPM: unavailable ({e})")
        capm_lbl.setStyleSheet("font-size: 11px; color: gray;")
        self.controls_layout.addWidget(capm_lbl)

        # Add balance-sheet value: None / Book Value / Cash (per share, shown inline)
        try:
            data = self.ticker._get_plot_data()
            bv_ps = data["bv"][-1] if data and len(data["bv"]) else float('nan')
            cash_ps = data["cash_ps"][-1] if data and len(data.get("cash_ps", [])) else float('nan')
            currency = fx_converter.base_currency
            add_names = ["None",
                         f"BV ({bv_ps:.2f} {currency})",
                         f"Cash ({cash_ps:.2f} {currency})"]
        except Exception:
            add_names = ["None", "BV", "Cash"]
        self.add_group = self._init_radio(prefix="Add:", names=add_names, horizontal=True)
        self.add_group.buttons()[0].setChecked(True)  # None default

        # Result label
        self.result_label = QLabel("")
        self.controls_layout.addWidget(self.result_label)

        # GO Button
        self.go_button = QPushButton("GO")
        self.controls_layout.addWidget(self.go_button)
        self.go_button.clicked.connect(self.handle_go_press)

        # Save Button
        self.save_button = QPushButton("Save DCF Model")
        self.controls_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.handle_save)

        self.controls_layout.addStretch()

        # Pre-fill from saved model if available
        self._load_saved_model()

    def _load_saved_model(self):
        """Pre-fill fields from saved DCF model if one exists for this ticker."""
        model = getattr(self.ticker, 'dcf_model', None)
        if not model:
            return

        # Show indicator
        saved_at = model.get("saved_at", "?")
        benchmark = model.get("growth_benchmark", "?")
        self.result_label.setText(f"Saved model ({saved_at}, {benchmark})")

        # Growth rate — always load as Custom with the saved rate.
        # In Linear trend, the Custom input shows the FCF $/share/yr slope,
        # otherwise the compounding percent.
        for btn in self.growth_benchmark_group.buttons():
            if btn.text() == "Custom":
                btn.setChecked(True)
                break
        is_linear = model["growth_trend"] == "Linear"
        custom_value = model["linear_growth"] if is_linear else model["growth_rate_percent"]
        self.custom_growth_input.setText(str(custom_value))

        # Growth trend
        trend = model.get("growth_trend", "")
        for btn in self.trend_group.buttons():
            if btn.text() == trend:
                btn.setChecked(True)
                break

        # Growth time (remaining years from target date)
        remaining = dcf_remaining_growth_years(model)
        self.growth_time_input.setText(str(max(1, round(remaining))))

        # Terminal model
        terminal = model.get("terminal_model", "")
        for btn in self.perpetuity_group.buttons():
            if btn.text() == terminal:
                btn.setChecked(True)
                break

        # Terminal growth
        self.perpetuity_growth_input.setText(str(model.get("terminal_growth_percent", 2.0)))

        # Discount rate
        self.discount_rate_input.setText(str(model.get("discount_rate_percent", 10.0)))

        # Add mode (None / Book Value / Cash)
        add_mode = model["add_mode"]
        add_label = {"none": "None", "book_value": "BV", "cash": "Cash"}[add_mode]
        for btn in self.add_group.buttons():
            if btn.text().split(" (")[0] == add_label:
                btn.setChecked(True)
                break

        # FCF basis
        fcf_basis = model.get("fcf_basis", "4-year avg")
        for btn in self.fcf_basis_group.buttons():
            if btn.text().startswith(fcf_basis):
                btn.setChecked(True)
                break

    def _get_benchmark_name(self) -> str:
        """Extract canonical benchmark name (without growth percentage in parentheses)."""
        text = self.growth_benchmark_group.checkedButton().text()
        return text.split(" (")[0]

    def _refresh_benchmark_labels(self):
        """Update benchmark radio labels based on Linear/Exponential mode.
        In Exponential: shows growth rate in %.
        In Linear: shows the equivalent FCF slope ($/yr) after conversion."""
        from fx_converter import fx_converter
        is_linear = self.trend_group.checkedButton().text() == "Linear"
        stats = self.ticker.statistics
        currency = fx_converter.base_currency

        if is_linear:
            # Compute FCF-equivalent linear slopes
            data = self.ticker._get_plot_data()
            avg_fcf = np.mean(data.get("free_cf_ps", data["free_cf"]))

            def _slope_to_fcf(field, avg_field_key):
                slope = self.ticker.get_growth_rate(field, linear=True)
                avg_field = np.mean(data[avg_field_key])
                if np.isnan(slope) or avg_field == 0:
                    return np.nan
                return slope * avg_fcf / avg_field

            slopes = {
                "Earnings": _slope_to_fcf("eps", "eps"),
                "Revenue":  _slope_to_fcf("revenue_ps", "revenue_ps"),
                "FCF":      self.ticker.get_growth_rate("free_cf_ps", linear=True),  # already in FCF units
            }

            def _fmt_slope(s):
                return f" ({s:.2f} {currency}/share/yr)" if not np.isnan(s) else ""

            new_labels = {
                "Earnings": f"Earnings{_fmt_slope(slopes['Earnings'])}",
                "Book Value": "Book Value",  # hidden anyway
                "Revenue": f"Revenue{_fmt_slope(slopes['Revenue'])}",
                "FCF": f"FCF{_fmt_slope(slopes['FCF'])}",
                "Custom": "Custom",
            }
        else:
            def _fmt_pct(rate):
                return f" ({rate:.1f}%)" if rate is not None and not np.isnan(rate) else ""

            new_labels = {
                "Earnings":   f"Earnings{_fmt_pct(stats.get('growth_rate'))}",
                "Book Value": f"Book Value{_fmt_pct(stats.get('bv_growth_rate'))}",
                "Revenue":    f"Revenue{_fmt_pct(stats.get('revenue_growth_rate'))}",
                "FCF":        f"FCF{_fmt_pct(stats.get('fcf_growth_rate'))}",
                "Custom":     "Custom",
            }

        # Apply: relabel + hide BV in linear mode
        for btn in self.growth_benchmark_group.buttons():
            canonical = btn.text().split(" (")[0]
            btn.setText(new_labels.get(canonical, canonical))
            if canonical == "Book Value":
                btn.setVisible(not is_linear)
                if is_linear and btn.isChecked():
                    # Switch to Earnings if BV was selected
                    self.growth_benchmark_group.buttons()[0].setChecked(True)

    def _get_fcf_basis(self) -> str:
        """Extract canonical FCF basis name."""
        text = self.fcf_basis_group.checkedButton().text()
        return text.split(" (")[0]

    def _get_add_mode(self) -> str:
        """Return the balance-sheet add mode: 'none' | 'book_value' | 'cash'."""
        label = self.add_group.checkedButton().text().split(" (")[0]
        return {"None": "none", "BV": "book_value", "Cash": "cash"}[label]

    def _resolve_growth(self, benchmark, is_linear):
        """Return (growth_rate_percent, linear_growth_per_share_per_year) for the chosen benchmark.
        - growth_rate_percent: compounding rate in % (always meaningful, used for exp mode)
        - linear_growth: absolute FCF $/share/yr (only meaningful in linear mode; 0 otherwise)
        """
        stats = self.ticker.statistics
        if benchmark == "Book Value":
            growth_rate = stats["bv_growth_rate"]
        elif benchmark == "Earnings":
            growth_rate = stats["growth_rate"]
        elif benchmark == "Revenue":
            growth_rate = stats["revenue_growth_rate"]
        elif benchmark == "FCF":
            growth_rate = stats.get("fcf_growth_rate", float('nan'))
        else:  # Custom
            growth_rate = float(self.custom_growth_input.text())

        linear_growth = 0
        if is_linear:
            data = self.ticker._get_plot_data()
            avg_fcf = np.mean(data.get("free_cf_ps", data["free_cf"]))
            if benchmark == "Custom":
                # User-provided slope in FCF $/year
                linear_growth = float(self.custom_growth_input.text())
            else:
                field_map = {"Earnings": ("eps", "eps"),
                             "Revenue":  ("revenue_ps", "revenue_ps"),
                             "FCF":      ("free_cf_ps", "free_cf_ps")}
                field, avg_key = field_map[benchmark]
                slope = self.ticker.get_growth_rate(field, linear=True)
                avg_value = np.mean(data[avg_key])
                linear_growth = slope * avg_fcf / avg_value if avg_value != 0 else 0

        return growth_rate, linear_growth

    def handle_go_press(self):
        benchmark = self._get_benchmark_name()
        is_linear = self.trend_group.checkedButton().text() == "Linear"

        # Block BV in linear mode (relationship to FCF is too indirect)
        if is_linear and benchmark == "Book Value":
            self.result_label.setText("Book Value not supported in linear mode")
            return

        growth_rate, linear_growth = self._resolve_growth(benchmark, is_linear)

        # Pick the single growth value: linear slope or compounding percent
        growth = linear_growth if is_linear else growth_rate

        args_iir = {
            "forward_to_present": True,
            "growth": growth,
            "add_mode": self._get_add_mode(),
            "short_term_is_linear": is_linear,
            "long_term_growth_duration": 0 if self.perpetuity_group.checkedButton().text() == "Nothing" else -1,
            "short_term_growth_duration": int(self.growth_time_input.text()),
            "maximal_long_term_growth_rate": float(self.perpetuity_growth_input.text()) / 100 if self.perpetuity_group.checkedButton().text() == "Slow Exponent" else 0,
        }
        print(args_iir)
        discount_rate = float(self.discount_rate_input.text()) / 100
        avg_fcf, fcf_offset = compute_avg_fcf(self.ticker._get_plot_data(), self._get_fcf_basis())
        if fcf_offset > 0:
            if is_linear:
                avg_fcf += linear_growth * fcf_offset
            else:
                avg_fcf *= (1 + growth_rate / 100) ** fcf_offset
        calc_npv, price = self.ticker._build_dcf_from_plot_data(
            growth=growth,
            avg_fcf=avg_fcf,
            short_term_is_linear=is_linear,
            add_mode=args_iir["add_mode"],
            long_term_growth_duration=args_iir["long_term_growth_duration"],
            short_term_growth_duration=args_iir["short_term_growth_duration"],
            maximal_long_term_growth_rate=args_iir["maximal_long_term_growth_rate"],
        )
        if calc_npv is None:
            self.result_label.setText("Cannot compute: insufficient data")
            return
        price_target = calc_npv(discount_rate)
        iir = search_growth(calc_npv, price, min_growth=NPV_ASSUMPTIONS["irr_search_min"])
        print(iir)
        if is_linear:
            growth_str = "Linear ({:.2f} {}/share/yr)".format(linear_growth, "USD")
        else:
            growth_str = "Exponential ({:.1f}%)".format(growth_rate)

        self.result_label.setText(
            "Growth: {}\nPrice Target: {:.2f}\nIRR: {:.2f}%".format(growth_str, price_target, iir))

    def handle_save(self):
        import datetime

        benchmark = self._get_benchmark_name()
        is_linear = self.trend_group.checkedButton().text() == "Linear"
        stats = self.ticker.statistics

        growth_rate, linear_growth = self._resolve_growth(benchmark, is_linear)

        growth_years = int(self.growth_time_input.text())
        growth_phase_end = (datetime.date.today() + datetime.timedelta(days=int(growth_years * 365.25))).isoformat()

        save_dcf_model(
            symbol=self.ticker.symbol,
            market=self.ticker.market,
            growth_rate_percent=growth_rate,
            linear_growth=linear_growth,
            growth_benchmark=benchmark,
            growth_trend=self.trend_group.checkedButton().text(),
            growth_phase_end=growth_phase_end,
            terminal_growth_percent=float(self.perpetuity_growth_input.text()),
            terminal_model=self.perpetuity_group.checkedButton().text(),
            discount_rate_percent=float(self.discount_rate_input.text()),
            add_mode=self._get_add_mode(),
            fcf_basis=self._get_fcf_basis(),
            last_report_date=str(stats.get("updated at", "")),
        )
        self.result_label.setText(f"Saved DCF model for {self.ticker.symbol}:{self.ticker.market}")
        # Update in-memory model so reopening the NPV calculator reflects the save
        self.ticker.dcf_model = load_dcf_model(self.ticker.symbol, self.ticker.market)


def main():
    app = QApplication(sys.argv)

    # setup stylesheet
    apply_stylesheet(app, theme='dark_red.xml')

    ex = GrowthApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
