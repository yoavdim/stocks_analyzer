#!/usr/bin/env python3

import colorsys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSplitter, QFileDialog
from PyQt5.QtCore import Qt

from ticker import Ticker, TickerGroup, FundamentalMixin, market_data


def allocate_portfolio(method: str, symbols: list, markets: list, prices: list,
                       market_caps: list, total_value: float):
    """Compute weights and run DiscreteAllocation.greedy_portfolio in one step.

    All inputs are pre-filtered parallel lists (no None / zero entries).
    market_caps is only used when method == "market_cap"; pass None otherwise.

    Returns (out_symbols, out_markets, out_quantities) or None if no shares allocated.
    """
    from pypfopt.discrete_allocation import DiscreteAllocation

    n = len(symbols)
    if method == "equal":
        weights = np.ones(n) / n
    elif method == "market_cap":
        caps = np.array(market_caps, dtype=float)
        total_cap = caps.sum()
        if total_cap <= 0:
            return None
        weights = caps / total_cap
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # DiscreteAllocation needs per-asset string keys; use "SYM:MKT" so we can map back.
    keys = [f"{s}:{m}" for s, m in zip(symbols, markets)]
    weights_dict = dict(zip(keys, weights))
    latest_prices = pd.Series(dict(zip(keys, prices)))

    da = DiscreteAllocation(weights_dict, latest_prices, total_portfolio_value=total_value)
    allocation, leftover = da.greedy_portfolio()

    print(f"\n{method} Allocation (Total: ${total_value:.2f}, Leftover: ${leftover:.2f}):")
    key_to_info = dict(zip(keys, zip(symbols, markets, prices)))
    out_symbols, out_markets, out_quantities = [], [], []
    for key, quantity in allocation.items():
        sym, mkt, price = key_to_info[key]
        out_symbols.append(sym)
        out_markets.append(mkt)
        out_quantities.append(quantity)
        print(f"  {sym}: {quantity} shares @ ${price:.2f} = ${quantity * price:.2f}")

    if not out_symbols:
        return None
    return out_symbols, out_markets, out_quantities


class Portfolio(TickerGroup, FundamentalMixin):
    """
    Used to predict future growth and volatility.
    Can show the efficient frontier and this portfolio plotted on it.
    """
    def __init__(self, symbols: list, markets: list, quantities: list, *,
                 risk_free_rate=None, existing_tickers: dict = dict(), use_past_growth=False):
        super().__init__(symbols, markets, risk_free_rate=risk_free_rate,
                         existing_tickers=existing_tickers, use_past_growth=use_past_growth)
        self.current_prices = self.get_stock_prices_now()
        self.quantities = np.array(quantities)
        # Check if this is an actual portfolio with holdings
        self.has_holdings = any(q > 0 for q in self.quantities)
        
        if self.has_holdings:
            self.weights = self.quantities * np.array(self.current_prices)
            self.weights = self.weights / np.sum(self.weights)
        else:
            # For portfolios without holdings, set equal weights for analysis purposes TODO: remove this?
            self.weights = np.ones(len(quantities)) / len(quantities) if len(quantities) > 0 else np.array([])
        
        self.weights_dict = dict(zip(zip(self.symbols, self.markets), self.weights))
        self.portfolio_annual_growth_forecast = np.nan
        self.portfolio_std = np.nan
        # Set by calculate_correlation -> calculate_portfolio_beta TODO: better names
        self.portfolio_monthly_returns = None
        self.portfolio_betas = {}  # betas of tickers with regard to the avg portfolio
        self.portfolio_beta = np.nan # beta of the avg portfolio with regard to the market

    def get_weight(self, symbol: str, market: str):
        return round((self.weights_dict[(symbol, market)] * 100), 2)

    def calculate_correlation(self):
        super().calculate_correlation()
        valid_mask = np.array([f in self.valid_full_symbols for f in self.full_symbols])
        # 1st calculate the portfolio weights discarding all the non valid covariance tickers (Todo its dangerous to say its real)
        weights = self.weights[valid_mask]
        weights = weights / np.sum(weights)

        forecasts = np.array(self.annual_growth_forecasts)[valid_mask]
        self.portfolio_annual_growth_forecast = np.dot(weights, forecasts)
        self.portfolio_std = np.sqrt(weights.T @ self.cov @ weights)

        # portfolio historic returns and per-ticker beta relative to portfolio
        monthly = self.get_monthly_growths()[self.valid_full_symbols]
        self.portfolio_monthly_returns = monthly @ weights
        portfolio_var = self.portfolio_monthly_returns.var()
        self.portfolio_betas = {}
        for f in self.full_symbols:
            if f not in self.valid_full_symbols:
                self.portfolio_betas[f] = np.nan
                continue
            cov_with_portfolio = monthly[f].cov(self.portfolio_monthly_returns)
            self.portfolio_betas[f] = cov_with_portfolio / portfolio_var if portfolio_var > 0 else np.nan

        # Calculate weighted average beta for the portfolio relative to the market
        self.calculate_portfolio_beta()

    def calculate_portfolio_beta(self):
        """Calculate the weighted average beta of the portfolio"""
        total_beta = 0.0
        total_weight = 0.0
        
        for full_symbol, weight in zip(self.full_symbols, self.weights):
            beta = self.beta_dictionary.get(full_symbol, np.nan)
            if not np.isnan(beta):
                total_beta += weight * beta
                total_weight += weight
        
        if total_weight > 0:
            self.portfolio_beta = total_beta / total_weight
        else:
            self.portfolio_beta = np.nan

    def get_betas_df(self):
        """Return portfolio betas as a DataFrame indexed by symbol, including all stocks."""
        full_to_sym = dict(zip(self.full_symbols, self.symbols))
        
        # Start with betas for stocks with valid covariance then add NaN to the rest
        data = {full_to_sym.get(f, f): {"Portfolio Beta": b} for f, b in self.portfolio_betas.items()}
        data.update({full_to_sym[f]: {"Portfolio Beta": np.nan} for f in self.full_symbols if full_to_sym[f] not in data})
        df = pd.DataFrame.from_dict(data, orient="index")
        return df
    
    def get_portfolio_table_df(self):
        """Return a consolidated DataFrame with all portfolio data: weights (if has_holdings) and betas."""
        data = {}
        
        for symbol, full_symbol, weight in zip(self.symbols, self.full_symbols, self.weights):
            beta = self.portfolio_betas[full_symbol]
            
            if self.has_holdings:
                # Show weights and betas
                data[symbol] = {
                    "Weight (%)": weight * 100,
                    "Portfolio Beta": beta
                }
            else:
                # Just list the symbol
                data[symbol] = {}
        
        df = pd.DataFrame.from_dict(data, orient="index")
        return df

    def plot_pie(self, ax=None):
        if not ax:
            _, ax = plt.subplots()
        ax.pie(self.weights, labels=self.symbols)

    def get_weighted_stats(self):
        """PE = weighted_sum(price) / weighted_sum(eps), ROE = weighted_sum(eps) / weighted_sum(bv)"""
        total_price, total_eps, total_bv = 0.0, 0.0, 0.0
        for (sym, mkt), w in self.weights_dict.items():
            if w == 0:
                continue  # Skip tickers with zero weight
            
            t = self.tickers_dictionary.get((sym, mkt))
            if not t:
                continue  # Skip if not in dictionary (indices won't be here)
            
            try:
                total_eps += w * t.statistics["eps"]
                total_bv += w * t.statistics["book_value"]
                total_price += w * t.statistics["price on update"]
            except e:
                print(f"Warning: Ticker {sym}:{mkt} missing required field")
                raise e
        
        pe = total_price / total_eps
        roe = (total_eps / total_bv * 100)
        return pe, roe

    def get_current_price(self):
        """Total portfolio price, excluding indices and tickers without financial reports."""
        return sum(q * p for (sym, mkt), q, p in zip(
            zip(self.symbols, self.markets), self.quantities, self.current_prices)
            if p and p > 0 and (sym, mkt) in self.tickers_dictionary)

    def get_name(self):
        return "Portfolio"

    def plot_concentric_pie(self, ax=None):
        """Two pies: left=tickers sorted by sector, right=sector+industry concentric."""
        if ax is None:
            _, (ax_tickers, ax_sectors) = plt.subplots(1, 2)
        else:
            ax_tickers, ax_sectors = ax

        data = []
        for sym, mkt, w in zip(self.symbols, self.markets, self.weights):
            if w == 0: continue
            t = self.tickers_dictionary.get((sym, mkt))
            sector = (t.statistics.get("sector") or "Unknown") if t else "Unknown"
            industry = (t.statistics.get("industry") or "Unknown") if t else "Unknown"
            data.append((sector, industry, sym, w))

        data.sort(key=lambda x: (x[0], x[1]))
        sectors_sorted    = [d[0] for d in data]
        industries_sorted = [d[1] for d in data]
        symbols_sorted    = [d[2] for d in data]
        weights_sorted    = [d[3] for d in data]

        industry_slices, sector_slices = [], []
        for ind, sec, w in zip(industries_sorted, sectors_sorted, weights_sorted):
            if industry_slices and industry_slices[-1][0] == ind:
                industry_slices[-1] = (ind, industry_slices[-1][1] + w)
            else:
                industry_slices.append((ind, w))
            if sector_slices and sector_slices[-1][0] == sec:
                sector_slices[-1] = (sec, sector_slices[-1][1] + w)
            else:
                sector_slices.append((sec, w))

        cmap = plt.get_cmap("tab10")
        n_sec = len(sector_slices)
        sec_color = {s[0]: cmap(i / max(n_sec, 1)) for i, s in enumerate(sector_slices)}

        ax_tickers.pie(weights_sorted, labels=symbols_sorted,
                       wedgeprops=dict(edgecolor='w'), labeldistance=0.6)
        ax_tickers.set_title("Tickers")

        sec_industry_count = {}
        sec_industry_idx = {}
        for s in industry_slices:
            ind = s[0]
            sec = sectors_sorted[[i for i, x in enumerate(industries_sorted) if x == ind][0]]
            sec_industry_count[sec] = sec_industry_count.get(sec, 0) + 1
        for sec in sec_industry_count:
            sec_industry_idx[sec] = 0

        mid_colors = []
        for s in industry_slices:
            ind = s[0]
            sec = sectors_sorted[[i for i, x in enumerate(industries_sorted) if x == ind][0]]
            base = sec_color[sec][:3]
            total = sec_industry_count[sec]
            idx = sec_industry_idx[sec]
            factor = 0.35 + 0.5 * (idx / max(total - 1, 1))
            h, l, sat = colorsys.rgb_to_hls(*base)
            mid_colors.append(colorsys.hls_to_rgb(h, factor, sat))
            sec_industry_idx[sec] += 1

        ax_sectors.pie([s[1] for s in industry_slices],
                       labels=None,
                       radius=1.0, colors=mid_colors,
                       wedgeprops=dict(width=0.4, edgecolor='w'))
        # Store industry names on wedges for hover tooltip
        for wedge, (ind, _) in zip(ax_sectors.patches, industry_slices):
            wedge.set_label(ind)
        ax_sectors.pie([s[1] for s in sector_slices],
                       labels=[s[0] for s in sector_slices],
                       radius=0.6, colors=[sec_color[s[0]] for s in sector_slices],
                       wedgeprops=dict(width=0.6, edgecolor='w'), labeldistance=0.8)
        ax_sectors.set_title("Sector / Industry")

        fig = ax_tickers.get_figure()
        annot = fig.text(0, 0, "", va="bottom", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8), visible=False)
        all_wedges = [w for ax in (ax_tickers, ax_sectors)
                      for w in ax.patches if hasattr(w, 'get_label')]

        def on_hover(event):
            visible = False
            for wedge in all_wedges:
                if wedge.contains(event)[0]:
                    pct_val = (wedge.theta2 - wedge.theta1) / 360 * 100
                    annot.set_text(f"{wedge.get_label()}: {pct_val:.1f}%")
                    annot.set_position((event.x / fig.get_size_inches()[0] / fig.dpi,
                                        event.y / fig.get_size_inches()[1] / fig.dpi))
                    visible = True
                    break
            annot.set_visible(visible)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        self._ticker_wedges = list(zip(ax_tickers.patches, symbols_sorted))

    def plot_portfolio(self, ax=None):
        ax = self.plot_frontier(ax=ax)
        # Plot portfolio point with actual std (axis labels will show normalized values)
        print(f"[plot_portfolio] std={self.portfolio_std}, growth={self.portfolio_annual_growth_forecast}, beta={self.portfolio_beta}")
        ax.plot(self.portfolio_std, self.portfolio_annual_growth_forecast, 'ro', markersize=10,
               label=f'Portfolio ({self.portfolio_annual_growth_forecast*100:.1f}%)', zorder=6)
        ax.legend(fontsize=8, markerscale=0.6)  # refresh legend to include portfolio dot

    def plot_portfolio_on_capm(self, ax):
        """Plot the current portfolio point on the CAPM graph"""
        if self.has_holdings:
            print(f"[plot_portfolio_on_capm] beta={self.portfolio_beta}, growth={self.portfolio_annual_growth_forecast}")
            if hasattr(self, 'portfolio_beta') and not np.isnan(self.portfolio_beta):
                growth_pct = self.portfolio_annual_growth_forecast * 100
                ax.plot(self.portfolio_beta, growth_pct, 'ro', markersize=12, 
                       label='Portfolio', zorder=5)

    def _get_plot_data(self):
        """Aggregate financial time series across all holdings for plot_me."""
        if hasattr(self, '_plot_data_cache'):
            return self._plot_data_cache
        series = {k: [] for k in ("bv", "eps", "revenue_ps", "operating_cf", "free_cf", "prices")}
        price_dfs = []

        for sym, mkt, qty in zip(self.symbols, self.markets, self.quantities):
            ticker = self.tickers_dictionary.get((sym, mkt))
            if ticker is None:
                continue
            try:
                d = ticker._get_plot_data()
            except Exception as e:
                print(f"Skipping {sym}:{mkt} in portfolio plot: {e}")
                continue

            amount = qty if qty > 0 else 1
            times = d["times"]
            for key in series:
                src_key = key + "_ps" if key in ("operating_cf", "free_cf") else key
                series[key].append(pd.Series(d[src_key] * amount, index=times))

            try:
                hist = ticker.yahoo_info.yf_ticker.history(period="10y")["Close"]
                price_dfs.append(hist * amount)
            except Exception:
                pass

        if not series["eps"]:
            return None

        def aggregate(series_list):
            resampled = [s.resample("ME").last().ffill() for s in series_list]
            return pd.concat(resampled, axis=1).ffill().sum(axis=1)

        agg = {k: aggregate(v) for k, v in series.items()}
        times = list(agg["eps"].index)

        # Store price_dfs for _get_plot_price_data
        self._price_dfs = price_dfs
        
        self._plot_data_cache = {
            "times": times,
            "bv": np.array(agg["bv"]),
            "eps": np.array(agg["eps"]),
            "revenue_ps": np.array(agg["revenue_ps"]),
            "operating_cf": np.array(agg["operating_cf"]),
            "free_cf": np.array(agg["free_cf"]),
            "prices": np.array(agg["prices"]),
        }
        return self._plot_data_cache

    def _get_plot_price_data(self):
        """Daily portfolio price history for the bottom price graph. Cached separately."""
        if hasattr(self, '_plot_price_data_cache'):
            return self._plot_price_data_cache
        price_dfs = getattr(self, '_price_dfs', [])
        if price_dfs:
            portfolio_daily = pd.concat(price_dfs, axis=1).ffill().sum(axis=1)
        else:
            portfolio_daily = pd.Series(dtype=float)
        self._plot_price_data_cache = {
            "price_times": portfolio_daily.index,
            "price_values": portfolio_daily,
            "new_price_times": None,
            "new_price_values": None,
            "price_series": portfolio_daily,
        }
        return self._plot_price_data_cache



class PortfolioGui(QWidget):
    """Base portfolio GUI: shows frontier, pie charts, screener button.
    Summary text area is left blank — subclasses fill it via set_summary()."""

    def __init__(self, portfolio: Portfolio, show_frontier=False):
        super().__init__()
        self.setWindowTitle("Portfolio Analyzer")
        self.setGeometry(100, 100, 1400, 700)
        self._portfolio = portfolio
        self._growth_windows = []
        self._screener_window = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        if show_frontier:
            fig_frontier, (ax_ef, ax_capm) = plt.subplots(2, 1, figsize=(8, 10))
            portfolio.plot_portfolio(ax=ax_ef)
            growth_mode = "Past Growth" if portfolio.use_past_growth else "DCF Forecast"
            ax_ef.set_ylabel("Expected Return (%s)" % growth_mode)
            ax_ef.grid(True)
            
            # Add CAPM graph below
            portfolio.plot_capm(ax=ax_capm)
            # Add portfolio point to CAPM graph
            portfolio.plot_portfolio_on_capm(ax=ax_capm)
            # Add legend after all elements are plotted
            ax_capm.legend(fontsize=8, markerscale=0.6)
            
            fig_frontier.subplots_adjust(hspace=0.35)
            canvas = FigureCanvas(fig_frontier)
            frontier_widget = QWidget()
            frontier_layout = QVBoxLayout(frontier_widget)
            frontier_layout.addWidget(canvas)
            frontier_layout.addWidget(NavigationToolbar(canvas, self))
            splitter.addWidget(frontier_widget)

        stats_widget = QWidget()
        self._stats_layout = QVBoxLayout(stats_widget)
        self._stats_layout.setAlignment(Qt.AlignTop)
        splitter.addWidget(stats_widget)

        if show_frontier:
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1 if portfolio.has_holdings else 0)

        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet("font-size: 14px; padding: 8px;")
        self._stats_layout.addWidget(self._summary_lbl)
        if not portfolio.has_holdings:
            self._summary_lbl.hide()

        # Consolidated table showing weights (if has_holdings) and betas
        self._table_lbl = QLabel("")
        self._table_lbl.setStyleSheet("font-family: monospace; font-size: 12px; padding: 8px;")
        self._table_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        table_df = portfolio.get_portfolio_table_df()
        if table_df.empty or table_df.columns.empty:
            self._table_lbl.setText("\n".join(portfolio.symbols))
        else:
            self._table_lbl.setText(table_df.to_string(float_format=lambda x: "%.2f" % x))
        self._stats_layout.addWidget(self._table_lbl)

        # portfolio stats (only show if has holdings)
        if portfolio.has_holdings:
            try:
                pe, roe = portfolio.get_weighted_stats()
                eps_growth = portfolio.get_growth_rate("eps")
                irr = portfolio.get_irr()
                capm_discount = portfolio.get_capm_discount()
                stats_text = "PE: {:.1f}  |  ROE: {:.1f}%  |  Growth: {:.1f}%  |  IRR: {:.1f}%  |  CAPM: {:.0f}%".format(
                    pe, roe, eps_growth, irr, capm_discount)
            except Exception as e:
                stats_text = "Stats unavailable ({})".format(e)
            self._stats_lbl = QLabel(stats_text)
            self._stats_lbl.setStyleSheet("font-size: 12px; padding: 4px; color: gray;")
            self._stats_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._stats_layout.addWidget(self._stats_lbl)

        btn = QPushButton("Open Screener")
        btn.clicked.connect(self._open_screener)
        self._stats_layout.addWidget(btn)

        btn_plot = QPushButton("Plot Fundamentals")
        btn_plot.clicked.connect(self._plot_portfolio)
        self._stats_layout.addWidget(btn_plot)

        # Optimal / Min Variance buttons on the same line
        opt_row = QHBoxLayout()
        if portfolio.tangency_portfolio:
            btn_optimal = QPushButton("Open Optimal")
            btn_optimal.clicked.connect(self._open_optimal)
            opt_row.addWidget(btn_optimal)

        if portfolio.min_var_portfolio:
            btn_min_var = QPushButton("Open Min Variance")
            btn_min_var.clicked.connect(self._open_min_variance)
            opt_row.addWidget(btn_min_var)

        if opt_row.count() > 0:
            self._stats_layout.addLayout(opt_row)

        # Build Portfolio & Save As on the same line
        build_row = QHBoxLayout()
        btn_build = QPushButton("Build Portfolio")
        btn_build.clicked.connect(self._open_portfolio_builder)
        build_row.addWidget(btn_build)

        btn_save = QPushButton("Save As")
        btn_save.clicked.connect(self._save_portfolio)
        build_row.addWidget(btn_save)
        self._stats_layout.addLayout(build_row)

        # Only show pie chart if portfolio has holdings
        if portfolio.has_holdings:
            fig_pie, (ax_t, ax_s) = plt.subplots(1, 2)
            try:
                portfolio.plot_concentric_pie(ax=(ax_t, ax_s))
            except Exception as e:
                ax_t.text(0.5, 0.5, f"Pie unavailable:\n{e}", ha='center', va='center', transform=ax_t.transAxes)
            self._stats_layout.addWidget(FigureCanvas(fig_pie))

            def on_pie_dblclick(event):
                if event.dblclick and event.inaxes is ax_t:
                    for wedge, symbol in portfolio._ticker_wedges:
                        if wedge.contains(event)[0]:
                            market = dict(zip(portfolio.symbols, portfolio.markets)).get(symbol)
                            if market:
                                from npv_calculator import GrowthApp
                                ticker = portfolio.tickers_dictionary.get((symbol, market)) or \
                                         Ticker.get_cache(symbol, market)
                                win = GrowthApp(ticker=ticker)
                                self._growth_windows.append(win)
                                win.show()
                            break
            fig_pie.canvas.mpl_connect('button_press_event', on_pie_dblclick)

    def set_summary(self, summary_text, perf_df):
        self._summary_lbl.setText(summary_text)
        self._table_lbl.setText(perf_df.to_string(float_format=lambda x: "%.1f" % x))

    def _open_screener(self):
        from ticker_gui import tickers_gui
        df = self._portfolio.to_df()
        self._screener_window = tickers_gui(df)
        self._screener_window.show()

    def _plot_portfolio(self):
        fig = self._portfolio.plot_me(show=False)
        if fig:
            fig.show()

    def _open_portfolio_builder(self):
        from gui.portfolio_builder import PortfolioBuilderDialog
        p = self._portfolio
        ticker_data = []
        for sym, mkt, price in zip(p.symbols, p.markets, p.current_prices):
            ticker_data.append((sym, mkt, price))
        amounts = list(p.quantities) if p.has_holdings else None
        self._builder_dialog = PortfolioBuilderDialog(
            ticker_data,
            existing_tickers=p.tickers_dictionary,
            use_past_growth=p.use_past_growth,
            amounts=amounts,
            parent=self
        )
        self._builder_dialog.show()

    def _save_portfolio(self):
        import os
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Portfolio", os.path.join(default_dir, "portfolio.tsv"),
            "TSV Files (*.tsv)"
        )
        if not path:
            return

        p = self._portfolio
        rows = []
        if p.has_holdings:
            rows.append("Ticker\tMarket\tAmount")
            for sym, mkt, qty in zip(p.symbols, p.markets, p.quantities):
                rows.append(f"{sym}\t{mkt}\t{qty}")
        else:
            rows.append("Ticker\tMarket")
            for sym, mkt in zip(p.symbols, p.markets):
                rows.append(f"{sym}\t{mkt}")

        with open(path, "w") as f:
            f.write("\n".join(rows) + "\n")

        portfolio_name = os.path.splitext(os.path.basename(path))[0]
        self.setWindowTitle(f"Portfolio Analyzer - {portfolio_name}")

    def _open_optimal(self):
        self._open_portfolio_from_weights(self._portfolio.tangency_portfolio, "Optimal Portfolio (Tangency)")

    def _open_min_variance(self):
        self._open_portfolio_from_weights(self._portfolio.min_var_portfolio, "Min Variance Portfolio")

    def _open_portfolio_from_weights(self, target_weights, title):
        """Create and open a new portfolio from a dict of {full_symbol: weight}."""
        try:
            from pypfopt.discrete_allocation import DiscreteAllocation
            from ticker import is_stock

            p = self._portfolio
            total_value = sum(q * pr for q, pr in zip(p.quantities, p.current_prices))
            if total_value <= 0:
                total_value = 100_000

            # TODO: revisit — do we actually need to ban ETFs/indices from EF outputs?
            # The motivation was to avoid the optimizer collapsing onto SPY-like assets,
            # but a user who included those tickers presumably wants them considered.
            # Filter to stocks with positive price that have weight > 0
            symbols, markets, full_symbols, prices = [], [], [], []
            for sym, mkt, fsym, price in zip(p.symbols, p.markets, p.full_symbols, p.current_prices):
                w = target_weights.get(fsym, 0)
                if w > 0 and price and price > 0 and is_stock(p.yf_ticker.tickers[fsym]):
                    symbols.append(sym)
                    markets.append(mkt)
                    full_symbols.append(fsym)
                    prices.append(price)

            if not symbols:
                print("Error: No valid investable assets found")
                return

            # Renormalize weights for the filtered set
            filtered_weights = {f: target_weights[f] for f in full_symbols}
            total_w = sum(filtered_weights.values())
            filtered_weights = {f: w / total_w for f, w in filtered_weights.items()}

            latest_prices = pd.Series(dict(zip(full_symbols, prices)))
            da = DiscreteAllocation(filtered_weights, latest_prices, total_portfolio_value=total_value)
            allocation, leftover = da.greedy_portfolio()

            print(f"\n{title} Allocation (Total: ${total_value:.2f}, Leftover: ${leftover:.2f}):")
            fsym_to_info = dict(zip(full_symbols, zip(symbols, markets, prices)))
            out_symbols, out_markets, out_quantities = [], [], []
            for fsym, quantity in allocation.items():
                sym, mkt, price = fsym_to_info[fsym]
                out_symbols.append(sym)
                out_markets.append(mkt)
                out_quantities.append(quantity)
                print(f"  {sym}: {quantity} shares @ ${price:.2f} = ${quantity * price:.2f}")

            if not out_symbols:
                print("Error: No shares allocated")
                return

            optimal_portfolio = Portfolio(
                out_symbols, out_markets, out_quantities,
                risk_free_rate=p.risk_free_rate,
                existing_tickers=p.tickers_dictionary,
                use_past_growth=p.use_past_growth
            )
            optimal_portfolio.calculate_correlation()

            optimal_gui = PortfolioGui(optimal_portfolio, show_frontier=True)
            optimal_gui.setWindowTitle(title)
            if not hasattr(self, '_portfolio_windows'):
                self._portfolio_windows = []
            self._portfolio_windows.append(optimal_gui)
            optimal_gui.show()

        except Exception as e:
            print(f"Error creating portfolio: {e}")
            import traceback
            traceback.print_exc()



if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    from qt_material import apply_stylesheet
    import sys

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_red.xml')

    portfolio = Portfolio(["msft", "aapl", "nvda", "googl", "brk.b"], 
                          ["nasdaq", "nasdaq", "nasdaq", "nasdaq", "nyse"], [10, 5, 3, 4, 2])
    portfolio.calculate_correlation()
    gui = PortfolioGui(portfolio, show_frontier=True)
    gui.show()
    sys.exit(app.exec_())
