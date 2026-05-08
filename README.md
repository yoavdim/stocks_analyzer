# Stock & Portfolio Analyzer

A quantitative stock analysis and portfolio optimization tool.

Are you sick of investing according to vibes alone? Ever wanted to incorporate solid economic & mathematical principles into your portfolio, but reading financial statements is just overwhelming? If so, then this might be just the tool you always wanted!

Our tool is here to help you with all the steps in portfolio creation:
- Throw in a bunch of stocks you think are interesting
- Screen out any with a sketchy balance sheet or that are just overvalued
- Get into more manual inspection and price target determination with our NPV calculator and financial graphs
- At the end, find the right portfolio composition to minimize risk and correlation while maintaining high growth
- Keep track of your past investment performance

Our tool fetches live data, computes 40+ financial metrics per stock, and provides portfolio-level analysis including efficient frontier optimization, CAPM modeling, and DCF valuation.

![Portfolio Analyzer](images/portfolio%20analyzer%20example.png)
![Stock Screener](images/screener%20example.png)
![DCF Calculator](images/npv%20calculator%20example.png)

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)
  - [Stock Screening](#stock-screening)
  - [Portfolio Analysis](#portfolio-analysis)
  - [Portfolio Optimization](#portfolio-optimization)
  - [DCF Valuation Calculator](#dcf-valuation-calculator)
- [Mathematical Models](#mathematical-models)
  - [Growth Estimation — Log-Linear Regression](#growth-estimation--log-linear-regression)
  - [Intrinsic Valuation — DCF & NPV](#intrinsic-valuation--dcf--npv)
  - [Risk & Return — CAPM](#risk--return--capm)
  - [Portfolio Theory — Covariance & Efficient Frontier](#portfolio-theory--covariance--efficient-frontier)
  - [Optimization — Maximum Sharpe Ratio](#optimization--maximum-sharpe-ratio)
  - [Bond Valuation — Yield to Maturity](#bond-valuation--yield-to-maturity)
  - [CAPM & Portfolio Visualization](#capm--portfolio-visualization)
- [Architecture](#architecture)
- [Input Formats](#input-formats)
- [Dependencies](#dependencies)
- [Future Directions](#future-directions)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Authors](#authors)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

This installs the package in editable mode — any code changes take effect immediately without reinstalling.

## Usage

**Portfolio analyzer** (main app — performance tracking, optimization, benchmark comparison):
```bash
portfolio-analyzer
```

**Stock screener** (batch analysis, CSV export, filterable table):
```bash
stocks-analyzer
```

**DCF valuation calculator** (interactive per-stock valuation):
```bash
npv-calculator
```

A helper script `analyzer.sh` is included that creates the venv, installs dependencies, and launches the portfolio analyzer in one step:
```bash
./analyzer.sh
```


## Features

### Stock Screening
- Analyze stocks across 12+ global exchanges (NASDAQ, NYSE, TPE, TYO, LON, AMS, STO, TLV, KRX, and more)
- Compute valuation metrics (P/E, P/B, PEG, ROE, ROA, debt-to-equity, current ratio, ...)
- Health scoring with configurable filters: *healthy*, *overvalued*, *leveraged* — conditions are defined in `npv_config.json` and can be customized without code changes
- Multi-threaded fetching with 30-day ticker cache
- Export to CSV (full and TLDR versions)
- Sortable, filterable screener GUI with color-coded health indicators
- Double-click any ticker in the screener table to open its DCF calculator
- NPV/IRR assumptions (discount rate, growth durations, perpetuity cap, etc.) are configurable via `npv_config.json`

![Stock Screener](images/screener%20example.png)

### Portfolio Analysis
- Track portfolio performance with buy/sell transaction history
- Compute annualized IRR and compare against a benchmark index (NASDAQ-100)
- Sector and industry allocation via concentric pie charts (double-click a ticker to open its DCF calculator)
- Per-ticker and portfolio-level beta calculation
- Weighted P/E and ROE

### Portfolio Optimization
- **Efficient Frontier** via mean-variance optimization (PyPortfolioOpt)
- **Tangency portfolio** (maximum Sharpe ratio) with Capital Allocation Line
- **Minimum-variance portfolio** — less sensitive to growth forecast errors than the tangency portfolio, since it depends only on the covariance structure
- Discrete share allocation from optimal weights
- Risk-free rate sourced live from the 10-Year Treasury (^TNX)

![Efficient Frontier](images/portfolio%20analyzer%20example.png)

### DCF Valuation Calculator
- Interactive per-stock DCF model with configurable parameters:
  - Growth trend (linear / exponential)
  - Growth benchmark (earnings, book value, revenue, FCF, or custom)
  - Perpetuity model (none, constant, slow exponent)
  - Discount rate
- Real-time price target and IRR output
- CAPM reference display (β, risk-free rate, market return)
- Interactive price graph with growth calculation on selected ranges
- **Save/Load DCF models** — save your valuation assumptions per ticker. Saved models are automatically used in the screener's IRR calculation. The growth phase shortens over time (uses an absolute end date), and stale models (>6 months) trigger a warning.

![DCF Calculator](images/npv%20calculator%20example.png)

## Mathematical Models

The tool chains several quantitative models together — from estimating individual stock growth, through intrinsic valuation, to portfolio-level risk assessment and optimization.

### Growth Estimation — Log-Linear Regression

Historical growth rates (earnings, revenue, book value) are estimated by fitting a linear regression on the log-transformed time series:

$$\ln(y_t) = a + b \cdot t$$

The annualized growth rate is then recovered as $g = e^b - 1$. This assumes exponential (compound) growth, which is more realistic for financial data than a linear trend. A linear trend (slope of a degree-1 polynomial fit) is also computed for short-term forecasting.

### Intrinsic Valuation — DCF & NPV

The Discounted Cash Flow model projects future cash flows using the estimated growth rate and discounts them back to present value:

$$V = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t} + \frac{CF_T \cdot (1+g)}{(r - g)(1+r)^T} + BV$$

where $CF_t$ is the projected cash flow at year $t$, $r$ is the discount rate, $g$ is the terminal perpetuity growth rate, and $BV$ is the book value per share. The Internal Rate of Return (IRR) is found by numerically solving for the rate $r^*$ such that:

$$\text{NPV}(r^*) = \text{Market Price}$$

This gives the annualized return implied by the current price under the DCF assumptions — effectively answering "what growth rate is the market pricing in?" When all projected cash flows are positive (positive FCF with non-decreasing growth), the NPV is monotonically decreasing in the discount rate, so the root is found efficiently using Brent's method (`scipy.optimize.brentq`). For cases where monotonicity isn't guaranteed (e.g. linear declining growth, or portfolio IRR with mixed buy/sell transactions), a linear scan is used as a fallback.

The same IRR approach is used at the portfolio level to measure historic performance: given all buy/sell transactions and the current portfolio value, solve for the single discount rate that zeroes out the net present value of all cash flows.

### Risk & Return — CAPM

Each stock's expected return is modeled using the Capital Asset Pricing Model:

$$E[R_i] = R_f + \beta_i \left( E[R_m] - R_f \right)$$

where $R_f$ is the 10-Year Treasury yield, $E[R_m]$ is the S&P 500 trailing return, and $\beta_i$ is sourced from Yahoo Finance or computed as a fallback from the return series:

$$\beta_i = \frac{\text{Cov}(R_i,\, R_m)}{\text{Var}(R_m)}$$

The CAPM-derived discount rate is used as an alternative to a fixed discount rate in the DCF model, producing a CAPM-adjusted intrinsic value.

### Portfolio Theory — Covariance & Efficient Frontier

Portfolio risk is computed from the covariance matrix of monthly asset returns:

$$\sigma_p = \sqrt{\mathbf{w}^\top \, \Sigma \, \mathbf{w}}$$

where $\Sigma$ is the annualized sample covariance matrix (monthly covariance × 12). The portfolio's expected return is the weighted sum of individual forecasts: $E[R_p] = \mathbf{w}^\top \mathbf{\mu}$.

The efficient frontier traces the set of portfolios that minimize $\sigma_p$ for each level of $E[R_p]$.

### Optimization — Maximum Sharpe Ratio

The optimal (tangency) portfolio is found by maximizing the Sharpe ratio:

$$\max_{\mathbf{w}} \; \frac{E[R_p] - R_f}{\sigma_p}$$

This portfolio lies at the tangent point of the Capital Allocation Line (CAL) with the efficient frontier. Its beta follows from linearity of covariance:

$$\beta_p = \sum_i w_i \, \beta_i$$

Both the current portfolio and the tangency portfolio are plotted on the CAPM Security Market Line for comparison.

### Bond Valuation — Yield to Maturity

For fixed-income instruments, yield-to-maturity $y$ is solved numerically from:

$$P = \sum_{t} \frac{C}{(1+y)^t} + \frac{F + C}{(1+y)^T}$$

### CAPM & Portfolio Visualization

> **A note on growth estimates:** The main limitation of the portfolio and screener analysis is that future growth is inherently unpredictable. The NPV parameters used for batch analysis are necessarily arbitrary — unlike the interactive DCF calculator where you control every assumption. For this reason the efficient frontier and CAPM graphs currently use past performance as the y-axis. IRR would be a better metric for quantitative analysis, but it relies on the same uncertain growth forecasts. Until there is a reliable way to estimate future returns, historical performance is the more honest default.

The bottom graph plots every asset and the portfolio on the CAPM plane — beta (systematic risk) on the x-axis, expected return on the y-axis. The Security Market Line (SML) connects the risk-free rate (10-Year Treasury) at β = 0 to the S&P 500 market return at β = 1. Assets above the SML are outperforming their risk-adjusted expectation; assets below it are underperforming. The current portfolio, the tangency (maximum Sharpe) portfolio, and the minimum-variance portfolio are each marked so you can see at a glance how your allocation compares to the theoretical optimum.

![CAPM Graph](images/portfolio%20analyzer%20example.png)

## Architecture

```
stocks_analyzer.py     # Batch analysis, CSV export, screener entry point
portfolio_analyzer.py  # Portfolio performance, IRR, benchmark comparison
npv_calculator.py      # Interactive DCF valuation GUI
ticker_gui.py          # Stock screener GUI
ticker.py              # Ticker class, TickerGroup, CAPM, efficient frontier
portfolio.py           # Portfolio class, optimization, portfolio GUI
yfinance_info.py       # Yahoo Finance data layer
fx_converter.py        # Multi-currency FX conversion (wraps yfinance)
reports.py             # MSN Money financial statement scraping
yahoo_reports.py       # Yahoo Finance financial statement scraping
bonds.py               # Bond yield-to-maturity calculator
npv_config.json        # All configurable parameters (NPV, filters, base currency)
dcf_models/            # Saved per-ticker DCF valuation parameters (JSON)
gui/ticker_table.py    # Table widget with health-colored rows
```

## Input Formats

**Stock list** (`inputs/my_stocks.txt`):
```
# Comments start with #
MSFT NASDAQ
BRK.B NYSE
2330 TPE
```

**Portfolio transactions** (`inputs/Portfolio_indices_example.tsv`):
```
Ticker  Market  Date        Amount  Action  Cost
QCOM    NASDAQ  2022-01-01  1       BUY         141.6
MSFT    NASDAQ  2020-10-05  2       BUY         100
```

Costs should be in the local trading currency of the exchange (USD for NYSE/NASDAQ, agora for TLV, CAD for TSE, etc.). All values are automatically converted to a common base currency internally. The base currency is configurable via `base_currency` in `npv_config.json` (default: USD).

Not all columns are required. The tool adapts to the information provided:

| Columns present                                        | Available features                                                                                                   |
| --------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------|
| `Ticker`, `Market`, `Date`, `Amount`, `Action`, `Cost` | Full analysis: portfolio optimization, historic IRR, benchmark comparison, per-ticker performance                    |
| `Ticker`, `Market`, `Amount`                           | Portfolio optimization with current holdings (efficient frontier, sector allocation, CAPM) — no historic performance |
| `Ticker`, `Market`                                     | Correlation analysis and efficient frontier with equal weights — no performance tracking                             |

## Dependencies

- **yfinance** — market data from Yahoo Finance
- **PyQt5** + **qt-material** — GUI framework with dark theme
- **PyPortfolioOpt** — mean-variance portfolio optimization
- **pandas**, **numpy**, **scipy** — numerical computation
- **matplotlib** — charting and visualization

## Future Directions

The core analysis pipeline is largely in place. The next steps are less about new features and more about finding data-driven ways to select the hyper-parameters of the program that currently rely on manual judgment. These are research directions rather than a concrete to-do list.

- [ ] Premium data backend for backtesting (financial reports beyond the 4-year yfinance limit)
- [ ] Basic support for viewing/using earning forecasts in the program (in dcf calculations)
- [ ] Optimism metric per stock — aggregate earnings forecasts and sentiment via LLM-powered web search to forecast deviations in future earnings growth relative to past performance
- [ ] ML-trained health/overvalued classification to improve on the rule-based quick filters (and for the rest of the config file parameters)
- [ ] AI-assisted NPV parameter selection based on ticker statistics and sector benchmarks
- [ ] Automatic screening / Optimal Portfolio Selection
- [ ] Agentic control interface — allow an AI agent to drive the program (research the available methods)

## Disclaimer

This tool is under active development, built mainly out of personal interest and curiosity. It is intended for educational and research purposes only and does not constitute financial advice. Always consult a qualified financial professional before making investment decisions.

## License

See [LICENSE](LICENSE).

## Authors

**Yoav Dim** · **Shay Agroskin**
