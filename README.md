
# UK Equity Portfolio Analysis

A Python tool that builds a 10-stock, equally weighted UK equity portfolio and measures its performance and risk against the FTSE 100.

![Portfolio vs FTSE 100](portfolio%20performance.png)

## What it does

- Downloads one year of daily prices for each holding using `yfinance`
- Tracks a buy-and-hold portfolio against a FTSE 100 **total-return** benchmark, so dividends are included on both sides
- Calculates volatility, Sharpe ratio, beta, Jensen's alpha, tracking error, information ratio and maximum drawdown
- Breaks down each stock's contribution to portfolio and relative return
- Plots cumulative returns for the portfolio, the benchmark and each holding

## Holdings

Greggs, Unilever, AstraZeneca, HSBC, BP, Rolls-Royce, Tesco, Rightmove, JD Sports and GSK, at 10% each, across banking, energy, pharma, consumer staples, retail, industrials and property technology.

## Results: 14 Oct 2025 – 13 Oct 2026

| Metric | Portfolio | FTSE 100 (TR) |
|---|---|---|
| Total return | 12.1% | 16.7% |
| Volatility | 12.4% | 11.7% |
| Sharpe ratio | 0.65 | 1.08 |
| Maximum drawdown | -8.8% | -8.9% |
| Beta | 0.87 | 1.00 |
| Tracking error | 7.3% | – |
| Information ratio | -0.62 | – |

**Top contributors:** BP (+46.6%), driven by the oil price spike after the US-Iran conflict began in February 2026, and HSBC (+43.7%), driven by strong net interest income and Asian wealth growth.

**Largest detractor:** Rightmove (-26.4%), after a profit downgrade to fund AI investment and its exit from the FTSE 100.

The portfolio lagged by 4.6 percentage points. Equal weighting meant six underperformers outweighed two strong winners. Volatility was above the index despite a beta below 1, showing high stock-specific risk from holding only ten names.

## Methodology lesson

My first version showed 2.4% outperformance. It compared dividend-adjusted stock prices with the price-only FTSE 100 index and assumed daily rebalancing. Correcting both reversed the conclusion.

## How to run

    pip install yfinance pandas numpy matplotlib
    python "uk equity portfolio analysis.py"

## Possible extensions

- Sector-level (Brinson) attribution against FTSE 100 weights
- Risk-based weighting instead of equal weights
- A longer test period, including stress periods such as 2022

*Educational project. Not investment advice.*
