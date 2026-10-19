"""
Alpha UK Equity Fund — mock portfolio analysis

Builds an equally weighted 10-stock UK equity portfolio, compares it with a FTSE 100
total-return benchmark, and reports risk metrics and stock-level contribution.
Findings for the Oct 2025 to Oct 2026 period are summarised at the bottom of this file.

Requirements: pip install yfinance pandas numpy matplotlib

"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Settings ────────────────────────────────────────────────────
START = "2025-10-14"          # first day of the analysis window
END = "2026-10-14"            # yfinance treats END as exclusive
RISK_FREE = 0.04              # annual risk-free rate for Sharpe/alpha; update to current Bank Rate
TRADING_DAYS = 252

stocks = {
    "Greggs":      "GRG.L",
    "Unilever":    "ULVR.L",
    "AstraZeneca": "AZN.L",
    "HSBC":        "HSBA.L",
    "BP":          "BP.L",
    "Rolls Royce": "RR.L",
    "Tesco":       "TSCO.L",
    "Rightmove":   "RMV.L",
    "JD Sports":   "JD.L",
    "GSK":         "GSK.L",
}
weights = pd.Series({name: 1 / len(stocks) for name in stocks})   # equal weight

# Total-return benchmark: iShares Core FTSE 100 UCITS ETF (Acc). The accumulating
# share class reinvests dividends, so its price is a total-return series.
# (The distributing class, ISF.L, was not dividend-adjusted by Yahoo and gave a
# figure almost identical to the price index.) Fallback: "VUKG.L".
BENCHMARK_TICKER = "CUKX.L"
BENCHMARK_NAME = "FTSE 100 (TR)"
PRICE_INDEX_TICKER = "^FTSE"   # kept only to show the size of the dividend effect


# ── 2. Download prices ─────────────────────────────────────────────
def load_prices():
    tickers = list(stocks.values()) + [BENCHMARK_TICKER, PRICE_INDEX_TICKER]
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                      progress=False)["Close"]
    names = {v: k for k, v in stocks.items()}
    names[BENCHMARK_TICKER] = BENCHMARK_NAME
    names[PRICE_INDEX_TICKER] = "FTSE 100 (price only)"
    raw = raw.rename(columns=names)

    missing = [c for c in names.values() if c not in raw or raw[c].isna().all()]
    if missing:
        raise SystemExit(f"No data downloaded for: {missing}. Check the tickers.")

    # Align dates: carry forward the last price over occasional gaps,
    # then drop any leading rows where something hasn't started trading.
    return raw.ffill().dropna()


def analyse(prices):
    stock_cols = list(stocks)
    bench = prices[BENCHMARK_NAME]

    # ── 3. Returns ────────────────────────────────────────────────
    growth = prices / prices.iloc[0]                     # value of £1 invested on day one
    stock_total = growth[stock_cols].iloc[-1] - 1        # total return per stock
    bench_total = bench.iloc[-1] / bench.iloc[0] - 1
    price_index_total = (prices["FTSE 100 (price only)"].iloc[-1]
                         / prices["FTSE 100 (price only)"].iloc[0] - 1)

    # Buy-and-hold portfolio value (weights drift with prices)
    port_value = (growth[stock_cols] * weights).sum(axis=1)
    port_total = port_value.iloc[-1] - 1

    # Daily-rebalanced version (what the original script calculated), for comparison
    daily = prices.pct_change().dropna()
    rebal_total = (1 + (daily[stock_cols] * weights).sum(axis=1)).prod() - 1

    port_daily = port_value.pct_change().dropna()
    bench_daily = bench.pct_change().dropna()

    # ── 4. Risk metrics ───────────────────────────────────────────
    years = len(port_daily) / TRADING_DAYS
    ann = lambda total: (1 + total) ** (1 / years) - 1
    port_ann, bench_ann = ann(port_total), ann(bench_total)

    port_vol = port_daily.std() * np.sqrt(TRADING_DAYS)
    bench_vol = bench_daily.std() * np.sqrt(TRADING_DAYS)
    beta = port_daily.cov(bench_daily) / bench_daily.var()
    sharpe = (port_ann - RISK_FREE) / port_vol
    bench_sharpe = (bench_ann - RISK_FREE) / bench_vol
    jensen_alpha = port_ann - (RISK_FREE + beta * (bench_ann - RISK_FREE))
    tracking_error = (port_daily - bench_daily).std() * np.sqrt(TRADING_DAYS)
    info_ratio = (port_ann - bench_ann) / tracking_error

    def max_drawdown(value):
        return (value / value.cummax() - 1).min()

    # ── 5. Print summary ─────────────────────────────────────────
    print(f"\nPeriod: {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(port_daily)} trading days)")

    print("\n── PERFORMANCE SUMMARY ──")
    print(f"{'Stock':<22}{'Return':>10}")
    print("-" * 32)
    for s in stock_total.sort_values(ascending=False).index:
        print(f"{s:<22}{stock_total[s]:>10.1%}")
    print("-" * 32)
    print(f"{'Portfolio (buy & hold)':<22}{port_total:>10.1%}")
    print(f"{BENCHMARK_NAME:<22}{bench_total:>10.1%}")
    print(f"\nExcess return vs {BENCHMARK_NAME}: {port_total - bench_total:+.1%}")
    print(f"(Daily-rebalanced portfolio: {rebal_total:.1%}. "
          f"FTSE 100 price index without dividends: {price_index_total:.1%})")

    print("\n── RISK ──")
    rows = [
        ("Annualised volatility", f"{port_vol:.1%}", f"{bench_vol:.1%}"),
        ("Sharpe ratio", f"{sharpe:.2f}", f"{bench_sharpe:.2f}"),
        ("Maximum drawdown", f"{max_drawdown(port_value):.1%}",
         f"{max_drawdown(bench / bench.iloc[0]):.1%}"),
        ("Beta", f"{beta:.2f}", "1.00"),
        ("Jensen's alpha", f"{jensen_alpha:+.1%}", "-"),
        ("Tracking error", f"{tracking_error:.1%}", "-"),
        ("Information ratio", f"{info_ratio:.2f}", "-"),
    ]
    print(f"{'Metric':<24}{'Portfolio':>12}{'Benchmark':>12}")
    print("-" * 48)
    for r in rows:
        print(f"{r[0]:<24}{r[1]:>12}{r[2]:>12}")
    print(f"(Risk-free rate assumed: {RISK_FREE:.1%})")

    # ── 6. Contribution analysis ─────────────────────────────────
    contrib = weights * stock_total                       # adds up to portfolio return
    active = weights * (stock_total - bench_total)        # adds up to excess return
    table = pd.DataFrame({"Return": stock_total, "Weight": weights,
                          "Contribution": contrib, "Active contribution": active}
                         ).sort_values("Contribution", ascending=False)

    print("\n── CONTRIBUTION ANALYSIS ──")
    print(f"{'Stock':<15}{'Return':>9}{'Weight':>9}{'Contrib.':>11}{'vs bench':>11}")
    print("-" * 55)
    for s, r in table.iterrows():
        print(f"{s:<15}{r['Return']:>9.1%}{r['Weight']:>9.0%}"
              f"{r['Contribution']:>11.2%}{r['Active contribution']:>+11.2%}")
    print("-" * 55)
    print(f"{'Total':<15}{'':>9}{weights.sum():>9.0%}"
          f"{table['Contribution'].sum():>11.2%}{table['Active contribution'].sum():>+11.2%}")
    print(f"\nTop contributor:  {table.index[0]} ({table['Return'].iloc[0]:.1%})")
    print(f"Worst detractor:  {table.index[-1]} ({table['Return'].iloc[-1]:.1%})")
    print("Note: the benchmark is the current FTSE 100 tracker; index membership "
          "changes during the period (e.g. reshuffles) are not modelled.")

    # ── 7. Chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in stock_cols:
        ax.plot(growth.index, (growth[s] - 1) * 100,
                color="lightgrey", linewidth=0.8, alpha=0.7)
    ax.plot(port_value.index, (port_value - 1) * 100, color="#1a3a6b",
            linewidth=2.5, label="Portfolio (buy & hold)")
    ax.plot(bench.index, (bench / bench.iloc[0] - 1) * 100, color="#e87722",
            linewidth=2.5, label=BENCHMARK_NAME)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Portfolio vs {BENCHMARK_NAME}: {prices.index[0]:%d %b %Y} to "
                 f"{prices.index[-1]:%d %b %Y}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative total return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("portfolio_performance.png", dpi=150)
    print("\nChart saved as portfolio_performance.png")
    plt.show()


if __name__ == "__main__":
    print("Downloading price data...")
    analyse(load_prices())


# ── FINDINGS: 14 Oct 2025 – 13 Oct 2026 ─────────────────────────────
#
# Results (buy and hold, dividends included on both sides)
#   Portfolio return     12.1%      FTSE 100 total return   16.7%
#   Excess return        -4.6 percentage points
#   Volatility           12.4%      (benchmark 11.7%)
#   Sharpe ratio         0.65       (benchmark 1.08)
#   Beta                 0.87       Tracking error  7.3%
#   Jensen's alpha       -2.9%      Information ratio  -0.62
#   Maximum drawdown     -8.8%      (benchmark -8.9%)
#
# Why the portfolio lagged
#   BP and HSBC added about 9 points of return between them, but with equal weights
#   six holdings lagged the index (Rightmove, JD Sports, AstraZeneca, Unilever, GSK,
#   Tesco) and together cost more than the winners added. Beta was below 1, yet
#   volatility was above the index: much of the risk was stock-specific, which is the
#   cost of holding only ten names.
#
# Top contributor: BP (+46.6%, 4.7 points of portfolio return)
#   - Oil price spike: crude rose sharply after the US-Iran conflict began at the end
#     of February 2026. BP gained more than the other supermajors over that period.
#   - Trading and refining: BP reported "exceptional" oil trading in Q1 2026, and Q2
#     profit more than doubled on higher prices, refining margins and trading.
#   - Strategy reset: since 2025 BP has cut renewables spending and refocused on oil
#     and gas under pressure from activist investor Elliott, with new CEO Meg O'Neill
#     from April 2026.
#   - Note: BP suspended its share buyback in February 2026 to cut debt, so the gain
#     came from earnings and oil prices, not capital returns.
#
# Second contributor: HSBC (+43.7%, 4.4 points)
#   - Interest rates: a supportive rate backdrop kept net interest income high;
#     HSBC raised its 2026 net interest income guidance to above $46bn.
#   - Asian wealth: wealth revenue grew 18% in H1 2026, led by Asia.
#   - Results beat: H1 2026 pre-tax profit rose 23% to $19.5bn, ahead of forecasts,
#     and buybacks restarted after a pause for the Hang Seng Bank privatisation.
#
# Worst detractor: Rightmove (-26.4%, -2.6 points; -4.3 points vs the index)
#   - Profit downgrade: in November 2025 Rightmove cut its 2026 profit growth outlook
#     to 3-5% to fund AI investment, and the shares fell more than 25% in a day.
#   - Index exit: after a prolonged decline it dropped from the FTSE 100 to the
#     FTSE 250 in June 2026, which can force selling by index funds.
#   - Sentiment: activist pressure, a large legal claim, competition from CoStar's
#     OnTheMarket and high mortgage rates weighed on the shares.
#
# What I would change
#   - Hold more stocks (30+) or weight by risk to reduce stock-specific risk.
#   - Make sector bets relative to FTSE 100 weights deliberately, and add
#   - Test over a longer period: one year is too short to judge stock selection.
