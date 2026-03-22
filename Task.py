"""
Task.py
=======================
Main runner script:
  1. Fetches 4-month daily price history for 9 Nifty50 stocks + Gold via yfinance
  2. Computes annualised expected returns and covariance matrix
  3. Takes user input for target annual return (%)
  4. Calls portfolio_optimizer.py → which calls wolfe_qp_solver.py
  5. Displays and plots results

The 9 Nifty50 stocks are hardcoded for reproducibility. Gold is fixed
as the 10th asset (GOLDBEES.NS — Gold BeES ETF on NSE).

Dependencies
------------
    QuadraticSimplex.py      (Wolfe's Simplex QP backend)
    PortfolioOptimizer.py  (Markowitz optimiser interface)
    yfinance, numpy, pandas, matplotlib
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ── Locate sibling modules ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PortfolioOptimizer import (
    PortfolioInput,
    optimise_portfolio,
    efficient_frontier_wolfe,
)


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

NIFTY50_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "WIPRO.NS",
]

GOLD_TICKER     = "GOLDBEES.NS"
MONTHS_HISTORY  = 4
TRADING_DAYS    = 252   # annualisation factor


# ═══════════════════════════════════════════════════════════════════
#  STEP 1: FETCH PRICES
# ═══════════════════════════════════════════════════════════════════

def fetch_prices(tickers: list[str], months: int) -> pd.DataFrame:
    """
    Download adjusted-close prices for the last `months` months.
    Forward-fills gaps and drops any remaining NaNs.
    """
    end   = datetime.today()
    start = end - timedelta(days=int(months * 30.5))
    print(f"\n  Fetching {months}-month price history  "
          f"({start.date()} → {end.date()}) …")

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])

    raw = raw.reindex(columns=tickers)
    raw = raw.bfill().ffill().dropna()
    print(f"  Got {len(raw)} trading days for {len(tickers)} assets.")
    return raw


# ═══════════════════════════════════════════════════════════════════
#  STEP 2: COMPUTE STATISTICS  (in PERCENTAGE form)
# ═══════════════════════════════════════════════════════════════════

def compute_stats_pct(prices: pd.DataFrame, trading_days: int = TRADING_DAYS):
    """
    Returns
    -------
    r_pct     : annualised expected returns, in PERCENT   (array, n)
    Sigma_pct2: annualised covariance matrix, in (%²)     (array, n×n)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()

    r_dec      = log_ret.mean().values  * trading_days          # decimal
    Sigma_dec2 = log_ret.cov().values   * trading_days          # decimal^2

    r_pct      = r_dec      * 100.0                             # percent
    Sigma_pct2 = Sigma_dec2 * 10_000.0                         # (%²)

    return r_pct, Sigma_pct2


# ═══════════════════════════════════════════════════════════════════
#  STEP 3: USER INPUT
# ═══════════════════════════════════════════════════════════════════

def get_target_return(r_lo_pct: float, r_hi_pct: float) -> float:
    print(f"\n  Feasible return range: [{r_lo_pct:.2f}%, {r_hi_pct:.2f}%]")
    while True:
        try:
            T = float(input("  Enter target annual return (%): ").strip())
            return T
        except ValueError:
            print("  ✗  Please enter a numeric value.")


# ═══════════════════════════════════════════════════════════════════
#  STEP 4: PLOTTING
# ═══════════════════════════════════════════════════════════════════

PALETTE = [
    "#00f5c4", "#ffd166", "#ff4d6d", "#a29bfe",
    "#74b9ff", "#fd79a8", "#55efc4", "#fdcb6e",
    "#e17055", "#b2bec3",
]
BG    = "#0b0b12"
PANEL = "#13131f"
MUTED = "#4a4a6a"
WHITE = "#e8e8f4"


def plot_results(labels, r_pct, Sigma_pct2, result, T_pct, ef):
    """Draw a three-panel result figure and save to outputs."""
    fig = plt.figure(figsize=(20, 7), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1 : Efficient Frontier ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PANEL)

    ax1.plot(ef["stds_pct"], ef["returns_pct"],
             color="#00f5c4", lw=2.5, label="Efficient Frontier", zorder=3)

    asset_stds = np.sqrt(np.diag(Sigma_pct2))
    for i, (lbl, col) in enumerate(zip(labels, PALETTE)):
        ax1.scatter(asset_stds[i], r_pct[i],
                    color=col, s=80, zorder=5,
                    edgecolors=BG, linewidths=0.8, label=lbl)

    ax1.scatter(result.portfolio_std_pct, result.portfolio_return_pct,
                color=WHITE, s=220, marker="*", zorder=6,
                label=f"Optimal  (T={T_pct:.1f}%)",
                edgecolors="#00f5c4", linewidths=1)

    ax1.set_title("Efficient Frontier", color=WHITE, fontsize=13, pad=10)
    ax1.set_xlabel("Std Dev (%)", color=MUTED, fontsize=10)
    ax1.set_ylabel("Expected Return (%)", color=MUTED, fontsize=10)
    ax1.tick_params(colors=MUTED)
    for sp in ax1.spines.values(): sp.set_edgecolor(MUTED)
    ax1.legend(fontsize=7, labelcolor=WHITE, facecolor=PANEL,
               edgecolor=MUTED, framealpha=0.8, loc="upper left")
    ax1.grid(color=MUTED, alpha=0.15, linestyle="--")

    # ── Panel 2 : Optimal Weights Bar Chart ──────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PANEL)

    w_pct = result.weights_pct
    sorted_idx  = np.argsort(w_pct)[::-1]
    sorted_lbls = [labels[i]    for i in sorted_idx]
    sorted_w    = [w_pct[i]     for i in sorted_idx]
    sorted_cols = [PALETTE[i]   for i in sorted_idx]

    bars = ax2.bar(sorted_lbls, sorted_w, color=sorted_cols,
                   edgecolor=BG, linewidth=0.6, zorder=3)
    for bar, val in zip(bars, sorted_w):
        if val > 0.5:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom",
                     color=WHITE, fontsize=8, fontweight="bold")

    ax2.set_title(f"Optimal Weights  (T = {T_pct:.1f}%)",
                  color=WHITE, fontsize=13, pad=10)
    ax2.set_ylabel("Weight (%)", color=MUTED, fontsize=10)
    ax2.tick_params(axis="x", colors=MUTED, rotation=45, labelsize=8)
    ax2.tick_params(axis="y", colors=MUTED)
    for sp in ax2.spines.values(): sp.set_edgecolor(MUTED)
    ax2.grid(color=MUTED, alpha=0.15, linestyle="--", axis="y")
    ax2.set_ylim(0, max(sorted_w) * 1.25 if sorted_w else 100)

    # ── Panel 3 : Weight Evolution Along Frontier ────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(PANEL)

    ef_w = ef["weights_pct"]   # shape (n_points, n)
    for i, (lbl, col) in enumerate(zip(labels, PALETTE)):
        if i < ef_w.shape[1]:
            ax3.plot(ef["returns_pct"], ef_w[:, i],
                     color=col, lw=1.8, label=lbl)

    ax3.axvline(result.portfolio_return_pct, color=WHITE, lw=1.2,
                linestyle="--", alpha=0.7, label=f"T = {T_pct:.1f}%")
    ax3.set_title("Allocation Along Frontier", color=WHITE, fontsize=13, pad=10)
    ax3.set_xlabel("Target Return (%)", color=MUTED, fontsize=10)
    ax3.set_ylabel("Weight (%)", color=MUTED, fontsize=10)
    ax3.tick_params(colors=MUTED)
    for sp in ax3.spines.values(): sp.set_edgecolor(MUTED)
    ax3.legend(fontsize=7, labelcolor=WHITE, facecolor=PANEL,
               edgecolor=MUTED, framealpha=0.8, loc="upper right")
    ax3.grid(color=MUTED, alpha=0.15, linestyle="--")

    fig.suptitle(
        "Markowitz Portfolio Optimisation  |  Nifty50 + Gold  "
        f"(Wolfe QP, {MONTHS_HISTORY}-month data)",
        color=WHITE, fontsize=14, fontweight="bold", y=1.01,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "nifty_gold_portfolio.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Plot saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Nifty50 + Gold Portfolio Optimiser")
    print("  Backend: Wolfe's Modified Simplex QP Solver")
    print("=" * 62)

    all_tickers = NIFTY50_TICKERS + [GOLD_TICKER]
    labels = [t.replace(".NS", "").replace("GOLDBEES", "Gold")
              for t in all_tickers]
    labels[-1] = "Gold"

    # ── 1. Fetch prices ──────────────────────────────────────────────
    prices = fetch_prices(all_tickers, MONTHS_HISTORY)

    # Drop tickers with insufficient data
    valid = prices.notna().all()
    bad   = [all_tickers[i] for i, ok in enumerate(valid) if not ok]
    if bad:
        print(f"\n  ⚠  Dropping tickers with missing data: {bad}")
        prices = prices.loc[:, valid]
        all_tickers = list(prices.columns)
        labels = [t.replace(".NS", "") for t in all_tickers]
        if "GOLDBEES.NS" in all_tickers:
            labels[all_tickers.index("GOLDBEES.NS")] = "Gold"

    if len(all_tickers) == 0:
        print("\n  ✗  No valid asset data available after dropping incomplete histories. Exiting.")
        sys.exit(1)

    n = len(all_tickers)

    # ── 2. Compute statistics in percent ─────────────────────────────
    r_pct, Sigma_pct2 = compute_stats_pct(prices)

    print("\n  Individual asset statistics (annualised, 4-month window):")
    print(f"\n  {'Asset':<16} {'Exp. Return':>13} {'Std Dev':>10}")
    print("  " + "─" * 42)
    for lbl, ri, si in zip(labels, r_pct, np.sqrt(np.diag(Sigma_pct2))):
        print(f"  {lbl:<16} {ri:>12.4f}%  {si:>9.4f}%")

    # ── 3. Target return ─────────────────────────────────────────────
    T_pct = get_target_return(r_pct.min(), r_pct.max())

    # ── 4. Build input and solve ─────────────────────────────────────
    inp = PortfolioInput(
        expected_returns_pct = r_pct.tolist(),
        cov_matrix_pct2      = Sigma_pct2.tolist(),
        target_return_pct    = T_pct,
        asset_names          = labels,
        cov_in_decimal       = False,    # already in (%²)
    )

    print("\n  Solving QP via Wolfe's Modified Simplex …")
    result = optimise_portfolio(inp)

    if not result.success:
        print(f"\n  ✗  Optimisation failed: {result.message}")
        sys.exit(1)

    result.print_summary()

    # ── 5. Efficient frontier ────────────────────────────────────────
    print("\n  Tracing efficient frontier (50 points) …")
    ef = efficient_frontier_wolfe(
        expected_returns_pct = r_pct.tolist(),
        cov_matrix_pct2      = Sigma_pct2.tolist(),
        asset_names          = labels,
        cov_in_decimal       = False,
        n_points             = 50,
    )
    print(f"  Frontier traced: {len(ef['returns_pct'])} feasible points.")

    # ── 6. Plot ──────────────────────────────────────────────────────
    plot_results(labels, r_pct, Sigma_pct2, result, T_pct, ef)

    # ── 7. Summary table ─────────────────────────────────────────────
    print("\n  WEIGHT ARRAY  (pass directly to allocation system):")
    print("  weights = [")
    for lbl, wpct in zip(result.asset_names, result.weights_pct):
        print(f"      {wpct:7.4f},   # {lbl}")
    print("  ]  # sum = {:.4f}%".format(result.weights_pct.sum()))
    print("\n  Done.")


if __name__ == "__main__":
    main()