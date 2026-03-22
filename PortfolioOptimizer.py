"""
PortfolioOptimizer.py
======================
Markowitz Mean-Variance Portfolio Optimiser
Uses Wolfe's QP solver as the backend.

Interface
---------
Returns WEIGHTS (fractions summing to 1, each in [0, 1]).
No budget is required — caller multiplies weights by their budget.
All returns are expressed as PERCENTAGES (e.g. 12.5 means 12.5% p.a.).

Problem solved
--------------
    min   w^T Σ w               (minimise portfolio variance)
    s.t.  sum(w)   = 1          (fully invested)
          r^T w   >= T          (meet or beat target return)
          w_i     >= 0          (long-only, no short selling)

where
    Σ   = annualised covariance matrix (decimal form internally)
    r   = annualised expected returns  (decimal form internally)
    T   = target return                (decimal form internally)

All inputs / outputs in PERCENTAGE form for convenience.

Dependencies
------------
    QuadraticSimplex.py
    numpy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

# ── Locate and import the Quadratic Simplex solver ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from QuadraticSimplex import solve_qp_wolfe, WolfeQPResult


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PortfolioInput:
    """
    All inputs expressed in PERCENTAGE form.

    Parameters
    ----------
    expected_returns_pct : list or array of length n
        Annualised expected return for each asset, in percent.
        e.g. [12.5, 8.3, 15.0, ...]
    cov_matrix_pct2 : 2-D list or array, shape (n, n)
        Annualised covariance matrix.
        Units: (percent)^2, i.e. if you computed it from decimal returns
        and multiplied by 10000, pass it here.
        Alternatively pass it in (decimal)^2 and set
        cov_in_decimal=True.
    target_return_pct : float
        Desired minimum annualised portfolio return, in percent.
        e.g. 10.0 means "I want at least 10% per year".
    asset_names : list of str, optional
        Labels for each asset. Defaults to ["A0", "A1", ...].
    cov_in_decimal : bool
        If True, the covariance matrix is in (decimal)^2 and will be
        converted internally to (percent)^2 by multiplying by 10000.
        Default: False (matrix already in (percent)^2).
    """
    expected_returns_pct : list
    cov_matrix_pct2      : list
    target_return_pct    : float
    asset_names          : list  = field(default_factory=list)
    cov_in_decimal       : bool  = False


@dataclass
class PortfolioResult:
    """
    All outputs expressed in PERCENTAGE form.

    Attributes
    ----------
    weights_pct : ndarray, shape (n,)
        Optimal weight for each asset as a percentage of total capital.
        Sums to 100. e.g. [30.0, 20.5, 49.5, ...].
    weights     : ndarray, shape (n,)
        Same weights as fractions in [0, 1], summing to 1.
    portfolio_return_pct : float
        Achieved annualised expected return (percent).
    portfolio_std_pct    : float
        Annualised portfolio standard deviation (percent).
    portfolio_variance_pct2 : float
        Annualised portfolio variance ((percent)^2).
    sharpe_ratio         : float
        Return / Std  (risk-free rate assumed 0).
    success  : bool
    message  : str
    asset_names : list of str
    """
    weights_pct              : Optional[np.ndarray] = None
    weights                  : Optional[np.ndarray] = None
    portfolio_return_pct     : Optional[float]      = None
    portfolio_std_pct        : Optional[float]      = None
    portfolio_variance_pct2  : Optional[float]      = None
    sharpe_ratio             : Optional[float]      = None
    success                  : bool                 = False
    message                  : str                  = ""
    asset_names              : list                 = field(default_factory=list)

    def print_summary(self):
        """Pretty-print the portfolio result."""
        w  = "=" * 62
        print(f"\n{w}")
        print("  OPTIMAL PORTFOLIO  (Wolfe QP Solver)")
        print(w)
        if not self.success:
            print(f"  ✗  Optimisation failed: {self.message}")
            print(w)
            return

        print(f"\n  {'Asset':<20} {'Weight (%)':>12}  {'Fraction':>10}")
        print("  " + "─" * 46)
        for name, wpct, wfrac in zip(
                self.asset_names, self.weights_pct, self.weights):
            print(f"  {name:<20} {wpct:>11.4f}%  {wfrac:>10.6f}")
        print("  " + "─" * 46)
        print(f"\n  Portfolio return   : {self.portfolio_return_pct:>10.4f}%")
        print(f"  Portfolio std dev  : {self.portfolio_std_pct:>10.4f}%")
        print(f"  Portfolio variance : {self.portfolio_variance_pct2:>10.4f} (%²)")
        print(f"  Sharpe ratio       : {self.sharpe_ratio:>10.4f}  (rf = 0)")
        print(w)


# ═══════════════════════════════════════════════════════════════════
#  CORE SOLVER
# ═══════════════════════════════════════════════════════════════════

def optimise_portfolio(inp: PortfolioInput) -> PortfolioResult:
    """
    Solve the Markowitz minimum-variance problem for a given target return.

    Parameters
    ----------
    inp : PortfolioInput
        See PortfolioInput docstring.  All values in PERCENTAGE form.

    Returns
    -------
    PortfolioResult
        Weights in percentage form (sum to 100) and fraction form (sum to 1).
    """
    # ── Parse and validate inputs ────────────────────────────────────
    r_pct   = np.array(inp.expected_returns_pct, dtype=float).ravel()
    Sig_raw = np.array(inp.cov_matrix_pct2,      dtype=float)
    T_pct   = float(inp.target_return_pct)
    n       = len(r_pct)

    names = list(inp.asset_names) if inp.asset_names else [f"A{i}" for i in range(n)]
    if len(names) != n:
        names = [f"A{i}" for i in range(n)]

    if Sig_raw.shape != (n, n):
        return PortfolioResult(
            success=False,
            message=f"Covariance matrix shape {Sig_raw.shape} != ({n},{n}).",
            asset_names=names,
        )

    # Convert cov to (percent)^2 if supplied in (decimal)^2
    Sigma_pct2 = Sig_raw * 10_000.0 if inp.cov_in_decimal else Sig_raw

    # Force symmetry and clip negative eigenvalues to ensure strictly PSD matrix
    Sigma_pct2 = 0.5 * (Sigma_pct2 + Sigma_pct2.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma_pct2)
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, 0.0)
        Sigma_pct2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Sigma_pct2 = 0.5 * (Sigma_pct2 + Sigma_pct2.T)  # Reforce symmetry after reconstruction

    # ── Feasibility clamp ────────────────────────────────────────────
    r_lo, r_hi = r_pct.min(), r_pct.max()
    if T_pct < r_lo:
        T_pct = r_lo + 1e-6
    elif T_pct > r_hi:
        T_pct = r_hi - 1e-6

    # ── Build QP in (percent) units ──────────────────────────────────
    #
    #   min   (1/2) w^T (2 Σ) w + 0^T w
    #   (Note: using Q = 2Σ so the objective is w^T Σ w)
    #
    #   Equality:    [1 1 … 1] w = 100   (weights sum to 100%)
    #   Inequality:  -r^T w <= -100·T_pct  (return >= T_pct, w sums to 100)
    #   Bounds:      w >= 0

    Q_qp = 2.0 * Sigma_pct2
    c_qp = np.zeros(n)

    A_eq = np.ones((1, n))
    b_eq = np.array([100.0])

    # Since w sums to 100: portfolio_return = r^T w / 100
    # For r^T w / 100 >= T_pct →  r^T w >= 100·T_pct →  -r^T w <= -100·T_pct
    A_ub = (-r_pct).reshape(1, n)
    b_ub = np.array([-T_pct * 100.0])

    # ── Call Wolfe solver ────────────────────────────────────────────
    qp_res: WolfeQPResult = solve_qp_wolfe(
        Q    = Q_qp,
        c    = c_qp,
        A_eq = A_eq,
        b_eq = b_eq,
        A_ub = A_ub,
        b_ub = b_ub,
    )

    if not qp_res.success:
        return PortfolioResult(
            success=False,
            message=qp_res.message,
            asset_names=names,
        )

    w_pct  = np.clip(qp_res.x, 0, None)
    # Renormalise to exactly 100 (numerical drift)
    total = w_pct.sum()
    if total < 1e-10:
        return PortfolioResult(
            success=False,
            message="Solver returned a near-zero weight vector.",
            asset_names=names,
        )
    w_pct *= 100.0 / total
    w_frac = w_pct / 100.0

    # ── Portfolio statistics ─────────────────────────────────────────
    port_ret_pct = float(r_pct @ w_frac)
    port_var_pct2 = float(w_frac @ Sigma_pct2 @ w_frac)
    port_std_pct  = float(np.sqrt(max(port_var_pct2, 0.0)))
    sharpe = port_ret_pct / port_std_pct if port_std_pct > 1e-12 else 0.0

    return PortfolioResult(
        weights_pct             = w_pct,
        weights                 = w_frac,
        portfolio_return_pct    = port_ret_pct,
        portfolio_std_pct       = port_std_pct,
        portfolio_variance_pct2 = port_var_pct2,
        sharpe_ratio            = sharpe,
        success                 = True,
        message                 = "Optimal solution found.",
        asset_names             = names,
    )


def efficient_frontier_wolfe(
    expected_returns_pct : list,
    cov_matrix_pct2      : list,
    asset_names          : Optional[list] = None,
    cov_in_decimal       : bool = False,
    n_points             : int  = 50,
) -> dict:
    """
    Trace the efficient frontier by solving the QP across a range of target returns.

    Parameters
    ----------
    expected_returns_pct : list, length n   — expected returns in %
    cov_matrix_pct2      : (n,n)            — covariance in (%^2)
    asset_names          : list of str
    cov_in_decimal       : bool             — if True, cov is in decimal^2
    n_points             : int              — number of points on the frontier

    Returns
    -------
    dict with keys:
        'returns_pct'     : array of achieved returns (%)
        'stds_pct'        : array of std devs (%)
        'weights_pct'     : (n_points × n) weight matrix (%)
        'asset_names'     : list of str
    """
    r_pct = np.array(expected_returns_pct, dtype=float).ravel()
    r_lo  = r_pct.min() + 1e-6
    r_hi  = r_pct.max() - 1e-6

    # Gracefully handle single-point frontiers
    if r_hi - r_lo < 1e-4:
        T_vals = [r_pct.mean()]
    else:
        T_vals = np.linspace(r_lo, r_hi, n_points)

    frontier_ret, frontier_std, frontier_w = [], [], []

    for T in T_vals:
        inp = PortfolioInput(
            expected_returns_pct = expected_returns_pct,
            cov_matrix_pct2      = cov_matrix_pct2,
            target_return_pct    = T,
            asset_names          = asset_names or [],
            cov_in_decimal       = cov_in_decimal,
        )
        res = optimise_portfolio(inp)
        if res.success:
            frontier_ret.append(res.portfolio_return_pct)
            frontier_std.append(res.portfolio_std_pct)
            frontier_w.append(res.weights_pct)

    return {
        "returns_pct" : np.array(frontier_ret),
        "stds_pct"    : np.array(frontier_std),
        "weights_pct" : np.array(frontier_w),
        "asset_names" : asset_names or [f"A{i}" for i in range(len(r_pct))],
    }


# ═══════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("  Portfolio Optimiser — Self-Test  (3 assets)")
    print("=" * 62)

    # Toy 3-asset example (returns and covariance in percent / percent^2)
    returns_pct = [10.0, 15.0, 20.0]
    # Σ in (%²)  — moderately correlated assets
    cov_pct2 = [
        [25.0,  5.0,  3.0],
        [ 5.0, 64.0,  8.0],
        [ 3.0,  8.0, 100.0],
    ]

    for target in [10.5, 13.0, 18.0]:
        inp = PortfolioInput(
            expected_returns_pct = returns_pct,
            cov_matrix_pct2      = cov_pct2,
            target_return_pct    = target,
            asset_names          = ["LowRisk", "MidRisk", "HighRisk"],
        )
        res = optimise_portfolio(inp)
        res.print_summary()