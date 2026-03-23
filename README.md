# Markowitz Portfolio Optimisation via Wolfe's Modified Simplex

This repository implements a robust, end-to-end portfolio optimizer that fetches live market data, computes historical statistics, and solves the classic **Markowitz Mean-Variance Portfolio Optimization** problem. 

The core optimization engine is built entirely from scratch using **Wolfe's Modified Simplex Method** for Quadratic Programming (QP).

## Mathematical Formulation

### 1. The Markowitz Portfolio Model
Our goal is to find the optimal asset weights $w$ (in percentages, $\sum w_i = 100$) that minimize the portfolio variance for a given target return constraint $T$.

$$
\begin{align*}
\min_{w} \quad & \frac{1}{2} w^T (2\Sigma) w \\
\text{subject to:} \quad 
& \sum_{i=1}^n w_i = 100 \quad \text{(Fully invested)} \\
& \sum_{i=1}^n r_i w_i \ge 100 \cdot T \quad \text{(Target return constraint)} \\
& w \ge 0 \quad \text{(No short-selling)}
\end{align*}
$$
where $\Sigma$ is the covariance matrix of asset returns, and $r$ is the vector of expected returns.

### 2. Wolfe's Modified Simplex Method
The Markowitz model is a Quadratic Programming (QP) problem. Wolfe's method cleverly transforms convex QP problems into Linear Complementarity Problems (LCPs), which can then be solved using an augmented Simplex algorithm.

The Lagrangian of the QP yields the **KKT Stationarity Conditions**:
$$ Qw - s + A_{eq}^T \lambda + A_{ub}^T \mu = -c $$
where:
* $Q = 2\Sigma$
* $s \ge 0$ are the slack variables for $w \ge 0$
* $\lambda$ are the multipliers for equality constraints
* $\mu$ are the multipliers for inequality constraints

Wolfe's method sets up a **Big-M LP Tableau** to minimize a set of artificial variables (Phase I of the Simplex method) placed into the stationarity rows to find a feasible starting basis.

**The Complementarity Restriction**
The crucial modification to the standard Simplex algorithm is the *restricted basis entry rule*. Because $x_i$ and $s_i$ are complementary ($x_i s_i = 0$), the solver mathematically enforces that $x_i$ and $s_i$ can **never** be in the basis at the same time. The same applies to inequality slack variables $t_j$ and their multipliers $\mu_j$.

## Codebase Architecture

The project is split into three main modules:

### `QuadraticSimplex.py` (The Backend Solver)
Contains `solve_qp_wolfe()`, a strictly typed, pure Python implementation of Wolfe's method using NumPy.
* Computes the augmented Lagrangian KKT stationarity rows.
* Identifies required initial artificial variables to form the Simplex identity matrix block.
* Solves the Phase I setup dynamically via the Big-M objective method.
* Strictly enforces Wolfe's complementarity restriction avoiding "unbounded" optimization failures.

### `PortfolioOptimizer.py` (The Mathematical Translation Layer)
Translates the real-world statistical problem into the canonical QP form.
* Accepts raw expected returns and covariance matrices.
* Calculates spectral decompositions to clip negative eigenvalues, explicitly forcing covariance matrices to be Positive Semi-Definite (PSD) for numerical stability.
* Transforms weight structures scaling them natively to summing to 100% for human readability and tighter numeric conditioning.
* Contains `efficient_frontier_wolfe()` which sweeps across target returns to build out the efficient frontier graph natively.

### `Task.py` (The Execution Driver)
The user-facing runner that glues the system together with live market data.
1. **Data Fetching:** Downloads 4 months of recent asset pricing using the `yfinance` API (e.g. NIFTY50 components + Gold). Handles missing/incomplete initial public offering (IPO) data with intelligent forward/backward filling.
2. **Statistical Generation:** Computes annualized expected returns and the historical covariance matrix in percentage terms.
3. **Execution:** Prompts the user for a target annualized return, triggers the `PortfolioOptimizer`, and displays the final allocation weights alongside expected Sharpe ratios.
4. **Visualization:** Evaluates 50 nodes along the risk-return curve and mathematically plots the Markowitz Efficient Frontier to `./outputs/nifty_gold_portfolio.png`.

## Execution
Ensure dependencies are installed: `pip install numpy pandas matplotlib yfinance`

To run the optimizer and trace the frontier:
```bash
python Task.py
```
