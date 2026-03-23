"""
QuadraticSimplex.py
==================
Quadratic Programming solver using Wolfe's Modified Simplex Method.

Solves the standard QP problem:
    min   (1/2) x^T Q x + c^T x
    s.t.  A_eq x  = b_eq       (equality constraints)
          A_ub x <= b_ub       (inequality constraints)
          x >= 0               (non-negativity)

Wolfe's method converts the QP into a Linear Complementarity Problem (LCP)
via the KKT conditions, then solves the resulting augmented LP using the
standard Simplex method with a **restricted basis entry rule** that enforces
complementarity.

KKT stationarity:  Q x + c - A_eq^T λ - A_ub^T μ + s = 0
Primal feasibility: A_eq x = b_eq
                    A_ub x + t = b_ub
Complementarity:    s_i * x_i = 0   for all i
                    μ_j * t_j = 0   for all j
Non-negativity:     x, s, λ (free), μ, t >= 0

Reference
---------
Wolfe, P. (1959). "The Simplex Method for Quadratic Programming."
Econometrica, 27(3), 382–398.
"""

import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
#  INTERNAL: Big-M Simplex Tableau with Complementarity Restriction
# ═══════════════════════════════════════════════════════════════════

_BIG_M     = 1e6    # Penalty for artificial variables
_PIVOT_TOL = 1e-10  # Numerical zero for pivot selection
_FEAS_TOL  = 1e-8   # Feasibility tolerance for solution validation


def _simplex(
    tableau: np.ndarray,
    basis: list[int],
    complement: dict[int, int],
) -> tuple[np.ndarray, list[int], bool]:
    """
    Run the Big-M Simplex method on a tableau in canonical form,
    with Wolfe's restricted basis entry rule.

    Parameters
    ----------
    tableau : ndarray, shape (m+1, n+1)
        Rows 0..m-1 are constraints; row m is the objective (minimise).
        Last column is RHS.
    basis : list of int, length m
        Initial basic variable indices (one per constraint row).
    complement : dict[int, int]
        Mapping from variable index to its complementary partner.
        e.g. {x_i: s_i, s_i: x_i, mu_j: t_j, t_j: mu_j}
        At most one variable of each pair may be in the basis.

    Returns
    -------
    tableau : updated tableau
    basis   : updated basis
    success : True if optimal solution found
    """
    m, cols = tableau.shape[0] - 1, tableau.shape[1]
    n = cols - 1  # number of variables (columns excluding RHS)

    basis_set = set(basis)
    max_iter = max(20 * n, 500)

    for _iter in range(max_iter):
        obj_row = tableau[m, :-1]

        # ── Find entering variable (most negative reduced cost) ──────
        # Wolfe's restricted basis entry rule:
        #   Variable j may enter if:
        #     (a) j has no complement, or
        #     (b) complement(j) is NOT in the basis, or
        #     (c) complement(j) IS in the basis but would LEAVE via a
        #         degenerate pivot (min ratio = 0 and it is the leaving row).
        #
        # We use a two-pass approach:
        #   Pass 1: candidates satisfying (a) or (b)
        #   Pass 2: if no candidate from Pass 1, try candidates via (c)

        candidates = []
        blocked_candidates = []

        for j in range(n):
            if obj_row[j] < -_PIVOT_TOL:
                comp = complement.get(j, -1)
                if comp == -1 or comp not in basis_set:
                    candidates.append((obj_row[j], j))
                else:
                    blocked_candidates.append((obj_row[j], j))

        enter = -1
        leave_row = -1

        if candidates:
            # Pick the most negative reduced cost
            candidates.sort()
            enter = candidates[0][1]

            # Standard min-ratio test
            col = tableau[:m, enter]
            rhs = tableau[:m, -1]
            ratios = np.full(m, np.inf)
            pos = col > _PIVOT_TOL
            ratios[pos] = rhs[pos] / col[pos]

            if np.all(ratios == np.inf):
                return tableau, basis, False  # unbounded

            leave_row = int(np.argmin(ratios))

        elif blocked_candidates:
            # Try each blocked candidate; allow it if its complement
            # would be the leaving variable (degenerate pivot, ratio=0).
            blocked_candidates.sort()
            for _, j in blocked_candidates:
                comp = complement[j]
                # Find which row comp occupies in the basis
                comp_row = -1
                for r_idx, b_var in enumerate(basis):
                    if b_var == comp:
                        comp_row = r_idx
                        break
                if comp_row == -1:
                    continue

                col = tableau[:m, j]
                rhs = tableau[:m, -1]
                ratios = np.full(m, np.inf)
                pos = col > _PIVOT_TOL
                ratios[pos] = rhs[pos] / col[pos]

                if np.all(ratios == np.inf):
                    continue

                min_ratio = np.min(ratios)
                lr = int(np.argmin(ratios))

                # Allow if the complement row has ratio = min_ratio ≈ 0
                # (degenerate pivot where the complement leaves)
                if (comp_row == lr or
                    (abs(ratios[comp_row] - min_ratio) < _PIVOT_TOL + 1e-12 * max(1, abs(min_ratio)))):
                    enter = j
                    leave_row = comp_row
                    break

        if enter == -1:
            break  # Optimal (no eligible entering variable)

        # ── Pivot ────────────────────────────────────────────────────
        pivot = tableau[leave_row, enter]
        tableau[leave_row, :] /= pivot
        for r in range(m + 1):
            if r != leave_row:
                factor = tableau[r, enter]
                if abs(factor) > 1e-16:
                    tableau[r, :] -= factor * tableau[leave_row, :]

        leaving_var = basis[leave_row]
        basis_set.discard(leaving_var)
        basis_set.add(enter)
        basis[leave_row] = enter

    return tableau, basis, True


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════

class WolfeQPResult:
    """Container for QP solution results."""

    def __init__(self,
                 x: Optional[np.ndarray],
                 obj_val: Optional[float],
                 success: bool,
                 message: str):
        self.x        = x          # Optimal primal variables (or None)
        self.obj_val  = obj_val    # Optimal objective value (or None)
        self.success  = success
        self.message  = message

    def __repr__(self):
        if self.success:
            w_str = np.array2string(self.x, precision=6, suppress_small=True)
            return (f"WolfeQPResult(success=True, obj_val={self.obj_val:.6f},\n"
                    f"  x={w_str})")
        return f"WolfeQPResult(success=False, message='{self.message}')"


def solve_qp_wolfe(
    Q:      np.ndarray,
    c:      np.ndarray,
    A_eq:   Optional[np.ndarray] = None,
    b_eq:   Optional[np.ndarray] = None,
    A_ub:   Optional[np.ndarray] = None,
    b_ub:   Optional[np.ndarray] = None,
) -> WolfeQPResult:
    """
    Solve a convex QP via Wolfe's Modified Simplex Method.

    Problem
    -------
        min   (1/2) x^T Q x + c^T x
        s.t.  A_eq x  = b_eq
              A_ub x <= b_ub
              x >= 0

    Parameters
    ----------
    Q     : (n, n) symmetric positive semi-definite cost matrix
    c     : (n,)  linear cost vector
    A_eq  : (p, n) equality constraint matrix  [optional]
    b_eq  : (p,)   equality RHS                [optional]
    A_ub  : (q, n) inequality constraint matrix [optional]
    b_ub  : (q,)   inequality RHS              [optional]

    Returns
    -------
    WolfeQPResult
        .x       : optimal primal solution (length n)
        .obj_val : optimal objective value
        .success : bool
        .message : status string
    """
    Q = np.atleast_2d(np.array(Q, dtype=float))
    c = np.array(c, dtype=float).ravel()
    n = len(c)

    # ── Validate symmetry / PSD ─────────────────────────────────────
    if Q.shape != (n, n):
        raise ValueError(f"Q must be ({n},{n}), got {Q.shape}.")
    Q = 0.5 * (Q + Q.T)   # force symmetry

    # ── Default constraints ─────────────────────────────────────────
    if A_eq is None or b_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    else:
        A_eq = np.atleast_2d(np.array(A_eq, dtype=float))
        b_eq = np.array(b_eq, dtype=float).ravel()

    if A_ub is None or b_ub is None:
        A_ub = np.zeros((0, n))
        b_ub = np.zeros(0)
    else:
        A_ub = np.atleast_2d(np.array(A_ub, dtype=float))
        b_ub = np.array(b_ub, dtype=float).ravel()

    p, q = A_eq.shape[0], A_ub.shape[0]

    # ══════════════════════════════════════════════════════════════════
    #  Build Wolfe's augmented LP tableau
    # ══════════════════════════════════════════════════════════════════
    #
    # Variables (in column order):
    #   x[n]           : primal variables
    #   s[n]           : KKT surplus for stationarity  (complementary to x)
    #   lam_p[p]       : λ+ (positive part of equality multipliers)
    #   lam_n[p]       : λ- (negative part, so λ = lam_p - lam_n)
    #   mu[q]          : inequality multipliers >= 0  (complementary to t)
    #   t[q]           : primal slack for inequality   (complementary to mu)
    #
    # Rows:
    #   [0..n-1]       : Stationarity   Qx + s - A_eq^T(lam_p - lam_n) - A_ub^T mu = -c
    #   [n..n+p-1]     : Equality       A_eq x = b_eq
    #   [n+p..n+p+q-1] : Inequality     A_ub x + t = b_ub
    #
    # Artificial variables will be added after RHS sign-correction.

    # Column offsets for structural variables
    off_x    = 0
    off_s    = n
    off_lp   = off_s  + n
    off_ln   = off_lp + p
    off_mu   = off_ln + p
    off_t    = off_mu + q

    num_struct = n + n + p + p + q + q   # total structural + slack variables
    num_rows   = n + p + q

    # Build the structural part of the tableau (no artificials yet)
    # We'll add artificial columns after we know which rows need them.
    struct = np.zeros((num_rows, num_struct + 1))  # +1 for RHS

    # ── Stationarity rows: Q x - s + A_eq^T lam_p - A_eq^T lam_n + A_ub^T mu = -c
    #    (from KKT: Qx + c + A_eq^T λ + A_ub^T μ = s, rearranged)
    for i in range(n):
        row = i
        struct[row, off_x  : off_x  + n]  =  Q[i, :]
        struct[row, off_s  + i]            = -1.0       # -s_i
        if p > 0:
            struct[row, off_lp : off_lp + p] =  A_eq[:, i]   # +A_eq^T λ+
            struct[row, off_ln : off_ln + p] = -A_eq[:, i]   # -A_eq^T λ-
        if q > 0:
            struct[row, off_mu : off_mu + q] =  A_ub[:, i]   # +A_ub^T μ
        struct[row, -1] = -c[i]

    # ── Equality rows: A_eq x = b_eq
    for j in range(p):
        row = n + j
        struct[row, off_x : off_x + n] = A_eq[j, :]
        struct[row, -1] = b_eq[j]

    # ── Inequality rows: A_ub x + t = b_ub
    for k in range(q):
        row = n + p + k
        struct[row, off_x  : off_x  + n] = A_ub[k, :]
        struct[row, off_t + k]            = 1.0
        struct[row, -1] = b_ub[k]

    # ── Flip rows with negative RHS so all RHS >= 0 ─────────────────
    for i in range(num_rows):
        if struct[i, -1] < 0:
            struct[i, :] *= -1

    # ── Determine which rows need artificial variables ──────────────
    # After sign-flipping, check if each row already has an identity
    # column from its natural slack/surplus variable.
    needs_art = []

    for i in range(n):
        # Stationarity row: s_i should provide the identity column,
        # but only if its coefficient is still +1
        if abs(struct[i, off_s + i] - 1.0) > 1e-12:
            needs_art.append(i)
        else:
            # Also check that no other row has a non-zero entry in this column
            col_ok = True
            for r in range(num_rows):
                if r != i and abs(struct[r, off_s + i]) > 1e-12:
                    col_ok = False
                    break
            if not col_ok:
                needs_art.append(i)

    for j in range(p):
        # Equality rows never have a natural slack → always need artificial
        needs_art.append(n + j)

    for k in range(q):
        row = n + p + k
        # Inequality row: t_k should provide the identity column
        if abs(struct[row, off_t + k] - 1.0) > 1e-12:
            needs_art.append(row)

    num_art = len(needs_art)

    # ── Build full tableau with artificial columns ───────────────────
    # Columns: [structural vars | artificial vars | RHS]
    off_art = num_struct
    total_cols = num_struct + num_art + 1  # +1 for RHS

    tableau = np.zeros((num_rows + 1, total_cols))  # +1 row for objective
    # Copy structural part
    tableau[:num_rows, :num_struct] = struct[:, :num_struct]
    tableau[:num_rows, -1]         = struct[:, -1]  # RHS

    # Add artificial variables
    art_col_map = {}  # row → artificial column index
    for idx, row in enumerate(needs_art):
        art_col = off_art + idx
        tableau[row, art_col] = 1.0
        art_col_map[row] = art_col

    # ── Set up initial basis ─────────────────────────────────────────
    basis = [0] * num_rows
    for i in range(n):
        if i in needs_art:
            basis[i] = art_col_map[i]
        else:
            basis[i] = off_s + i

    for j in range(p):
        row = n + j
        basis[row] = art_col_map[row]   # always artificial

    for k in range(q):
        row = n + p + k
        if row in needs_art:
            basis[row] = art_col_map[row]
        else:
            basis[row] = off_t + k

    # ── Objective: minimise sum of artificials (Big-M penalty) ───────
    # For minimisation with Big-M:
    #   1. Each artificial variable has cost +M in the objective.
    #   2. To achieve canonical form (zero reduced cost for basic vars),
    #      subtract M × (constraint row) for each basic artificial.
    obj_row_idx = num_rows
    # Step 1: Set +M cost for each artificial variable
    for idx in range(num_art):
        art_col = off_art + idx
        tableau[obj_row_idx, art_col] = _BIG_M
    # Step 2: Canonicalise — subtract M * row for each artificial in basis
    for idx, row in enumerate(needs_art):
        tableau[obj_row_idx, :] -= _BIG_M * tableau[row, :]

    # ── Build complementarity pairs ──────────────────────────────────
    # Wolfe's rule: x_i ↔ s_i  and  mu_j ↔ t_j
    complement: dict[int, int] = {}
    for i in range(n):
        complement[off_x + i] = off_s + i
        complement[off_s + i] = off_x + i
    for k in range(q):
        complement[off_mu + k] = off_t + k
        complement[off_t + k]  = off_mu + k

    # ── Run Simplex with complementarity restriction ─────────────────
    try:
        tableau, basis, ok = _simplex(tableau, basis, complement)
    except Exception as e:
        return WolfeQPResult(None, None, False, str(e))

    if not ok:
        return WolfeQPResult(None, None, False,
                             "QP is unbounded or solver failed to converge.")

    # ── Extract solution ─────────────────────────────────────────────
    x_sol = np.zeros(n)
    for row_idx, var_idx in enumerate(basis):
        if off_x <= var_idx < off_x + n:
            x_sol[var_idx - off_x] = tableau[row_idx, -1]

    # Clip tiny negatives from numerical noise
    x_sol = np.maximum(x_sol, 0.0)

    # ── Feasibility check ────────────────────────────────────────────
    infeasible = False
    if p > 0:
        eq_violation = np.max(np.abs(A_eq @ x_sol - b_eq))
        if eq_violation > 1e-4:
            infeasible = True
    if q > 0:
        ub_violation = np.max(A_ub @ x_sol - b_ub)
        if ub_violation > 1e-4:
            infeasible = True

    # Check if any artificial variable remains in basis at non-zero level
    art_indices = set(off_art + idx for idx in range(num_art))
    art_in_basis = any(
        v in art_indices and abs(tableau[r, -1]) > _FEAS_TOL
        for r, v in enumerate(basis)
    )

    if infeasible or art_in_basis:
        return WolfeQPResult(
            None, None, False,
            "Problem is infeasible or no feasible solution found by Wolfe's method."
        )

    obj_val = 0.5 * float(x_sol @ Q @ x_sol) + float(c @ x_sol)

    return WolfeQPResult(x_sol, obj_val, True, "Optimal solution found.")


# ═══════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("─" * 55)
    print("  Wolfe QP Solver — Self-Test")
    print("─" * 55)

    Q = np.array([[0.0128, 0.00576], [0.00576, 0.0288]])
    c = np.array([0., 0.])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])

    r1 = solve_qp_wolfe(Q, c, A_eq=A, b_eq=b)
    print(f"\nManual Problem:")
    print(f"  {r1}")