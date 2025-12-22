from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from scipy.optimize import linprog, OptimizeResult


@dataclass
class RobustBounds:
    """Stores the risk metrics from the Discrete Conditional Moment Problem."""
    confidence_level: float
    wc_var: float   # Worst-Case VaR (reported as positive loss)
    wc_cvar: float  # Worst-Case CVaR (reported as positive loss)
    dmp_type: str   # 'Standard' or 'Conditional (DCMP)' or other status
    gap: float      # Difference between Worst-Case and Empirical/EVT (computed elsewhere)


class DiscreteConditionalMomentSolver:
    """
    Discrete Moment Problem (DMP) / Discrete Conditional Moment Problem (DCMP) solver.

    We approximate the (standardized) residual distribution Z by a discrete distribution on a grid z_i
    with unknown probabilities p_i. We enforce moment matching constraints on p_i. Optionally we add
    relaxed (toleranced) constraints that "anchor" the left tail (DCMP idea).

    Then we compute a *worst-case* CVaR bound by solving an LP using the dual form of CVaR.

    Conventions:
      - data are standardized returns Z (negative is bad)
      - losses are L = -Z (positive is bad)
      - alpha is tail probability (e.g. 0.05 => 95% confidence)
      - we return wc_var, wc_cvar as POSITIVE LOSS metrics
    """

    def __init__(self, n_points: int = 500, support_range: Tuple[float, float] = (-10.0, 10.0)):
        self.n_points = int(n_points)
        self.range = support_range
        self.z_grid = np.linspace(float(support_range[0]), float(support_range[1]), self.n_points)

    def solve_dcmp(
        self,
        data: np.ndarray,
        alpha: float = 0.05,
        use_conditional: bool = True,
        tail_mass_tol: float = 0.02,
        tail_mean_rel_tol: float = 0.05,
        solver_method: str = "highs",
    ) -> RobustBounds:
        """
        Worst-case VaR/CVaR bound via DMP/DCMP.

        Core LP (dual CVaR):
          variables: p_i (distribution on grid), w_i (tail reweighting)
          maximize:  sum_i w_i * L_i   where L_i = -z_i
          s.t.:      sum_i w_i = 1,  w_i >= 0
                    alpha * w_i <= p_i    for all i
                    p_i >= 0, sum p_i = 1
                    moment matching constraints on p

        Optional DCMP tail-anchor (relaxed inequalities on p):
          - tail mass around alpha below empirical q_alpha
          - tail first moment around alpha * E[Z | Z <= q_alpha]
        """

        # ---- input cleanup ----
        x = np.asarray(data, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size < 10:
            return RobustBounds(1 - float(alpha), 0.0, 0.0, "InsufficientData", 0.0)

        alpha = float(alpha)
        if not (0.0 < alpha < 0.5):
            # we only support a meaningful left tail probability (e.g., 0.01..0.10)
            return RobustBounds(1 - alpha, 0.0, 0.0, "BadAlpha", 0.0)

        n = self.n_points
        z = self.z_grid

        # ---- global raw moments of Z ----
        mu = float(np.mean(x))
        m2 = float(np.mean(x ** 2))
        m3 = float(np.mean(x ** 3))

        # ---- equality constraints for p: sum p =1, sum p z = mu, sum p z^2 = m2, sum p z^3 = m3 ----
        A_eq_p = np.vstack([
            np.ones(n, dtype=float),
            z.astype(float),
            (z ** 2).astype(float),
            (z ** 3).astype(float),
        ])
        b_eq_p = np.array([1.0, mu, m2, m3], dtype=float)

        # ---- optional relaxed tail-anchor inequalities (on p only) ----
        # tail is LEFT tail of returns: Z <= q_alpha
        A_ub_extra_rows: List[np.ndarray] = []
        b_ub_extra: List[float] = []
        dmp_type = "Standard"

        if use_conditional:
            q = float(np.quantile(x, alpha))
            tail = x[x <= q]
            if tail.size > 0:
                dmp_type = "Conditional (DCMP)"
                cond_mean = float(np.mean(tail))
                mask = (z <= q).astype(float)

                # Tail mass approx alpha:  alpha - tol <= sum mask*p <= alpha + tol
                tol_mass = max(1e-6, abs(alpha) * float(tail_mass_tol))

                A_ub_extra_rows.append(mask)
                b_ub_extra.append(alpha + tol_mass)

                A_ub_extra_rows.append(-mask)
                b_ub_extra.append(-(alpha - tol_mass))

                # Tail first moment approx alpha * cond_mean:
                #   target - tol <= sum (mask*z*p) <= target + tol
                target = alpha * cond_mean
                tol_mean = max(1e-6, abs(target) * float(tail_mean_rel_tol))

                row = (mask * z).astype(float)
                A_ub_extra_rows.append(row)
                b_ub_extra.append(target + tol_mean)

                A_ub_extra_rows.append(-row)
                b_ub_extra.append(-(target - tol_mean))

        use_extra = (use_conditional and len(A_ub_extra_rows) > 0)

        # ---- build CVaR LP in variables [p (n), w (n)] ----
        # objective: maximize sum w_i * (-z_i)  <=> minimize sum w_i * z_i
        c = np.concatenate([np.zeros(n, dtype=float), z.astype(float)])

        # equalities:
        #   moment constraints on p
        #   sum w = 1
        A_eq = np.zeros((A_eq_p.shape[0] + 1, 2 * n), dtype=float)
        b_eq = np.zeros(A_eq_p.shape[0] + 1, dtype=float)
        A_eq[:A_eq_p.shape[0], :n] = A_eq_p
        b_eq[:A_eq_p.shape[0]] = b_eq_p
        A_eq[-1, n:] = 1.0
        b_eq[-1] = 1.0

        # inequalities:
        #   alpha*w_i <= p_i  =>  -p_i + alpha*w_i <= 0
        A_ub_base = np.zeros((n, 2 * n), dtype=float)
        for i in range(n):
            A_ub_base[i, i] = -1.0
            A_ub_base[i, n + i] = alpha
        b_ub_base = np.zeros(n, dtype=float)

        # optional extra (tail anchor) inequalities on p only
        if use_extra:
            A_extra = np.zeros((len(A_ub_extra_rows), 2 * n), dtype=float)
            for r_i, row in enumerate(A_ub_extra_rows):
                A_extra[r_i, :n] = row
            b_extra = np.array(b_ub_extra, dtype=float)

            A_ub = np.vstack([A_ub_base, A_extra])
            b_ub = np.concatenate([b_ub_base, b_extra])
        else:
            A_ub = A_ub_base
            b_ub = b_ub_base

        # bounds: p in [0,1], w in [0, +inf)
        bounds = [(0.0, 1.0) for _ in range(n)] + [(0.0, None) for _ in range(n)]

        def _solve_lp(with_extra: bool) -> OptimizeResult:
            if with_extra and use_extra:
                A_use = A_ub
                b_use = b_ub
            else:
                A_use = A_ub_base
                b_use = b_ub_base

            return linprog(
                c=c,
                A_ub=A_use, b_ub=b_use,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds,
                method=solver_method,
            )

        # ---- solve (conditional first, fallback to standard) ----
        res = _solve_lp(with_extra=True)
        if (not res.success) and use_extra:
            dmp_type = "Standard"
            res = _solve_lp(with_extra=False)

        if not res.success or res.x is None:
            return RobustBounds(1 - alpha, 0.0, 0.0, "Failed", 0.0)

        sol = res.x
        p = np.clip(sol[:n], 0.0, None)
        s = float(np.sum(p))
        if s <= 0.0:
            return RobustBounds(1 - alpha, 0.0, 0.0, "Failed", 0.0)
        p = p / s

        # ---- compute VaR and CVaR from p in the LEFT tail of Z ----
        # Grid is already ascending. Compute CDF and find q_alpha (Z-quantile).
        cdf = np.cumsum(p)
        j = int(np.searchsorted(cdf, alpha, side="left"))
        j = min(max(j, 0), n - 1)
        q_alpha = float(z[j])

        # CVaR_Z = E[Z | Z <= q_alpha] computed with partial mass at boundary
        prev_cdf = float(cdf[j - 1]) if j > 0 else 0.0
        tail_sum = float(np.sum(p[:j] * z[:j]))
        take = max(0.0, alpha - prev_cdf)
        tail_sum += take * q_alpha
        cvar_z = tail_sum / alpha

        # Convert to positive LOSS metrics (Loss = -Z)
        wc_var = -q_alpha
        wc_cvar = -cvar_z

        # LP objective corresponds to maximizing CVaR of losses, so -res.fun is a valid CVaR bound too.
        # Use the maximum of the two to be safe against discretization mismatch.
        wc_cvar_lp = float(-res.fun) if np.isfinite(res.fun) else float("nan")
        if np.isfinite(wc_cvar_lp):
            wc_cvar = max(wc_cvar, wc_cvar_lp)

        return RobustBounds(
            confidence_level=1 - alpha,
            wc_var=float(wc_var),
            wc_cvar=float(wc_cvar),
            dmp_type=dmp_type,
            gap=0.0,
        )
