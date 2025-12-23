from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from scipy.optimize import linprog, OptimizeResult


@dataclass
class RobustBounds:
    """Stores the risk metrics from the Discrete Conditional Moment Problem."""
    confidence_level: float
    wc_var: float   # Worst-Case VaR (positive loss)
    wc_cvar: float  # Worst-Case CVaR / ES (positive loss)
    dmp_type: str   # 'Conditional (DCMP)', 'Conditional (relaxed xN)', 'Standard (fallback)', etc.
    gap: float      # Filled elsewhere


class DiscreteConditionalMomentSolver:
    """
    Discrete Moment Problem (DMP) / Discrete Conditional Moment Problem (DCMP) solver.

    We approximate the standardized residual distribution Z by a discrete distribution on a grid z_i
    with unknown probabilities p_i. We enforce global moment constraints on p_i. Optionally we add
    tail-anchor inequalities (DCMP) derived from the empirical left tail (Z <= q_alpha).

    Then we compute a *worst-case* Expected Shortfall bound by solving an LP.

    Conventions:
      - data are standardized returns residuals Z (negative is bad)
      - losses are L = -Z (positive is bad)
      - alpha is tail probability (e.g., 0.05 => 95% confidence)
      - returned wc_var, wc_cvar are positive losses
    """

    def __init__(self, n_points: int = 801, support_range: Tuple[float, float] = (-15.0, 15.0)):
        self.n_points = int(n_points)
        self.range = (float(support_range[0]), float(support_range[1]))
        self.z_grid = np.linspace(self.range[0], self.range[1], self.n_points)

    def solve_dcmp(
        self,
        data: np.ndarray,
        alpha: float = 0.05,
        use_conditional: bool = True,
        tail_mass_tol: float = 0.02,
        tail_mean_rel_tol: float = 0.05,
        solver_method: str = "highs",
        max_relax_rounds: int = 3,
    ) -> RobustBounds:
        """
        Compute worst-case VaR/CVaR (ES) bound.

        If use_conditional=True, we add relaxed inequalities:
          - tail mass approximately alpha in the left tail (Z <= q_alpha)
          - tail first moment approximately alpha * E[Z | Z <= q_alpha]

        If those constraints make the LP infeasible (common when sample is small/noisy),
        we *relax tolerances progressively* for a few rounds before falling back to Standard DMP.
        """
        x = np.asarray(data, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size < 10:
            return RobustBounds(1 - float(alpha), 0.0, 0.0, "InsufficientData", 0.0)

        alpha = float(alpha)
        if not (0.0 < alpha < 0.5):
            return RobustBounds(1 - alpha, 0.0, 0.0, "BadAlpha", 0.0)

        n = self.n_points
        z = self.z_grid

        # ---- global raw moments of Z ----
        mu = float(np.mean(x))
        m2 = float(np.mean(x ** 2))
        m3 = float(np.mean(x ** 3))
        if not (np.isfinite(mu) and np.isfinite(m2) and np.isfinite(m3)):
            return RobustBounds(1 - alpha, 0.0, 0.0, "BadMoments", 0.0)

        # ---- equalities for p: sum p=1, sum p z=mu, sum p z^2=m2, sum p z^3=m3 ----
        A_eq_p = np.vstack([
            np.ones(n, dtype=float),
            z.astype(float),
            (z ** 2).astype(float),
            (z ** 3).astype(float),
        ])
        b_eq_p = np.array([1.0, mu, m2, m3], dtype=float)

        # ---- build CVaR LP in variables [p (n), w (n)] ----
        # objective: maximize sum w_i * L_i where L_i=-z_i
        #          = maximize sum w_i * (-z_i)  <=> minimize sum w_i * z_i
        c = np.concatenate([np.zeros(n, dtype=float), z.astype(float)])

        # equalities: moment constraints on p, plus sum w = 1
        A_eq = np.zeros((A_eq_p.shape[0] + 1, 2 * n), dtype=float)
        b_eq = np.zeros(A_eq_p.shape[0] + 1, dtype=float)
        A_eq[:A_eq_p.shape[0], :n] = A_eq_p
        b_eq[:A_eq_p.shape[0]] = b_eq_p
        A_eq[-1, n:] = 1.0
        b_eq[-1] = 1.0

        # inequalities: alpha*w_i <= p_i  =>  -p_i + alpha*w_i <= 0
        A_ub_base = np.zeros((n, 2 * n), dtype=float)
        for i in range(n):
            A_ub_base[i, i] = -1.0
            A_ub_base[i, n + i] = alpha
        b_ub_base = np.zeros(n, dtype=float)

        bounds = [(0.0, 1.0) for _ in range(n)] + [(0.0, None) for _ in range(n)]

        def _make_conditional_constraints(mass_tol: float, mean_rel_tol: float) -> Tuple[np.ndarray, np.ndarray, str]:
            # Empirical left tail anchor at q_alpha
            q = float(np.quantile(x, alpha))
            tail = x[x <= q]
            if tail.size == 0 or not np.isfinite(q):
                return A_ub_base, b_ub_base, "Standard"

            cond_mean = float(np.mean(tail))
            if not np.isfinite(cond_mean):
                return A_ub_base, b_ub_base, "Standard"

            mask = (z <= q).astype(float)

            tol_mass = max(1e-6, abs(alpha) * float(mass_tol))
            target = alpha * cond_mean
            tol_mean = max(1e-6, abs(target) * float(mean_rel_tol))

            rows: List[np.ndarray] = []
            rhs: List[float] = []

            # alpha - tol <= sum mask*p <= alpha + tol
            rows.append(mask); rhs.append(alpha + tol_mass)
            rows.append(-mask); rhs.append(-(alpha - tol_mass))

            # target - tol <= sum (mask*z*p) <= target + tol
            row = (mask * z).astype(float)
            rows.append(row); rhs.append(target + tol_mean)
            rows.append(-row); rhs.append(-(target - tol_mean))

            A_extra = np.zeros((len(rows), 2 * n), dtype=float)
            for r_i, row in enumerate(rows):
                A_extra[r_i, :n] = row

            A_ub = np.vstack([A_ub_base, A_extra])
            b_ub = np.concatenate([b_ub_base, np.array(rhs, dtype=float)])
            return A_ub, b_ub, "Conditional (DCMP)"

        def _solve(A_ub: np.ndarray, b_ub: np.ndarray) -> OptimizeResult:
            return linprog(
                c=c,
                A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds,
                method=solver_method,
            )

        # ---- try conditional with progressive relaxation ----
        dmp_type = "Standard"
        if use_conditional:
            mass_tol = float(tail_mass_tol)
            mean_tol = float(tail_mean_rel_tol)
            for r in range(max_relax_rounds + 1):
                A_ub, b_ub, dmp_type = _make_conditional_constraints(mass_tol, mean_tol)
                if dmp_type.startswith("Conditional"):
                    res = _solve(A_ub, b_ub)
                    if res.success and res.x is not None:
                        if r > 0:
                            dmp_type = f"Conditional (relaxed x{2 ** r})"
                        break
                    # relax tolerances and retry
                    mass_tol *= 2.0
                    mean_tol *= 2.0
                else:
                    res = _solve(A_ub_base, b_ub_base)
                    break
            else:
                res = _solve(A_ub_base, b_ub_base)
                dmp_type = "Standard (fallback)"
        else:
            res = _solve(A_ub_base, b_ub_base)
            dmp_type = "Standard"

        if (not res.success) or (res.x is None):
            return RobustBounds(1 - alpha, 0.0, 0.0, "Failed", 0.0)

        sol = res.x
        p = np.clip(sol[:n], 0.0, None)
        s = float(np.sum(p))
        if s <= 0.0:
            return RobustBounds(1 - alpha, 0.0, 0.0, "Failed", 0.0)
        p = p / s

        # ---- VaR and CVaR for LEFT tail of Z ----
        cdf = np.cumsum(p)
        j = int(np.searchsorted(cdf, alpha, side="left"))
        j = min(max(j, 0), n - 1)
        q_alpha = float(z[j])

        prev_cdf = float(cdf[j - 1]) if j > 0 else 0.0
        tail_sum = float(np.sum(p[:j] * z[:j]))
        take = max(0.0, alpha - prev_cdf)
        tail_sum += take * q_alpha
        cvar_z = tail_sum / alpha

        # Convert to positive losses: VaR_L = -q_alpha, CVaR_L = -E[Z | Z <= q]
        wc_var = -q_alpha
        wc_cvar = -cvar_z

        # The LP objective corresponds to maximizing ES of loss L=-Z, so -res.fun is an ES bound too.
        wc_cvar_lp = float(-res.fun) if np.isfinite(res.fun) else float("nan")
        if np.isfinite(wc_cvar_lp):
            wc_cvar = max(wc_cvar, wc_cvar_lp)

        return RobustBounds(confidence_level=1 - alpha, wc_var=float(wc_var), wc_cvar=float(wc_cvar), dmp_type=dmp_type, gap=0.0)
