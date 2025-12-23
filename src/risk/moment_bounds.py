import numpy as np
import scipy.optimize as optimize
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class RobustBounds:
    """Stores the risk metrics from the Discrete Conditional Moment Problem."""
    confidence_level: float
    wc_var: float  # Worst-Case VaR
    wc_cvar: float  # Worst-Case CVaR
    dmp_type: str  # 'Standard' or 'Conditional (DCMP)'
    gap: float  # Difference between Worst-Case and Empirical/EVT


class DiscreteConditionalMomentSolver:
    """
    Solves the Discrete Conditional Moment Problem (DCMP) based on Naumova (2015).

    This solver partitions the support into regions (e.g., Body vs Tail) and
    allows enforcing moment constraints specifically within those regions.
    This tightens the risk bounds by incorporating 'shape' information.
    """

    def __init__(self, n_points: int = 500, support_range: Tuple[float, float] = (-10.0, 10.0)):
        self.n_points = n_points
        self.range = support_range
        # Create the grid of potential outcomes (z-scores)
        self.z_grid = np.linspace(support_range[0], support_range[1], n_points)

    def solve_dcmp(self,
                   data: np.ndarray,
                   alpha: float = 0.05,
                   use_conditional: bool = True) -> RobustBounds:
        """
        Estimates Worst-Case Risk using DCMP.

        Args:
            data: Standardized residuals (Z-scores).
            alpha: Risk level (e.g., 0.05 for 95% confidence).
            use_conditional: If True, adds constraints on the tail moments.
        """
        # 1. Global Moments
        mu_global = np.mean(data)
        sigma_global = np.std(data)
        skew_global = np.mean(data ** 3)
        kurt_global = np.mean(data ** 4)

        # 2. Define Constraints Matrix (LHS) and Bounds (RHS)
        # We solve for probabilities p_i on the grid z_i
        # Variables: p = [p_0, p_1, ..., p_n]

        # Base Constraints:
        # Sum(p) = 1
        # Sum(p * z) = mu
        # Sum(p * z^2) = sigma^2 + mu^2 (Second raw moment)

        A_eq = []
        b_eq = []

        # Constraint 0: Probability sums to 1
        A_eq.append(np.ones(self.n_points))
        b_eq.append(1.0)

        # Constraint 1: Mean (First Moment)
        A_eq.append(self.z_grid)
        b_eq.append(mu_global)

        # Constraint 2: Second Moment
        A_eq.append(self.z_grid ** 2)
        b_eq.append(sigma_global ** 2 + mu_global ** 2)

        # Constraint 3: Skewness (Third Moment)
        A_eq.append(self.z_grid ** 3)
        b_eq.append(skew_global)

        # --- THE NAUMOVA IMPROVEMENT (DCMP) ---
        if use_conditional:
            # We "anchor" the tail.
            # Let's define the "Tail" as the worst q% of data.
            # We check what the ACTUAL average loss is in that tail.

            # Find empirical threshold for the tail
            threshold = np.quantile(data, alpha)

            # Get data in the tail
            tail_data = data[data <= threshold]

            if len(tail_data) > 0:
                # Calculate Conditional Expectation E[Z | Z <= threshold]
                cond_mean = np.mean(tail_data)
                prob_mass = alpha  # Approximately

                # Constraint: Sum(p_i * z_i | z_i <= threshold) = cond_mean * prob_mass
                # This prevents the solver from putting all mass at z = -10

                # Create a mask for grid points in the tail region
                mask = (self.z_grid <= threshold).astype(float)

                # Row: sum(p_i * z_i * I(z_i <= thresh))
                A_eq.append(self.z_grid * mask)
                b_eq.append(cond_mean * prob_mass)

                dmp_type = "DCMP (Naumova)"
            else:
                dmp_type = "Standard DMP (Fallback)"
        else:
            dmp_type = "Standard DMP"

        A_eq = np.vstack(A_eq)
        b_eq = np.array(b_eq)

        # 3. Solve for Worst-Case VaR
        # We want to find the largest 'v' such that Prob(Z <= v) >= alpha is plausible.
        # But efficiently, we just use the WC-CVaR optimization directly.

        # 4. Solve for Worst-Case CVaR (Expected Shortfall)
        # Objective: Maximize E[ Loss ] in the tail.
        # Since we are working with returns, "Loss" is negative Z.
        # We want to Minimize Sum(p_i * z_i) over the worst alpha mass.

        # However, purely maximizing tail loss with fixed moments is equivalent to
        # minimizing the expected value of the tail outcomes.

        # approximate WC-CVaR by finding the worst expectation over the lower tail.
        # Minimize sum(p_i * z_i * I(z_i < VaR_threshold))
        # But we don't know VaR_threshold perfectly.
        # A robust proxy: Minimize sum(p_i * z_i) weighted by tail probability.

        # minimize the First Moment (Mean) strictly on the negative side
        # subject to the global constraints. This pushes mass as far left as possible.
        c = self.z_grid.copy()  # Minimize Z (Maximize Loss)

        # Bounds for p_i: [0, 1]
        bounds = [(0, 1) for _ in range(self.n_points)]

        try:
            res = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if res.success:
                # The solver found the "Worst Case Distribution" {p_i}
                worst_dist = res.x

                # Calculate VaR/CVaR on this worst-case distribution
                cum_prob = np.cumsum(worst_dist)

                # Find index where cumulative prob crosses alpha
                idx = np.searchsorted(cum_prob, alpha)
                wc_var = self.z_grid[idx]

                # Calculate CVaR (Expected value of Z conditional on Z <= wc_var)
                tail_probs = worst_dist[:idx + 1]
                tail_vals = self.z_grid[:idx + 1]

                if np.sum(tail_probs) > 0:
                    wc_cvar = np.sum(tail_probs * tail_vals) / np.sum(tail_probs)
                else:
                    wc_cvar = wc_var

                # Invert signs because these are "Returns" (Negative is bad)
                # We usually report VaR/CVaR as positive losses
                return RobustBounds(
                    confidence_level=1 - alpha,
                    wc_var=-wc_var,
                    wc_cvar=-wc_cvar,
                    dmp_type=dmp_type,
                    gap=0.0  # Calculated later
                )
            else:
                return RobustBounds(0.95, 0.0, 0.0, "Failed", 0.0)

        except Exception as e:
            print(f"DMP Solver Error: {e}")
            return RobustBounds(0.95, 0.0, 0.0, "Error", 0.0)