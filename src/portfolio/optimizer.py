# src/portfolio/optimizer.py
import numpy as np
import scipy.optimize as opt
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class MeanCVaROptimizer:
    """
    Portfolio Optimizer that minimizes Conditional Value at Risk (CVaR).
    Methodology: Rockafellar & Uryasev (2000) Linear Programming formulation.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.alpha = confidence_level

    def optimize(self,
                 expected_returns: np.ndarray,
                 scenarios: np.ndarray,
                 target_return: float) -> Dict:
        """
        Solves the Linear Program to find optimal weights.

        Args:
            expected_returns (N,): Predicted returns for N assets (from KRNN).
            scenarios (T, N): Historical/Simulated return scenarios for N assets.
                             (Constructed from: Prediction + Residuals)
            target_return (float): Minimum acceptable portfolio return.

        Returns:
            Dict containing optimal 'weights', 'risk_cvar', and 'status'.
        """
        T, N = scenarios.shape  # T: Time steps (scenarios), N: Assets

        # --- 1. Define Decision Variables ---
        # The solver vector x has size: 1 (gamma) + T (z variables) + N (weights)
        # Structure: x = [gamma, z_1, ..., z_T, w_1, ..., w_N]
        num_vars = 1 + T + N

        # Indices to help us map the vector
        idx_gamma = 0
        idx_z_start = 1
        idx_z_end = 1 + T
        idx_w_start = 1 + T

        # --- 2. Objective Function (c^T x) ---
        # Min: gamma + (1 / ((1-alpha) * T)) * sum(z)
        c = np.zeros(num_vars)
        c[idx_gamma] = 1.0
        c[idx_z_start:idx_z_end] = 1.0 / ((1 - self.alpha) * T)
        c[idx_w_start:] = 0.0  # Weights don't appear directly in objective cost

        # --- 3. Inequality Constraints (A_ub x <= b_ub) ---
        # We need to express: z_t >= -w^T r_t - gamma
        # Rearranged for linprog (<=): -gamma - z_t - w^T r_t <= 0

        A_ub = []
        b_ub = []

        # Constraint A: Loss Constraints (one per scenario t)
        # -1*gamma + -1*z_t + (-r_t)*w <= 0
        for t in range(T):
            row = np.zeros(num_vars)
            row[idx_gamma] = -1.0
            row[idx_z_start + t] = -1.0
            row[idx_w_start:] = -scenarios[t, :]  # Negative returns
            A_ub.append(row)
            b_ub.append(0.0)

        # Constraint B: Minimum Return Constraint
        # w^T mu >= target  =>  -w^T mu <= -target
            # --- Handle Negative Return Targets ---
            # Logic Fix: If market is down (avg return < 0), we just want to lose LESS than the average.
            # We set the target to be slightly strictly better (higher) than the worst assets.

            avg_asset_return = np.mean(expected_returns)
            if avg_asset_return < 0:
                # If returns are negative, target cannot be positive.
                # Set target to be the average return (neutral relative performance)
                effective_target = avg_asset_return
            else:
                effective_target = target_return

            # Constraint B: Minimum Return Constraint
            # w^T mu >= target  =>  -w^T mu <= -target
            row_ret = np.zeros(num_vars)
            row_ret[idx_w_start:] = -expected_returns
            A_ub.append(row_ret)
            b_ub.append(-effective_target)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # --- 4. Equality Constraints (A_eq x = b_eq) ---
        # Constraint: sum(w) = 1 (Fully invested)
        A_eq = np.zeros((1, num_vars))
        A_eq[0, idx_w_start:] = 1.0
        b_eq = np.array([1.0])

        # --- 5. Bounds ---
        # gamma: (-inf, inf) usually, but VaR is essentially bounded
        # z_t: [0, inf)
        # w_i: [0, 1] (No short selling for this class project)
        bounds = []
        bounds.append((None, None))  # Gamma
        for _ in range(T):
            bounds.append((0, None))  # z_t
        for _ in range(N):
            bounds.append((0, 1))  # w_i

        # --- 6. Solve ---
        logger.info(f"Solving Mean-CVaR optimization for {N} assets with {T} scenarios...")
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result.success:
            logger.error(f"Optimization failed: {result.message}")
            return None

        # Extract Weights
        optimal_weights = result.x[idx_w_start:]

        # Clean small numerical noise (e.g., 1e-10 becomes 0)
        optimal_weights[optimal_weights < 1e-4] = 0.0
        optimal_weights /= optimal_weights.sum()  # Re-normalize

        return {
            "weights": optimal_weights,
            "VaR": result.x[idx_gamma],
            "CVaR_Optimal": result.fun,  # The objective value is the CVaR
            "status": result.message
        }