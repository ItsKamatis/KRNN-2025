import torch
import numpy as np
import logging
from typing import Dict

from src.risk.moment_bounds import DiscreteConditionalMomentSolver

logger = logging.getLogger(__name__)


def get_standardized_residuals(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> np.ndarray:
    """
    Calculates Standardized Residuals: Z = (y - mu) / sigma
    Used to isolate the 'surprise' component from the predicted volatility.
    """
    model.eval()
    z_scores = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device).squeeze(-1).float()

            # Probabilistic output
            mu, sigma = model(features)

            # Z-score
            z = (targets - mu) / sigma
            z_scores.append(z.detach().cpu().numpy())

    if len(z_scores) == 0:
        return np.array([], dtype=float)

    out = np.concatenate(z_scores).reshape(-1)
    out = out[np.isfinite(out)]
    return out


class EVTEngine:
    """
    Extreme Value Theory Engine for Risk Management.
    """

    def __init__(self, tail_fraction: float = 0.10):
        self.tail_fraction = float(tail_fraction)

    def analyze_tails(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Estimates the Tail Index (Gamma) using the Hill Estimator.
        Input should be STANDARDIZED residuals (Z-scores).

        We care about the left tail of returns (negative Z), so define losses L = -Z.
        """
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        residuals = residuals[np.isfinite(residuals)]

        # If empty or too short, return safe defaults
        if residuals.size < 10:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(residuals.size), "k_exceedances": 0.0}

        # Loss = -Z (so big loss = big positive number)
        losses = -residuals
        losses = losses[np.isfinite(losses)]
        losses = np.sort(losses)[::-1]  # descending

        n = int(losses.size)
        if n < 10:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": 0.0}

        # Choose k exceedances
        k = int(n * self.tail_fraction)
        k = max(0, min(k, n - 1))

        if k < 5:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": float(k)}

        threshold_loss = float(losses[k])

        # Hill estimator requires positive tail losses.
        # If threshold <= 0, the "tail" includes non-loss outcomes; filter to positive losses only.
        if threshold_loss <= 0.0:
            valid_losses = losses[losses > 0.0]
            if valid_losses.size < 5:
                return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(valid_losses.size), "k_exceedances": 0.0}

            losses = valid_losses
            n = int(losses.size)
            k = int(n * self.tail_fraction)
            k = max(0, min(k, n - 1))

            if k < 5:
                return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": float(k)}

            threshold_loss = float(losses[k])
            if threshold_loss <= 0.0:
                return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": float(k)}

        # Hill estimator: gamma = mean(log(L_i) - log(u)), i=1..k
        top = losses[:k]
        top = top[top > 0.0]
        if top.size < 2:
            return {"gamma": 0.0, "threshold_loss": threshold_loss, "n_samples": float(n), "k_exceedances": float(k)}

        log_losses = np.log(top)
        log_threshold = np.log(threshold_loss)
        gamma = float(np.mean(log_losses - log_threshold))

        return {
            "gamma": gamma,                 # Tail index
            "threshold_loss": threshold_loss,
            "n_samples": float(n),
            "k_exceedances": float(k),
        }

    def calculate_risk_metrics(self, evt_params: Dict[str, float], confidence_level: float = 0.99) -> Dict[str, float]:
        """
        Calculates VaR and ES for the LOSS distribution L = -Z, using EVT/POT formulas.
        Returns positive loss VaR/ES on standardized residuals.
        """
        p = float(confidence_level)
        gamma = float(evt_params.get("gamma", 0.0))
        threshold = float(evt_params.get("threshold_loss", 0.0))
        n = float(evt_params.get("n_samples", 0.0))
        k = float(evt_params.get("k_exceedances", 0.0))

        if k <= 0 or threshold <= 0 or n <= 0 or not (0.0 < p < 1.0):
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": 0.0}

        # risk_ratio = k / (n * (1 - p))
        denom = n * (1.0 - p)
        if denom <= 0:
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": gamma}

        risk_ratio = k / denom
        if risk_ratio <= 0:
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": gamma}

        var_evt = threshold * (risk_ratio ** gamma)

        # ES exists only if gamma < 1
        if gamma >= 1.0:
            es_evt = float("inf")
        else:
            es_evt = var_evt / (1.0 - gamma)

        return {
            f"VaR_{p}": float(var_evt),
            f"ES_{p}": float(es_evt),
            "Gamma_Hill": float(gamma),
        }

    def calculate_risk_metrics_dmp(
        self,
        residuals: np.ndarray,
        confidence_level: float = 0.99,
        solver: DiscreteConditionalMomentSolver = None,
        use_conditional: bool = True,
        n_points: int = 500,
        support_range: tuple = (-10.0, 10.0),
    ) -> Dict[str, float]:
        """
        Calculates VaR and ES using the Discrete Moment Problem (DMP/DCMP).

        residuals are standardized returns Z (negative is bad).
        The solver returns wc_var, wc_cvar as POSITIVE LOSSES corresponding to the left tail of Z.
        """
        p = float(confidence_level)
        if not (0.0 < p < 1.0):
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "DMP_Type": "BadConfidence"}

        if solver is None:
            solver = DiscreteConditionalMomentSolver(n_points=n_points, support_range=support_range)

        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size < 10:
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "DMP_Type": "InsufficientData"}

        alpha = 1.0 - p
        bounds = solver.solve_dcmp(residuals, alpha=alpha, use_conditional=use_conditional)

        return {
            f"VaR_{p}": float(bounds.wc_var),
            f"ES_{p}": float(bounds.wc_cvar),
            "DMP_Type": str(bounds.dmp_type),
        }

    def generate_scenarios(
        self,
        n_simulations: int,
        gamma: float,
        volatility: float,
        expected_return: float
    ) -> np.ndarray:
        """
        Simulate future returns using the Heteroscedastic model + EVT shocks:

            Return_sim = expected_return + volatility * Z_sim
        """
        n_simulations = int(n_simulations)
        gamma = float(gamma)
        volatility = float(volatility)
        expected_return = float(expected_return)

        if n_simulations <= 0:
            return np.array([], dtype=float)

        # 1) draw standardized shocks
        if gamma <= 0.01:
            z_sim = np.random.normal(0.0, 1.0, n_simulations)
        elif gamma >= 1.0:
            z_sim = np.random.standard_t(df=2.0, size=n_simulations)
        else:
            df = 1.0 / gamma
            # numerical safety
            df = max(df, 2.0)
            z_sim = np.random.standard_t(df=df, size=n_simulations)

        # 2) scale + shift
        return expected_return + (z_sim * volatility)
