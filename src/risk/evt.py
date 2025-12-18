import torch
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def get_standardized_residuals(model: torch.nn.Module,
                               dataloader: torch.utils.data.DataLoader,
                               device: str) -> np.ndarray:
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

            # Get Probabilistic Output
            mu, sigma = model(features)

            # Calculate Z-Score
            # (Actual - Expected) / Predicted_Volatility
            z = (targets - mu) / sigma
            z_scores.append(z.cpu().numpy())

    return np.concatenate(z_scores)


class EVTEngine:
    """
    Extreme Value Theory Engine for Risk Management.
    """

    def __init__(self, tail_fraction: float = 0.10):
        self.tail_fraction = tail_fraction

    def analyze_tails(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Estimates the Tail Index (Gamma) using the Hill Estimator.
        Input should be STANDARDIZED residuals (Z-scores) for heteroscedastic models.
        """
        # We care about the Left Tail of Returns (Negative Z-scores)
        # Loss = -Z
        losses = -residuals
        losses = np.sort(losses)[::-1]  # Sort descending

        n = len(losses)
        k = int(n * self.tail_fraction)

        if k < 5:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": n, "k_exceedances": k}

        threshold_loss = losses[k]

        # Hill Estimator requires positive values (Tail > 0)
        # If threshold is negative, it means the 'tail' (top 10%) includes gains,
        # which implies very low volatility or bullish data. We filter.
        if threshold_loss <= 0:
            valid_losses = losses[losses > 0]
            if len(valid_losses) == 0:
                return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": n, "k_exceedances": 0}
            n = len(valid_losses)
            k = int(n * self.tail_fraction)
            losses = valid_losses
            threshold_loss = losses[k]

        log_losses = np.log(losses[:k])
        log_threshold = np.log(threshold_loss)

        gamma = np.mean(log_losses - log_threshold)

        return {
            "gamma": gamma,
            "threshold_loss": threshold_loss,  # This is the VaR of Z-scores
            "n_samples": n,
            "k_exceedances": k
        }

    def calculate_risk_metrics(self, evt_params: Dict[str, float], confidence_level: float = 0.99) -> Dict[str, float]:
        """
        Calculates VaR and ES for the Z-Score distribution.
        """
        gamma = evt_params["gamma"]
        threshold = evt_params["threshold_loss"]
        n = evt_params["n_samples"]
        k = evt_params["k_exceedances"]
        p = confidence_level

        if k == 0 or threshold == 0:
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": 0.0}

        risk_ratio = k / (n * (1 - p))
        var_evt = threshold * (risk_ratio ** gamma)

        if gamma >= 1.0:
            es_evt = float('inf')
        else:
            es_evt = var_evt / (1 - gamma)

        return {
            f"VaR_{p}": var_evt,
            f"ES_{p}": es_evt,
            "Gamma_Hill": gamma
        }

    def generate_scenarios(self,
                           n_simulations: int,
                           gamma: float,
                           volatility: float,
                           expected_return: float) -> np.ndarray:
        """
        Simulate Future Returns using the Heteroscedastic Model + EVT Shocks.

        Formula:
            Return_Sim = Mu_Pred + Sigma_Pred * Z_Sim

        Where Z_Sim is drawn from a Heavy-Tailed (Student-t) distribution
        defined by the historical Gamma.
        """
        # 1. Generate Standardized Shocks (Z)
        if gamma <= 0.01:
            # Gaussian Shocks
            z_sim = np.random.normal(0, 1, n_simulations)
        elif gamma >= 1.0:
            # Infinite Variance (Dangerous) - Clamp df
            z_sim = np.random.standard_t(df=2.0, size=n_simulations)
        else:
            # Heavy Tailed Shocks
            df = 1.0 / gamma
            z_sim = np.random.standard_t(df=df, size=n_simulations)

        # 2. Scale by Predicted Volatility and Shift by Predicted Mean
        return expected_return + (z_sim * volatility)