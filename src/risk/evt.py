import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def get_standardized_residuals(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    sigma_floor: float = 1e-3
) -> np.ndarray:
    """
    Calculates standardized residuals: Z = (y - mu) / sigma.

    Notes
    -----
    - Uses a sigma floor to avoid exploding Z when sigma is tiny.
    - Returns a 1D numpy array with non-finite values removed.
    """
    model.eval()
    z_scores = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).squeeze(-1).float()

            mu, sigma = model(features)

            sigma_safe = sigma.clamp(min=float(sigma_floor))
            z = (targets - mu) / sigma_safe

            z_scores.append(z.detach().cpu().numpy())

    if len(z_scores) == 0:
        return np.array([], dtype=float)

    out = np.concatenate(z_scores).reshape(-1)
    out = out[np.isfinite(out)]
    return out


class EVTEngine:
    """
    Extreme Value Theory Engine for tail risk on standardized residuals Z.

    Convention:
      - Z is a standardized return residual (negative is bad).
      - Loss is L = -Z (positive is bad).
      - VaR/ES returned here are POSITIVE LOSSES in Z-units.
    """

    def __init__(self, tail_fraction: float = 0.10):
        self.tail_fraction = float(tail_fraction)

    def analyze_tails(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Estimate tail index gamma using the Hill estimator on losses L = -Z.
        """
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        residuals = residuals[np.isfinite(residuals)]

        if residuals.size < 10:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(residuals.size), "k_exceedances": 0.0}

        losses = -residuals  # L = -Z
        losses = losses[np.isfinite(losses)]
        losses = np.sort(losses)[::-1]  # descending

        n = int(losses.size)
        if n < 10:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": 0.0}

        k = int(n * self.tail_fraction)
        k = max(0, min(k, n - 1))
        if k < 5:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": float(k)}

        threshold_loss = float(losses[k])

        # If threshold <= 0, tail contains non-loss values; restrict to positive losses
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

        top = losses[:k]
        top = top[top > 0.0]
        if top.size < 2:
            return {"gamma": 0.0, "threshold_loss": threshold_loss, "n_samples": float(n), "k_exceedances": float(k)}

        gamma = float(np.mean(np.log(top) - np.log(threshold_loss)))

        return {
            "gamma": gamma,
            "threshold_loss": threshold_loss,
            "n_samples": float(n),
            "k_exceedances": float(k),
        }

    def calculate_risk_metrics(self, evt_params: Dict[str, float], confidence_level: float = 0.99) -> Dict[str, float]:
        """
        Compute EVT VaR and ES for LOSS distribution L = -Z.

        Returns:
          - VaR_p: positive loss at confidence p
          - ES_p: positive expected shortfall at confidence p
        """
        p = float(confidence_level)
        gamma = float(evt_params.get("gamma", 0.0))
        threshold = float(evt_params.get("threshold_loss", 0.0))
        n = float(evt_params.get("n_samples", 0.0))
        k = float(evt_params.get("k_exceedances", 0.0))

        if k <= 0 or threshold <= 0 or n <= 0 or not (0.0 < p < 1.0):
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": 0.0}

        denom = n * (1.0 - p)
        if denom <= 0:
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": gamma}

        risk_ratio = k / denom
        if risk_ratio <= 0:
            return {f"VaR_{p}": 0.0, f"ES_{p}": 0.0, "Gamma_Hill": gamma}

        var_evt = threshold * (risk_ratio ** gamma)

        if gamma >= 1.0:
            es_evt = float("inf")
        else:
            es_evt = var_evt / (1.0 - gamma)

        return {
            f"VaR_{p}": float(var_evt),
            f"ES_{p}": float(es_evt),
            "Gamma_Hill": float(gamma),
        }

    def generate_scenarios(
        self,
        n_simulations: int,
        gamma: float,
        volatility: float,
        expected_return: float
    ) -> np.ndarray:
        """
        Simulate future returns:
            R_sim = expected_return + volatility * Z_sim

        Z_sim is drawn from a heavy-tailed distribution implied by gamma.
        """
        n_simulations = int(n_simulations)
        gamma = float(gamma)
        volatility = float(volatility)
        expected_return = float(expected_return)

        if n_simulations <= 0:
            return np.array([], dtype=float)

        if gamma <= 0.01:
            z_sim = np.random.normal(0.0, 1.0, n_simulations)
        elif gamma >= 1.0:
            z_sim = np.random.standard_t(df=2.0, size=n_simulations)
        else:
            df = max(2.0, 1.0 / gamma)
            z_sim = np.random.standard_t(df=df, size=n_simulations)

        return expected_return + (z_sim * volatility)
