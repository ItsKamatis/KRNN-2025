import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def get_standardized_residuals(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    sigma_floor: float = 1e-3,
) -> np.ndarray:
    """
    Calculates standardized residuals: Z = (y - mu) / sigma.

    Notes:
    - We clamp sigma from below to avoid inf/NaN residuals if sigma becomes tiny.
    - We drop non-finite residuals before returning.
    """
    model.eval()
    chunks = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).squeeze(-1).float()

            mu, sigma = model(features)

            sigma_safe = sigma.clamp(min=float(sigma_floor))
            z = (targets - mu) / sigma_safe
            chunks.append(z.detach().cpu().numpy())

    if len(chunks) == 0:
        return np.array([], dtype=float)

    out = np.concatenate(chunks).reshape(-1)
    out = out[np.isfinite(out)]
    return out


class EVTEngine:
    """
    Extreme Value Theory Engine for Risk Management.

    Conventions (Z-space):
      - Input residuals are standardized returns Z = (y - mu)/sigma
      - Bad outcomes are in the LEFT tail of Z (negative values)
      - We define losses as L = -Z (so large positive L means bad)
      - VaR/ES returned by calculate_risk_metrics are positive LOSS metrics in Z-units.
    """

    def __init__(self, tail_fraction: float = 0.10):
        self.tail_fraction = float(tail_fraction)

    def analyze_tails(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Estimate tail index (gamma) using the Hill estimator on LOSS = -Z.

        Returns:
          gamma: Hill tail index
          threshold_loss: u (loss threshold)
          n_samples: n
          k_exceedances: k
        """
        r = np.asarray(residuals, dtype=float).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size < 10:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(r.size), "k_exceedances": 0.0}

        # Loss = -Z
        losses = -r
        losses = losses[np.isfinite(losses)]
        losses = np.sort(losses)[::-1]  # descending

        n = int(losses.size)
        k = int(n * self.tail_fraction)

        if k < 5:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(n), "k_exceedances": float(k)}

        threshold_loss = float(losses[k])

        # Hill requires positive tail losses. If threshold <= 0, tail contains non-losses; filter.
        if threshold_loss <= 0.0:
            valid = losses[losses > 0.0]
            if valid.size < 5:
                return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": float(valid.size), "k_exceedances": 0.0}

            losses = valid
            n = int(losses.size)
            k = int(n * self.tail_fraction)
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
        if not np.isfinite(gamma) or gamma < 0.0:
            gamma = 0.0

        return {
            "gamma": gamma,
            "threshold_loss": threshold_loss,
            "n_samples": float(n),
            "k_exceedances": float(k),
        }

    def calculate_risk_metrics(self, evt_params: Dict[str, float], confidence_level: float = 0.99) -> Dict[str, float]:
        """
        Compute EVT VaR/ES for LOSS = -Z at confidence_level p (e.g. p=0.95).

        Returns positive loss metrics in Z-units.
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
        expected_return: float,
    ) -> np.ndarray:
        """
        Simulate future returns:
          R_sim = mu_pred + sigma_pred * Z_sim
        where Z_sim is heavy-tailed based on gamma.
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
