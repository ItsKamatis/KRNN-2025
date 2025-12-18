# src/risk/evt.py
import torch
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def get_residuals(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: str) -> np.ndarray:
    model.eval()
    residuals = []

    def get_residuals(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: str) -> np.ndarray:
        model.eval()
        residuals = []
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(device)
                # Squeeze targets to match preds shape [Batch]
                targets = targets.to(device).squeeze(-1).float()
                preds = model(features)
                batch_residuals = targets - preds
                residuals.append(batch_residuals.cpu().numpy())
        return np.concatenate(residuals)


class EVTEngine:
    """
    Extreme Value Theory Engine for Risk Management.
    Implements Hill's Estimator and GPD-based VaR/CVaR.
    """

    def __init__(self, tail_fraction: float = 0.10):
        """
        Args:
            tail_fraction: The percentage of data to consider as the "Tail" (k/n).
                          Standard values are 0.05 (5%) or 0.10 (10%).
        """
        self.tail_fraction = tail_fraction

    def analyze_tails(self, residuals: np.ndarray) -> Dict[str, float]:
        # Isolate "Losses" (Left Tail of Returns = Negative Residuals)
        losses = -residuals
        losses = np.sort(losses)[::-1]

        n = len(losses)
        k = int(n * self.tail_fraction)

        if k < 5:
            return {"gamma": 0.0, "threshold_loss": 0.0, "n_samples": n, "k_exceedances": k}

        threshold_loss = losses[k]

        # Handle non-positive threshold issues
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
            "threshold_loss": threshold_loss,
            "n_samples": n,
            "k_exceedances": k
        }

    def calculate_risk_metrics(self, evt_params: Dict[str, float], confidence_level: float = 0.99) -> Dict[str, float]:
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

    # --- NEW METHOD ---
    def generate_scenarios(self,
                           n_simulations: int,
                           gamma: float,
                           volatility: float,
                           expected_return: float = 0.0) -> np.ndarray:
        """
        Generates future return scenarios using Parametric Bootstrapping with EVT.
        Math: Uses Student-t distribution with df = 1/gamma to model heavy tails.
        """
        if gamma <= 0.01:
            # Essentially Normal
            noise = np.random.normal(0, volatility, n_simulations)
        elif gamma >= 1.0:
            # Infinite variance (Dangerous), cap at df=2 for stability
            logger.warning(f"Gamma {gamma:.4f} too high. Clamping to Student-t(df=2).")
            noise = np.random.standard_t(df=2.0, size=n_simulations) * volatility
        else:
            # Heavy Tailed Simulation
            df = 1.0 / gamma
            noise = np.random.standard_t(df=df, size=n_simulations) * volatility

        return expected_return + noise


# Helper function to run the full analysis
def calculate_portfolio_risk(model, dataloader, device):
    residuals = get_residuals(model, dataloader, device)

    engine = EVTEngine(tail_fraction=0.10)  # Look at top 10% worst errors

    # 1. Fit the tail
    evt_params = engine.analyze_tails(residuals)

    # 2. Calculate Metrics (e.g., 99% confidence)
    metrics = engine.calculate_risk_metrics(evt_params, confidence_level=0.99)

    return metrics