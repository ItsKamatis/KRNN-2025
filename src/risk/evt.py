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

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            # FIX: Squeeze targets to match preds shape [Batch]
            targets = targets.to(device).squeeze(-1).float()

            preds = model(features)

            # Now shapes align: [Batch] - [Batch]
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
        """
        Main function to estimate tail risk metrics.
        """
        # 1. Isolate "Losses"
        # We care about the Left Tail of returns, which corresponds to negative residuals.
        # We negate them to deal with positive numbers: Loss = -(y_true - y_pred)
        # We only look at cases where we lost money (or lost more than predicted).
        losses = -residuals

        # Filter for the "Right Tail" of the Loss distribution (which is the Left Tail of returns)
        # We sort descending: L_1 >= L_2 >= ...
        losses = np.sort(losses)[::-1]

        # 2. Determine Threshold 'k' (Number of exceedances)
        n = len(losses)
        k = int(n * self.tail_fraction)

        if k < 10:
            logger.warning(f"Not enough tail samples (k={k}) for reliable Hill estimation.")
            # Fallback or strict error handling depending on preference

        # 3. Hill Estimator Calculation
        # Formula: (1/k) * Sum( ln(L_i) - ln(L_{k+1}) )
        # Note: We take losses[0] to losses[k-1] as the top k.
        # losses[k] corresponds to L_{k+1} because of 0-based indexing.
        threshold_loss = losses[k]  # L_{k+1}

        # Avoid log(0) or log(negative) issues
        if threshold_loss <= 0:
            # Shift distribution if losses are not strictly positive (common practical hack)
            # or just take the positive subset
            valid_losses = losses[losses > 0]
            n = len(valid_losses)
            k = int(n * self.tail_fraction)
            losses = valid_losses
            threshold_loss = losses[k]

        log_losses = np.log(losses[:k])
        log_threshold = np.log(threshold_loss)

        gamma = np.mean(log_losses - log_threshold)  # The Hill Estimator (shape parameter)

        return {
            "gamma": gamma,
            "threshold_loss": threshold_loss,
            "n_samples": n,
            "k_exceedances": k
        }

    def calculate_risk_metrics(self,
                               evt_params: Dict[str, float],
                               confidence_level: float = 0.99) -> Dict[str, float]:
        """
        Calculates VaR and ES using the EVT parameters.
        """
        gamma = evt_params["gamma"]
        threshold = evt_params["threshold_loss"]
        n = evt_params["n_samples"]
        k = evt_params["k_exceedances"]
        p = confidence_level

        # 1. EVT Value at Risk (VaR)
        # Formula: L_{k+1} * ( k / (n * (1-p)) ) ^ gamma
        # This extrapolates the tail shape to the desired confidence level p
        risk_ratio = k / (n * (1 - p))
        var_evt = threshold * (risk_ratio ** gamma)

        # 2. EVT Expected Shortfall (ES / CVaR)
        # Formula: VaR / (1 - gamma) + (expected mean adjustment if needed)
        # Simple approximation for GPD: ES = VaR / (1 - gamma)
        # Constraint: gamma must be < 1. If gamma >= 1, the tail is so fat the mean is infinite.
        if gamma >= 1.0:
            es_evt = float('inf')
            logger.error("Tail index gamma >= 1.0! Variance is infinite. Risk is unmanageable.")
        else:
            es_evt = var_evt / (1 - gamma)

        return {
            f"VaR_{p}": var_evt,
            f"ES_{p}": es_evt,
            "Gamma_Hill": gamma
        }


# Helper function to run the full analysis
def calculate_portfolio_risk(model, dataloader, device):
    residuals = get_residuals(model, dataloader, device)

    engine = EVTEngine(tail_fraction=0.10)  # Look at top 10% worst errors

    # 1. Fit the tail
    evt_params = engine.analyze_tails(residuals)

    # 2. Calculate Metrics (e.g., 99% confidence)
    metrics = engine.calculate_risk_metrics(evt_params, confidence_level=0.99)

    return metrics