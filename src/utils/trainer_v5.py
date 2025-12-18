import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the training loop for the Probabilistic KRNN.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Optimization
        lr = config.get('training', {}).get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # --- CRITICAL CHANGE: Gaussian Negative Log Likelihood ---
        # Minimizing this maximizes the probability of the data given the predicted distribution.
        # It balances minimizing Error (MSE) and estimating Uncertainty (Variance).
        self.criterion = nn.GaussianNLLLoss()

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(self.device)
            targets = targets.to(self.device).squeeze(-1)

            self.optimizer.zero_grad()

            # Forward pass now returns Tuple (Mu, Sigma)
            mu, sigma = self.model(features)

            # GaussianNLLLoss takes (input, target, var)
            # We must square sigma to get variance
            var = sigma.pow(2)

            loss = self.criterion(mu, targets, var)

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader: torch.utils.data.DataLoader) -> float:
        """Runs validation."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device).squeeze(-1)

                mu, sigma = self.model(features)
                var = sigma.pow(2)

                loss = self.criterion(mu, targets, var)
                total_loss += loss.item()

        return total_loss / len(loader)

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              experiment: Any = None) -> Dict[str, float]:
        """
        Main training loop.
        """
        epochs = self.config.get('training', {}).get('epochs', 10)
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        best_val_loss = float('inf')
        train_loss = 0.0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Train GNLL: {train_loss:.6f} | Val GNLL: {val_loss:.6f}")

            if experiment:
                experiment.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return {
            'loss': best_val_loss,
            'final_train_loss': train_loss
        }