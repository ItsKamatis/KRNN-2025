import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimization
        lr = config.get('training', {}).get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['training']['scheduler_patience'],
            factor=config['training']['scheduler_factor'],
            verbose=True
        )

        # Heteroscedastic Loss
        self.criterion = nn.GaussianNLLLoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device).squeeze(-1)

            self.optimizer.zero_grad()
            mu, sigma = self.model(features)
            var = sigma.pow(2)
            loss = self.criterion(mu, targets, var)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader):
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

    def train(self, train_loader, val_loader, experiment=None):
        epochs = self.config['training']['epochs']
        patience = self.config['training']['patience']
        min_delta = self.config['training']['min_delta']

        best_val_loss = float('inf')
        no_improve_count = 0

        logger.info(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Train GNLL: {train_loss:.6f} | Val GNLL: {val_loss:.6f}")

            # Early Stopping Logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                no_improve_count = 0
                # Save checkpoint logic could go here
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(f"Early Stopping triggered at Epoch {epoch + 1}")
                    break

        return {'loss': best_val_loss}