# src/utils/trainer_v5.py

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Trainer:
    """Lightweight trainer for the heteroscedastic KRNN.

    Speed-focused defaults for RTX 4090:
      - AMP autocast (bf16 by default)
      - non_blocking H2D transfers (works with pin_memory=True)
      - zero_grad(set_to_none=True)

    You can control these via config['training']:
      use_amp: bool (default True when CUDA)
      amp_dtype: 'bf16' or 'fp16' (default 'bf16')
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config

        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        training_cfg = config.get('training', {})
        lr = training_cfg.get('learning_rate', 0.001)

        # Optimizer
        # Adam is fine; AdamW can be slightly better. Keep existing behavior unless configured.
        opt_name = str(training_cfg.get('optimizer', 'adam')).lower()
        if opt_name in {'adamw', 'adam_w'}:
            # Use fused AdamW if available (PyTorch 2.x + CUDA)
            try:
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=(self.device.startswith('cuda')))
            except TypeError:
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            # Use fused Adam if available
            try:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=(self.device.startswith('cuda')))
            except TypeError:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=training_cfg.get('scheduler_patience', 5),
            factor=training_cfg.get('scheduler_factor', 0.5)
        )

        # Heteroscedastic Loss (Gaussian Negative Log Likelihood)
        self.criterion = nn.GaussianNLLLoss()

        # AMP
        self.use_amp: bool = bool(training_cfg.get('use_amp', torch.cuda.is_available() and self.device.startswith('cuda')))
        amp_dtype = str(training_cfg.get('amp_dtype', 'bf16')).lower()
        if amp_dtype not in {'bf16', 'fp16'}:
            amp_dtype = 'bf16'

        self.autocast_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16

        # GradScaler is only needed for fp16. For bf16, scaling is usually unnecessary.
        self.scaler: Optional[torch.cuda.amp.GradScaler]
        if self.use_amp and self.device.startswith('cuda') and self.autocast_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _gnll_loss(self, mu: torch.Tensor, sigma: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute GaussianNLLLoss in float32 for numerical stability.

        We keep the forward pass under autocast (if enabled), but cast to float32
        for the variance + loss computation to avoid bf16/fp16 edge cases.
        """
        mu_f = mu.float()
        sigma_f = sigma.float()
        targets_f = targets.float()

        # Variance must be strictly positive for GaussianNLLLoss
        var = (sigma_f * sigma_f).clamp_min(1e-8)
        return self.criterion(mu_f, targets_f, var)

    @staticmethod
    def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
        t_det = t.detach()
        # Flatten to 1D for robust stats
        t_flat = t_det.reshape(-1)
        finite = torch.isfinite(t_flat)
        if finite.any():
            t_fin = t_flat[finite]
            return {
                'finite_frac': float(finite.float().mean().item()),
                'min': float(t_fin.min().item()),
                'max': float(t_fin.max().item()),
                'mean': float(t_fin.mean().item()),
                'std': float(t_fin.std(unbiased=False).item()),
            }
        return {'finite_frac': 0.0, 'min': float('nan'), 'max': float('nan'), 'mean': float('nan'), 'std': float('nan')}

    def _raise_nonfinite(self, *, loss: torch.Tensor, features: torch.Tensor, targets: torch.Tensor,
                         mu: torch.Tensor, sigma: torch.Tensor) -> None:
        logger.error("Non-finite loss encountered. Dumping tensor stats:")
        logger.error(f"  loss: {loss.item()}")
        logger.error(f"  features: {self._tensor_stats(features)}")
        logger.error(f"  targets:  {self._tensor_stats(targets)}")
        logger.error(f"  mu:       {self._tensor_stats(mu)}")
        logger.error(f"  sigma:    {self._tensor_stats(sigma)}")
        raise FloatingPointError("Non-finite loss. Most common cause: NaN/inf in the dataset (features/Target).")
    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for features, targets in loader:
            # non_blocking only helps if loader uses pin_memory
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).squeeze(-1)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.device.startswith('cuda'):
                with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                    mu, sigma = self.model(features)
                loss = self._gnll_loss(mu, sigma, targets)

                if not torch.isfinite(loss):
                    self._raise_nonfinite(loss=loss, features=features, targets=targets, mu=mu, sigma=sigma)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
            else:
                mu, sigma = self.model(features)
                loss = self._gnll_loss(mu, sigma, targets)

                if not torch.isfinite(loss):
                    self._raise_nonfinite(loss=loss, features=features, targets=targets, mu=mu, sigma=sigma)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += float(loss.item())

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).squeeze(-1)

                if self.use_amp and self.device.startswith('cuda'):
                    with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                        mu, sigma = self.model(features)
                else:
                    mu, sigma = self.model(features)

                loss = self._gnll_loss(mu, sigma, targets)

                if not torch.isfinite(loss):
                    self._raise_nonfinite(loss=loss, features=features, targets=targets, mu=mu, sigma=sigma)

                total_loss += float(loss.item())

        return total_loss / max(1, len(loader))

    def get_lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return float(param_group['lr'])
        return 0.0

    def train(self, train_loader, val_loader, experiment=None) -> Dict[str, float]:
        epochs = int(self.config.get('training', {}).get('epochs', 10))
        patience = int(self.config.get('training', {}).get('patience', 10))
        min_delta = float(self.config.get('training', {}).get('min_delta', 0.0))

        best_val_loss = float('inf')
        no_improve_count = 0

        logger.info(f"Starting training for {epochs} epochs on {self.device} (AMP={self.use_amp}, dtype={self.autocast_dtype})")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            prev_lr = self.get_lr()
            self.scheduler.step(val_loss)
            curr_lr = self.get_lr()

            lr_msg = ""
            if curr_lr != prev_lr:
                lr_msg = f" | LR Reduced: {prev_lr:.1e} -> {curr_lr:.1e}"

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train GNLL: {train_loss:.6f} | Val GNLL: {val_loss:.6f}{lr_msg}"
            )

            if experiment:
                experiment.log_metrics({'train_loss': train_loss, 'val_loss': val_loss, 'lr': curr_lr}, step=epoch)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(f"Early Stopping triggered at Epoch {epoch + 1} (No improvement for {patience} epochs)")
                    break

        return {'loss': best_val_loss, 'final_train_loss': train_loss}
