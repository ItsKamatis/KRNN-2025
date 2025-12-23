# src/model/krnn_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple
from dataclasses import dataclass

# src/model/krnn_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Probabilistic K-Parallel RNN."""
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    k_dups: int = 3
    # Output dim is 2 for Heteroscedastic Regression (Mu, Sigma)
    output_dim: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KParallelEncoder(nn.Module):
    """
    The Core 'KRNN' Logic.
    Creates K parallel RNNs and averages their outputs to reduce variance.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.k_dups = config.k_dups
        self.hidden_dim = config.hidden_dim
        self.device = config.device

        # Create K independent RNN instances
        self.rnn_modules = nn.ModuleList()
        for _ in range(self.k_dups):
            self.rnn_modules.append(
                nn.GRU(
                    input_size=config.input_dim,
                    hidden_size=config.hidden_dim,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=False
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Features]
        Returns:
            Averaged Hidden Representation [Batch, Hidden_Dim]
        """
        parallel_outputs = []

        for rnn in self.rnn_modules:
            out, _ = rnn(x)
            # Take the output of the last time step
            last_step_out = out[:, -1, :]  # [batch, hidden]
            parallel_outputs.append(last_step_out)

        # Stack and Mean Pool
        stacked_outputs = torch.stack(parallel_outputs, dim=-1)
        krnn_representation = torch.mean(stacked_outputs, dim=-1)

        return krnn_representation


class KRNNRegressor(nn.Module):
    """
    Probabilistic KRNN Regressor.
    Predicts both Expected Return (Mu) and Volatility (Sigma).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = KParallelEncoder(config)

        # We need 2 outputs: Mu (Mean) and Sigma (Std Dev)
        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2)
        )
        self.to(config.device)

    def forward(self, x: torch.Tensor):
        # 1. Encode
        encoded = self.encoder(x)

        # 2. Predict parameters
        out = self.regressor(encoded)  # [Batch, 2]

        # Split output
        mu = out[:, 0]  # Predicted Mean Return
        log_var = out[:, 1]  # Raw output for volatility

        # Enforce positivity for Sigma using Softplus
        # Add epsilon to prevent division by zero in loss function
        sigma = F.softplus(log_var)
        # Stabilize: prevent extremely small/large sigma which can cause inf/nan in NLL
        sigma = sigma.clamp(min=1e-3, max=10.0)

        return mu, sigma

class KRNN(nn.Module):
    """K-rare class nearest neighbor enhanced RNN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Calculate dimensions
        self.rnn_hidden = config.hidden_dim
        self.rnn_directions = 2 if config.bidirectional else 1

        # Feature projection
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # GRU layer
        self.rnn = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(
            config.hidden_dim * self.rnn_directions,
            config.num_classes
        )

        self._init_weights()
        self.to(config.device)

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention mechanism."""
        # Project features
        x = self.feature_proj(x)
        x = F.relu(x)

        # RNN processing
        rnn_out, _ = self.rnn(x)

        # Self-attention
        attention = F.softmax(
            torch.bmm(rnn_out, rnn_out.transpose(1, 2)) /
            torch.sqrt(torch.tensor(self.rnn_hidden, dtype=torch.float32, device=x.device)),
            dim=2
        )

        # Apply attention
        context = torch.bmm(attention, rnn_out)

        # Classification
        pooled = context[:, -1]  # Use last timestep
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits, attention

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions with no gradient tracking."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            return F.softmax(logits, dim=1)


class KRNNPredictor:
    """Handles training and prediction."""

    def __init__(self, model: KRNN):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
            # Removed verbose=True to address deprecation warning
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.model.config.device)
            targets = targets.to(self.model.config.device).squeeze(1)  # Squeeze target tensor

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(val_loader):
            features = features.to(self.model.config.device)
            targets = targets.to(self.model.config.device).squeeze(1)  # Squeeze target tensor

            # Forward pass
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets)

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }

        self.scheduler.step(metrics['loss'])
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.model.config
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.config.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
