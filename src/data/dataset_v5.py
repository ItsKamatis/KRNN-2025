# src/data/dataset_v5.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 64
    sequence_length: int = 10
    train_shuffle: bool = True
    num_workers: int = 0  # Set to 0 for Windows if you see pickling errors, else 4
    pin_memory: bool = True


class StockDataset(Dataset):
    """
    Optimized Tensor-based Dataset.
    Pre-processes data into RAM to prevent CPU bottlenecks during training.
    """

    def __init__(
            self,
            features: pd.DataFrame,
            sequence_length: int,
            target_column: str = 'Target'
    ):
        self.sequence_length = sequence_length

        # 1. Identify Feature Columns
        # Prefer the engineered, scale-invariant features if present.
        preferred = [
            'Log_Return', 'RSI', 'MACD_Rel', 'MACD_Sig_Rel',
            'BB_Width', 'BB_Pos', 'ATR_Rel', 'Log_Volume'
        ]
        present = [c for c in preferred if c in features.columns]

        if len(present) >= 4:
            self.feature_cols = present
        else:
            # Fallback: any numeric column except identifiers + target
            self.feature_cols = [
                col for col in features.columns
                if col not in ['Date', 'Ticker', target_column]
                   and pd.api.types.is_numeric_dtype(features[col])
            ]


        # 2. Sort to ensure time coherence
        df_sorted = features.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        # 2.5 Sanitize: remove NaN/inf in features/target so training doesn't silently become NaN
        df_sorted = df_sorted.replace([np.inf, -np.inf], np.nan)
        before = len(df_sorted)
        df_sorted = df_sorted.dropna(subset=self.feature_cols + [target_column])
        dropped = before - len(df_sorted)
        if dropped > 0:
            logger.warning(f"Dropped {dropped:,} rows with non-finite values in features/target.")


        # 3. Pre-allocate lists to hold the Tensors
        # Converting to numpy first is much faster than iterating rows
        all_feats_list = []
        all_targets_list = []

        # Group by Ticker to ensure we don't mix data from different stocks in one window
        # (This part is still Python-loop heavy but only happens ONCE at startup)
        grouped = df_sorted.groupby('Ticker')

        for ticker, group in grouped:
            # Extract numpy arrays for this ticker
            group_feats = group[self.feature_cols].values.astype(np.float32)
            group_targets = group[target_column].values.astype(np.float32)
            # Guard against any lingering non-finite values (should be rare after dropna)
            if not np.isfinite(group_feats).all() or not np.isfinite(group_targets).all():
                continue


            num_samples = len(group)
            if num_samples <= sequence_length:
                continue

            # Rolling window view (efficient numpy stride trick or simple loop)
            # For clarity and safety, we'll use a loop, but append to lists
            # We want input [t-seq : t] and target [t] (which is already shifted in data_collector)

            # Vectorized sliding window creation is faster:
            # Create indices:
            # [[0, 1, ..., 9], [1, 2, ..., 10], ...]
            # This consumes more RAM but is instant for training.

            for i in range(sequence_length, num_samples + 1):
                # Input: Window of length 'sequence_length' ending at i
                # Target: The value at i-1 (since we already shifted target in Collector)
                # But wait, your DataCollector shifts Target by -1.
                # So row 'i' in the DF contains Features(t) and Target(t+1).
                # We want the window to predict the target at the end.

                # Slicing:
                window_feats = group_feats[i - sequence_length: i]
                # The target for this sequence is the target associated with the last step
                target_val = group_targets[i - 1]

                all_feats_list.append(window_feats)
                all_targets_list.append(target_val)

        # 4. Stack into one massive Tensor (The "RAM Cache")
        # Shape: [Total_Samples, Seq_Len, Features]
        if len(all_feats_list) == 0:
            raise ValueError("Not enough data to create sequences. Check sequence_length vs data size.")

        self.X = torch.from_numpy(np.stack(all_feats_list))
        self.y = torch.from_numpy(np.array(all_targets_list))

        logger.info(f"Dataset loaded into RAM: X shape {self.X.shape}, y shape {self.y.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Zero overhead access
        return self.X[idx], self.y[idx]


class DataModule:
    """Handles data loading and preparation."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, data_dir: Path) -> None:
        try:
            train_data = pd.read_parquet(data_dir / 'train.parquet')
            val_data = pd.read_parquet(data_dir / 'validation.parquet')
            test_data = pd.read_parquet(data_dir / 'test.parquet')
        except FileNotFoundError:
            raise RuntimeError("Data files not found.")

        self.train_dataset = StockDataset(train_data, self.config.sequence_length)
        self.val_dataset = StockDataset(val_data, self.config.sequence_length)
        self.test_dataset = StockDataset(test_data, self.config.sequence_length)

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        # Note: On Windows, num_workers>0 can duplicate RAM due to spawn/pickle.
        # Keep num_workers=0 unless you're sure it's stable.
        kwargs_common = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": (self.config.num_workers > 0),
        }
        return {
            'train': DataLoader(
                self.train_dataset,
                shuffle=self.config.train_shuffle,
                drop_last=True,  # consistent batch shapes helps GPU kernels
                **kwargs_common,
            ),
            'val': DataLoader(self.val_dataset, shuffle=False, **kwargs_common),
            'test': DataLoader(self.test_dataset, shuffle=False, **kwargs_common)
        }