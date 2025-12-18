# src/data/dataset_v5.py
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 64
    sequence_length: int = 10  # This matches the argument name in your class
    train_shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class StockDataset(Dataset):
    """
    Memory efficient dataset for stock data.
    Updated for REGRESSION (Plan B).
    """

    def __init__(
            self,
            features: pd.DataFrame,
            sequence_length: int,
            target_column: str = 'Target'  # Changed default from 'Label' to 'Target'
    ):
        self.sequence_length = sequence_length
        self.target_column = target_column

        # Get feature columns (exclude Date, Ticker, and target)
        self.feature_cols = [
            col for col in features.columns
            if col not in ['Date', 'Ticker', target_column]
        ]

        # Ensure we don't accidentally include object columns (like strings)
        self.feature_cols = [c for c in self.feature_cols if pd.api.types.is_numeric_dtype(features[c])]

        # Sort data by Ticker and Date
        features = features.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

        # Initialize data storage
        self.data = []

        # Group data by ticker
        groups = features.groupby('Ticker')

        # Build sequences
        for _, group in groups:
            group_size = len(group)
            if group_size >= self.sequence_length:
                # We iterate such that we can get a sequence of length N and the Target at N
                # The target for sequence [t-N : t] is usually at t (next step prediction)
                # Assuming 'Target' column is already shifted in DataCollector
                for i in range(self.sequence_length, group_size + 1):
                    seq_data = group.iloc[i - self.sequence_length:i]
                    self.data.append(seq_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sequence of features and target."""
        seq_data = self.data[idx]

        # Use all rows in the sequence for features
        feature_sequence = seq_data[self.feature_cols].values

        # The target is the value associated with the LAST step of the window
        # (Assuming the target column was pre-shifted in data_collector to represent t+1)
        target = seq_data[self.target_column].values[-1]

        # Return feature sequence and Float tensor for Regression
        return (
            torch.FloatTensor(feature_sequence),
            torch.tensor([target], dtype=torch.float32)  # Float for Regression, No "target-1"
        )


class DataModule:
    """Handles data loading and preparation."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, data_dir: Path) -> None:
        """Load and prepare datasets."""
        # Load data splits
        try:
            train_data = pd.read_parquet(data_dir / 'train.parquet')
            val_data = pd.read_parquet(data_dir / 'validation.parquet')
            test_data = pd.read_parquet(data_dir / 'test.parquet')
        except FileNotFoundError as e:
            logger.error(f"Data files not found in {data_dir}. Run data collection first.")
            raise e

        # Create datasets
        self.train_dataset = StockDataset(
            train_data,
            self.config.sequence_length
        )
        self.val_dataset = StockDataset(
            val_data,
            self.config.sequence_length
        )
        self.test_dataset = StockDataset(
            test_data,
            self.config.sequence_length
        )

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for each split."""
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            raise RuntimeError("Datasets not initialized. Call setup() first.")

        return {
            'train': DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.train_shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'val': DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        }