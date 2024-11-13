# tests/test_dataset_v5.py
import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from src.data.dataset_v5 import StockDataset, DataConfig, DataModule, TimeSeriesSplitter


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'Ticker': ['AAPL'] * 100,
        'Open': np.random.randn(100),
        'High': np.random.randn(100),
        'Low': np.random.randn(100),
        'Close': np.random.randn(100),
        'Volume': np.random.randint(1000, 10000, 100),
        'Label': np.random.randint(1, 4, 100)  # 1, 2, or 3
    }
    return pd.DataFrame(data)


@pytest.fixture
def dataset(sample_stock_data):
    """Create dataset instance for testing."""
    return StockDataset(sample_stock_data, sequence_length=10)


class TestStockDataset:
    """Test StockDataset functionality."""

    @staticmethod
    def test_initialization(dataset, sample_stock_data):
        """Test dataset initialization."""
        assert isinstance(dataset, StockDataset)
        assert len(dataset.feature_cols) == len(sample_stock_data.columns) - 3  # Exclude Date, Ticker, Label
        assert dataset.sequence_length == 10

    @staticmethod
    def test_valid_indices(dataset):
        """Test valid indices generation."""
        assert len(dataset.valid_indices) > 0
        assert dataset.valid_indices[0] >= dataset.sequence_length - 1

    @staticmethod
    def test_getitem(dataset):
        """Test item retrieval."""
        features, target = dataset[0]

        # Check shapes
        assert features.shape == (10, len(dataset.feature_cols))
        assert target.shape == (1,)

        # Check types
        assert isinstance(features, torch.FloatTensor)
        assert isinstance(target, torch.LongTensor)

        # Check value ranges
        assert target >= 0 and target < 3  # 0-based indexing

    @staticmethod
    def test_length(dataset):
        """Test dataset length."""
        assert len(dataset) == len(dataset.valid_indices)
        assert len(dataset) > 0


class TestDataModule:
    """Test DataModule functionality."""

    @pytest.fixture
    def data_module(self, tmp_path):
        """Create DataModule instance."""
        config = DataConfig(batch_size=32, sequence_length=10)
        return DataModule(config)

    @pytest.fixture
    def sample_data_files(self, tmp_path, sample_stock_data):
        """Create sample data files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Save splits
        for split in ['train', 'validation', 'test']:
            split_data = sample_stock_data.copy()
            split_data.to_parquet(data_dir / f"{split}.parquet")

        return data_dir

    @staticmethod
    def test_setup(data_module, sample_data_files):
        """Test data module setup."""
        data_module.setup(sample_data_files)

        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None

    @staticmethod
    def test_dataloaders(data_module, sample_data_files):
        """Test dataloader creation."""
        data_module.setup(sample_data_files)
        loaders = data_module.get_dataloaders()

        assert set(loaders.keys()) == {'train', 'val', 'test'}

        # Check loader properties
        for name, loader in loaders.items():
            assert loader.batch_size == data_module.config.batch_size
            assert isinstance(next(iter(loader)), tuple)
            assert len(next(iter(loader))) == 2  # features and targets


class TestTimeSeriesSplitter:
    """Test TimeSeriesSplitter functionality."""

    @pytest.fixture
    def splitter(self):
        return TimeSeriesSplitter(n_splits=3, gap_days=5, test_size=20)

    @staticmethod
    def test_split(splitter, sample_stock_data):
        """Test time series splitting."""
        splits = splitter.split(sample_stock_data)

        assert len(splits) > 0
        for train, val, test in splits:
            # Check chronological order
            assert train['Date'].max() < val['Date'].min()
            assert val['Date'].max() < test['Date'].min()

            # Check sizes
            assert len(test) == splitter.test_size

            # Check gap days
            gap_train_val = (val['Date'].min() - train['Date'].max()).days
            gap_val_test = (test['Date'].min() - val['Date'].max()).days
            assert gap_train_val >= splitter.gap_days
            assert gap_val_test >= splitter.gap_days


if __name__ == "__main__":
    pytest.main([__file__])