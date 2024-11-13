# tests/test_features_v5.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import talib
from datetime import datetime, timedelta

from src.data.features_v5 import FeatureEngineer, FeatureConfig, FeaturePipeline


@pytest.fixture
def sample_stock_data():
    """Create realistic sample stock data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Generate somewhat realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    data = {
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.001, 100)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'Close': prices,
        'Volume': np.random.lognormal(10, 1, 100).astype(int)
    }

    return pd.DataFrame(data)


@pytest.fixture
def feature_config():
    """Create feature configuration."""
    return FeatureConfig(
        window_size=5,
        indicators=['RSI', 'MACD', 'ATR']
    )


@pytest.fixture
def engineer(feature_config):
    """Create feature engineer instance."""
    return FeatureEngineer(feature_config)


class TestFeatureEngineer:
    """Test feature engineering calculations."""

    @staticmethod
    def test_price_features(engineer, sample_stock_data):
        """Test price-based feature calculations."""
        features = engineer._calculate_price_features(sample_stock_data)

        # Check expected columns exist
        expected_cols = ['Open_MA', 'High_MA', 'Low_MA', 'Close_MA',
                         'HL_Spread', 'OC_Spread']
        assert all(col in features.columns for col in expected_cols)

        # Verify calculations
        window = engineer.config.window_size
        pd.testing.assert_series_equal(
            features['Open_MA'],
            sample_stock_data['Open'].rolling(window=window, min_periods=1).mean(),
            check_names=False
        )

        # Check spread calculations
        assert (features['HL_Spread'] ==
                sample_stock_data['High'] - sample_stock_data['Low']).all()

    @staticmethod
    def test_technical_indicators(engineer, sample_stock_data):
        """Test technical indicator calculations."""
        features = engineer._calculate_indicators(sample_stock_data)

        # Check RSI
        if 'RSI' in engineer.config.indicators:
            expected_rsi = talib.RSI(sample_stock_data['Close'])
            pd.testing.assert_series_equal(
                features['RSI'],
                expected_rsi,
                check_names=False
            )

        # Check basic properties
        if 'MACD' in engineer.config.indicators:
            assert not features['MACD'].isnull().all()

        # Verify Bollinger Bands
        if 'BB_UPPER' in features.columns:
            assert (features['BB_UPPER'] > features['BB_LOWER']).all()

    @staticmethod
    def test_volume_features(engineer, sample_stock_data):
        """Test volume feature calculations."""
        features = engineer._calculate_volume_features(sample_stock_data)

        assert 'Volume_MA' in features.columns
        assert 'Volume_Volatility' in features.columns

        # Check volume MA calculation
        pd.testing.assert_series_equal(
            features['Volume_MA'],
            sample_stock_data['Volume'].rolling(
                window=engineer.config.window_size,
                min_periods=1
            ).mean(),
            check_names=False
        )

    @staticmethod
    def test_returns(engineer, sample_stock_data):
        """Test return-based feature calculations."""
        features = engineer._calculate_returns(sample_stock_data)

        # Check log returns
        expected_returns = np.log(
            sample_stock_data['Close'] / sample_stock_data['Close'].shift(1)
        )
        pd.testing.assert_series_equal(
            features['LogReturn'],
            expected_returns,
            check_names=False
        )

        # Check volatility
        assert 'ReturnVolatility' in features.columns
        assert not features['ReturnVolatility'].isnull().all()

    @staticmethod
    def test_complete_feature_calculation(engineer, sample_stock_data):
        """Test end-to-end feature calculation."""
        features = engineer.calculate_features(sample_stock_data)

        # Check basics
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert not features.isnull().any().any()

        # Check index
        assert isinstance(features.index, pd.DatetimeIndex)

        # Verify feature groups present
        price_features = [col for col in features.columns if '_MA' in col]
        tech_features = [col for col in features.columns
                         if col in engineer.config.indicators]

        assert len(price_features) > 0
        assert len(tech_features) > 0


class TestFeaturePipeline:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def sample_stock_files(self, tmp_path, sample_stock_data):
        """Create sample stock files."""
        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()

        # Create multiple stock files
        for ticker in ['AAPL', 'GOOGL', 'MSFT']:
            stock_data = sample_stock_data.copy()
            stock_data['Ticker'] = ticker
            stock_data.to_csv(stock_dir / f"{ticker}.csv", index=False)

        return stock_dir

    @staticmethod
    def test_process_stock(sample_stock_files):
        """Test processing single stock."""
        pipeline = FeaturePipeline()
        stock_file = next(sample_stock_files.glob('*.csv'))

        features, ticker = pipeline._process_stock(stock_file)

        assert isinstance(features, pd.DataFrame)
        assert isinstance(ticker, str)
        assert not features.empty
        assert not features.isnull().any().any()

    @staticmethod
    def test_process_stocks(sample_stock_files):
        """Test processing multiple stocks."""
        pipeline = FeaturePipeline()
        output_dir = sample_stock_files.parent / "features"

        features = pipeline.process_stocks(
            list(sample_stock_files.glob('*.csv')),
            output_dir
        )

        # Check results
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert 'Ticker' in features.columns
        assert len(features['Ticker'].unique()) == 3

        # Check output file
        assert (output_dir / 'features.parquet').exists()

    @staticmethod
    def test_error_handling(sample_stock_files):
        """Test error handling for invalid files."""
        pipeline = FeaturePipeline()

        # Create invalid file
        invalid_file = sample_stock_files / "invalid.csv"
        invalid_file.write_text("invalid,data\n1,2")

        # Should skip invalid file but process others
        output_dir = sample_stock_files.parent / "features"
        features = pipeline.process_stocks(
            list(sample_stock_files.glob('*.csv')),
            output_dir
        )

        assert isinstance(features, pd.DataFrame)
        assert not features.empty