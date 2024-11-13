import numpy as np
import pandas as pd
import talib
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable feature engineering configuration."""
    window_size: int = 5  # Match trading week
    essential_columns: List[str] = None
    technical_indicators: List[str] = None

    def __post_init__(self):
        # Set defaults if not provided
        object.__setattr__(self, 'essential_columns',
                           self.essential_columns or ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        object.__setattr__(self, 'technical_indicators',
                           self.technical_indicators or ['RSI', 'MACD', 'ATR', 'BB_UPPER', 'BB_LOWER'])


class FeatureEngineer:
    """Efficient feature calculation for stock data."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate technical indicators exist in TA-Lib."""
        available_indicators = set(talib.get_functions())
        for indicator in self.config.technical_indicators:
            if indicator not in available_indicators and indicator not in ['BB_UPPER', 'BB_LOWER']:
                raise ValueError(f"Indicator not available: {indicator}")

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for a stock."""
        df = df.copy()

        # Ensure datetime index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        features = pd.DataFrame(index=df.index)

        # Price features
        features = features.join(self._calculate_price_features(df))

        # Technical indicators
        features = features.join(self._calculate_indicators(df))

        # Volume features
        features = features.join(self._calculate_volume_features(df))

        # Return features
        features = features.join(self._calculate_returns(df))

        return features.dropna()

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        features = pd.DataFrame(index=df.index)

        for col in ['Open', 'High', 'Low', 'Close']:
            # Moving averages
            features[f'{col}_MA'] = df[col].rolling(
                window=self.config.window_size, min_periods=1
            ).mean()

        # Price channels
        features['HL_Spread'] = df['High'] - df['Low']
        features['OC_Spread'] = np.abs(df['Open'] - df['Close'])

        return features

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        features = pd.DataFrame(index=df.index)

        if 'RSI' in self.config.technical_indicators:
            features['RSI'] = talib.RSI(df['Close'])

        if 'MACD' in self.config.technical_indicators:
            features['MACD'], _, _ = talib.MACD(df['Close'])

        if 'ATR' in self.config.technical_indicators:
            features['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])

        # Bollinger Bands
        if any(x in self.config.technical_indicators for x in ['BB_UPPER', 'BB_LOWER']):
            upper, middle, lower = talib.BBANDS(df['Close'])
            if 'BB_UPPER' in self.config.technical_indicators:
                features['BB_UPPER'] = upper
            if 'BB_LOWER' in self.config.technical_indicators:
                features['BB_LOWER'] = lower

        return features

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=df.index)

        # Volume moving average
        features['Volume_MA'] = df['Volume'].rolling(
            window=self.config.window_size, min_periods=1
        ).mean()

        # Volume volatility
        features['Volume_Volatility'] = df['Volume'].rolling(
            window=self.config.window_size, min_periods=1
        ).std()

        return features

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate return-based features."""
        features = pd.DataFrame(index=df.index)

        # Log returns
        features['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

        # Return volatility
        features['ReturnVolatility'] = features['LogReturn'].rolling(
            window=self.config.window_size, min_periods=1
        ).std()

        return features


class FeaturePipeline:
    """Manages feature engineering pipeline."""

    def __init__(self, engineer: Optional[FeatureEngineer] = None, max_workers: int = 4):
        self.engineer = engineer or FeatureEngineer()
        self.max_workers = max_workers

    def process_stocks(self, stock_files: List[Path], output_dir: Path) -> pd.DataFrame:
        """Process multiple stocks in parallel."""
        output_dir.mkdir(parents=True, exist_ok=True)

        features_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_stock, file): file
                for file in stock_files
            }

            for future in futures:
                try:
                    stock_features = future.result()
                    if stock_features is not None:
                        features_list.append(stock_features)
                except Exception as e:
                    logger.error(f"Failed to process stock: {str(e)}")
                    continue

        if not features_list:
            raise ValueError("No stocks were processed successfully")

        # Combine all features
        combined = pd.concat(features_list)
        combined = combined.reset_index()

        # Save features
        self._save_features(combined, output_dir)

        return combined

    def _process_stock(self, file_path: Path) -> pd.DataFrame:
        """Process single stock file."""
        try:
            df = pd.read_csv(file_path)
            features = self.engineer.calculate_features(df)
            features['Ticker'] = file_path.stem
            return features
        except Exception as e:
            logger.error(f"Error processing {file_path.stem}: {str(e)}")
            return None

    @staticmethod
    def _save_features(features: pd.DataFrame, output_dir: Path) -> None:
        """Save processed features."""
        output_path = output_dir / 'features.parquet'
        features.to_parquet(output_path)