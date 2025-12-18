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


import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Centralizes feature generation and scaling.
    Ensures 'Fit on Train, Transform on Test' to prevent leakage.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Columns that will be fed into the model
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Technical Indicators and applies transformations.
        Expected input: DataFrame with Open, High, Low, Close, Volume.
        """
        # Copy to avoid warnings
        df = df.copy()

        # 1. Basic Indicators (TA-Lib)
        # RSI (Momentum)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

        # MACD (Trend) - Normalized by Close to be scale-invariant
        macd, signal, _ = talib.MACD(df['Close'])
        df['MACD_Rel'] = macd / df['Close']
        df['MACD_Sig_Rel'] = signal / df['Close']

        # Bollinger Bands (Volatility) - Relative Bandwidth
        upper, middle, lower = talib.BBANDS(df['Close'])
        df['BB_Width'] = (upper - lower) / middle
        df['BB_Pos'] = (df['Close'] - lower) / (upper - lower)  # Position within band (0-1)

        # ATR (Volatility) - Relative
        # Normalizing ATR by price allows comparing Volatility of $100 stock vs $10 stock
        abs_atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ATR_Rel'] = abs_atr / df['Close']

        # 2. Volume Log Transform
        # Volume follows a power law; log makes it more Gaussian-like
        df['Log_Volume'] = np.log1p(df['Volume'])

        # 3. Log Return (The Primary Signal)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # 4. Target Generation (Shifted Return)
        df['Target'] = df['Log_Return'].shift(-1)

        # Drop NaNs created by lags
        df.dropna(inplace=True)

        # Define the feature set
        # We explicitly exclude raw prices (Open/High/Low/Close) to ensure
        # the model learns patterns, not price levels.
        self.feature_cols = [
            'Log_Return', 'RSI', 'MACD_Rel', 'MACD_Sig_Rel',
            'BB_Width', 'BB_Pos', 'ATR_Rel', 'Log_Volume'
        ]

        return df

    def fit_scaler(self, train_df: pd.DataFrame):
        """Fits the Standard Scaler on Training Data ONLY."""
        if not self.feature_cols:
            raise ValueError("Run generate_features first to define columns.")

        logger.info("Fitting StandardScaler on Training Data...")
        self.scaler.fit(train_df[self.feature_cols])
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies scaling to data."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fit on training data first.")

        df = df.copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

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