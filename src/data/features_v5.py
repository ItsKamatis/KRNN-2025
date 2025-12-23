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
        Calculates technical indicators and transformations.

        Expected input columns: Open, High, Low, Close, Volume.
        Output includes:
          - engineered features (self.feature_cols)
          - Log_Return
          - Target (next-step Log_Return)
        """
        df = df.copy()

        eps = 1e-12

        # Ensure numeric columns
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        vol = df['Volume'].astype(float)

        # 1) RSI (Momentum)
        df['RSI'] = talib.RSI(close, timeperiod=14)

        # 2) MACD (Trend), normalized by price to be scale-invariant
        macd, signal, _ = talib.MACD(close)
        denom_close = close.where(close.abs() > eps, np.nan)
        df['MACD_Rel'] = macd / denom_close
        df['MACD_Sig_Rel'] = signal / denom_close

        # 3) Bollinger Bands (Volatility/Mean-reversion)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        bb_den = (upper - lower)
        bb_den = np.where(np.abs(bb_den) > eps, bb_den, np.nan)
        df['BB_Pos'] = (close - lower) / bb_den

        mid_den = np.where(np.abs(middle) > eps, middle, np.nan)
        df['BB_Width'] = (upper - lower) / mid_den

        # 4) ATR (Volatility), normalized by price
        abs_atr = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_Rel'] = abs_atr / denom_close

        # 5) Volume log transform
        df['Log_Volume'] = np.log1p(vol)

        # 6) Log returns (primary signal)
        prev_close = close.shift(1)
        valid_lr = (close.abs() > eps) & (prev_close.abs() > eps)
        df['Log_Return'] = np.where(valid_lr, np.log(close / prev_close), np.nan)

        # 7) Target = next-step return
        df['Target'] = pd.Series(df['Log_Return']).shift(-1)

        # Clean: drop NaN and +/-inf produced by indicators/divisions
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Feature set (exclude raw prices to avoid learning price levels)
        self.feature_cols = [
            'Log_Return', 'RSI', 'MACD_Rel', 'MACD_Sig_Rel',
            'BB_Width', 'BB_Pos', 'ATR_Rel', 'Log_Volume'
        ]

        # Final sanity: enforce finiteness for feature columns + target
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=self.feature_cols + ['Target'], inplace=True)

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
    def _calculate_returns(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """Calculate return-based features (currently unused)."""
        features = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df.get('Close'), errors='coerce')
        features['LogReturn'] = np.log(close / close.shift(1))
        features['ReturnVolatility'] = features['LogReturn'].rolling(window=window_size, min_periods=1).std()
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