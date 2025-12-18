# data_collector_v5.py

import os
import ssl
import certifi
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any
from src.data.features_v5 import FeatureEngineer
import urllib.request
import talib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config['paths']['data'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.ssl_context = ssl._create_unverified_context()

        # Initialize the Engineer
        self.engineer = FeatureEngineer()

    def get_nasdaq100_tickers(self) -> List[str]:
        """Fetch NASDAQ-100 tickers from Wikipedia."""
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                html = response.read()
            dfs = pd.read_html(html)
            for df in dfs:
                if 'Ticker' in df.columns:
                    return df['Ticker'].tolist()
                elif 'Symbol' in df.columns:
                    return df['Symbol'].tolist()
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']  # Fallback
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    def collect_data(self, start_date: str = '2018-01-01'):
        """Main pipeline: Download -> FeatureEng -> Split -> Scale -> Save"""
        tickers = self.get_nasdaq100_tickers()
        logger.info(f"Collecting data for {len(tickers)} tickers...")

        all_dfs = []
        for ticker in tickers:
            try:
                # 1. Download
                df = yf.download(ticker, start=start_date, progress=False)
                if len(df) < 250: continue

                # Flatten MultiIndex columns if necessary (yfinance update)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df['Ticker'] = ticker
                df = df.reset_index()

                # 2. Generate Features (Raw)
                df = self.engineer.generate_features(df)

                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df['Date'] = pd.to_datetime(full_df['Date'])

        # 3. Time-Aware Splitting
        train_end = self.config['data']['train_end']
        val_end = self.config['data']['val_end']

        logger.info(f"Splitting Data: Train < {train_end} | Val < {val_end}")

        train_df = full_df[full_df['Date'] < train_end].copy()
        val_df = full_df[(full_df['Date'] >= train_end) & (full_df['Date'] < val_end)].copy()
        test_df = full_df[full_df['Date'] >= val_end].copy()

        # 4. Fit Scaler (TRAIN ONLY)
        logger.info("Fitting Scaler on Training set...")
        self.engineer.fit_scaler(train_df)

        # 5. Transform All Splits
        logger.info("Scaling datasets...")
        train_df = self.engineer.transform(train_df)
        val_df = self.engineer.transform(val_df)
        test_df = self.engineer.transform(test_df)

        # 6. Save
        train_df.to_parquet(self.data_path / 'train.parquet')
        val_df.to_parquet(self.data_path / 'validation.parquet')
        test_df.to_parquet(self.data_path / 'test.parquet')

        logger.info(f"Data Collection Complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    @staticmethod
    def download_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download stock data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return None

            df = df.reset_index()
            df.columns = df.columns.str.title()
            return df

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return None

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            # Basic indicators
            df['RSI'] = talib.RSI(df['Close'])
            df['MACD'], _, _ = talib.MACD(df['Close'])
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])

            # Bollinger Bands
            bb_upper, _, bb_lower = talib.BBANDS(df['Close'])
            df['BB_UPPER'] = bb_upper
            df['BB_LOWER'] = bb_lower

            # --- THE FIX: TARGET GENERATION ---
            # 1. Calculate Log Returns (Next Day)
            # Formula: ln(P_t / P_{t-1})
            # We shift(-1) because we want to predict tomorrow's return using today's data
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

            # The 'Target' is the Next Day's Return
            df['Target'] = df['Log_Return'].shift(-1)

            # Drop NaNs created by shifting and indicators
            return df.dropna()

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            raise

    def collect_data(self, start_date: str = '2018-01-01',
                     end_date: Optional[str] = None) -> None:
        """Collect and process stock data for NASDAQ-100."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Get tickers
        tickers = self.get_nasdaq100_tickers()

        # Process each stock
        all_data = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for ticker in tickers:
                logger.info(f"Processing {ticker}")

                # Download data
                df = self.download_stock_data(ticker, start_date, end_date)
                if df is None:
                    continue

                # Calculate features
                df = self.calculate_features(df)
                df['Ticker'] = ticker
                all_data.append(df)

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Split data
        train_cutoff = int(len(combined_data) * 0.7)
        val_cutoff = int(len(combined_data) * 0.85)

        train_data = combined_data[:train_cutoff]
        val_data = combined_data[train_cutoff:val_cutoff]
        test_data = combined_data[val_cutoff:]

        # Save splits
        for name, data in [('train', train_data),
                           ('validation', val_data),
                           ('test', test_data)]:
            path = self.data_dir / f'{name}.parquet'
            data.to_parquet(path, index=False)
            logger.info(f"Saved {name} data with {len(data)} samples to {path}")


def main():
    """Run data collection process."""
    import yaml

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    config_path = project_root / 'config_v5.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update paths in config to be absolute so other classes don't get lost
    # (Optional but recommended safety step)
    if 'paths' in config:
        for key in config['paths']:
            # Make data/checkpoints paths absolute relative to root
            config['paths'][key] = str(project_root / config['paths'][key])

    collector = DataCollector(config)
    collector.collect_data()


if __name__ == "__main__":
    main()