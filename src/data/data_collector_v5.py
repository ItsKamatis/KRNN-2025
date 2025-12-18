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
from typing import List, Dict, Optional
import urllib.request
import talib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and processes stock data for KRNN model."""

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['paths']['data'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._setup_ssl()

    def _setup_ssl(self):
        """Configure SSL context for web requests."""
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def get_nasdaq100_tickers(self) -> List[str]:
        """Fetch current NASDAQ-100 constituents."""
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'

        try:
            # --- THE FIX: Add a User-Agent Header ---
            # Wikipedia blocks default python-urllib requests.
            # We pretend to be a standard browser (Chrome on Windows).
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Create a Request object with headers
            req = urllib.request.Request(url, headers=headers)

            # Pass the Request object instead of just the URL
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                html = response.read()

            dfs = pd.read_html(html)

            # Find table with ticker symbols
            for df in dfs:
                if 'Ticker' in df.columns:
                    tickers = df['Ticker'].tolist()
                    logger.info(f"Found {len(tickers)} NASDAQ-100 tickers")
                    return tickers
                elif 'Symbol' in df.columns:
                    tickers = df['Symbol'].tolist()
                    logger.info(f"Found {len(tickers)} NASDAQ-100 tickers")
                    return tickers

            raise ValueError("Could not find ticker symbols in Wikipedia page")

        except Exception as e:
            logger.error(f"Error fetching NASDAQ-100 tickers: {e}")
            # Fallback tickers in case Wikipedia completely fails (prevents total crash)
            logger.warning("Using fallback ticker list due to Wikipedia error.")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'PEP', 'AVGO', 'COST']

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