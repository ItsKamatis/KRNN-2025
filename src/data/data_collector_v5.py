import os
import ssl
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from typing import List, Any, Dict
import urllib.request
from src.data.features_v5 import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config['paths']['data'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.ssl_context = ssl._create_unverified_context()
        self.engineer = FeatureEngineer()

    def get_nasdaq100_tickers(self) -> List[str]:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                html = response.read()
            dfs = pd.read_html(html)
            for df in dfs:
                if 'Ticker' in df.columns:
                    return df['Ticker'].tolist()
                elif 'Symbol' in df.columns:
                    return df['Symbol'].tolist()
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    def collect_data(self):
        """Main pipeline: Download -> FeatureEng -> Split -> Scale -> Save"""
        tickers = self.get_nasdaq100_tickers()
        logger.info(f"Collecting data for {len(tickers)} tickers...")

        all_dfs = []
        for ticker in tickers:
            try:
                # 1. Download
                df = yf.download(ticker, start=self.config['data']['train_start'], progress=False)
                if len(df) < 250: continue

                # Flatten MultiIndex if exists
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df['Ticker'] = ticker
                df = df.reset_index()

                # 2. Generate Features (Use the Robust FeatureEngineer)
                # This selects the specific 8 features defined in features_v5.py
                df = self.engineer.generate_features(df)

                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")

        if not all_dfs:
            raise RuntimeError("No data collected!")

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df['Date'] = pd.to_datetime(full_df['Date'])

        # 3. Time-Aware Splitting
        train_end = self.config['data']['train_end']
        val_end = self.config['data']['val_end']

        train_df = full_df[full_df['Date'] < train_end].copy()
        val_df = full_df[(full_df['Date'] >= train_end) & (full_df['Date'] < val_end)].copy()
        test_df = full_df[full_df['Date'] >= val_end].copy()

        # 4. Fit Scaler (TRAIN ONLY) to prevent look-ahead bias
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


def main():
    import yaml
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    config_path = project_root / 'config_v5.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Fix paths
    for key in config['paths']:
        config['paths'][key] = str(project_root / config['paths'][key])

    collector = DataCollector(config)
    collector.collect_data()


if __name__ == "__main__":
    main()