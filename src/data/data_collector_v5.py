import json
import os
import ssl
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from typing import List, Any, Dict
import urllib.request
from datetime import datetime, timezone
from src.data.features_v5 import FeatureEngineer
from src.data.wrds_client import WRDSClient
from src.data.wrds_equity_source import WRDSEquitySource

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DataCollector:
    DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config['paths']['data'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.ssl_context = ssl._create_unverified_context()
        self.engineer = FeatureEngineer()
        self.data_cfg = config.get('data', {})
        self.wrds_cfg = config.get('wrds', {})

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
            return self.DEFAULT_TICKERS
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return self.DEFAULT_TICKERS

    def resolve_tickers(self) -> List[str]:
        universe_mode = str(
            self.wrds_cfg.get('universe_mode') or self.data_cfg.get('universe_mode', 'nasdaq100_current_members')
        ).lower()

        if universe_mode == 'static_ticker_list':
            raw_tickers = self.wrds_cfg.get('tickers') or self.data_cfg.get('tickers') or []
        elif universe_mode == 'nasdaq100_current_members':
            raw_tickers = self.get_nasdaq100_tickers()
        else:
            raise ValueError(f"Unsupported universe_mode='{universe_mode}'")

        tickers = sorted({str(ticker).strip().upper() for ticker in raw_tickers if str(ticker).strip()})
        if not tickers:
            raise RuntimeError("Ticker universe resolved to an empty set.")

        return tickers

    def fetch_raw_market_data(self, tickers: List[str]) -> pd.DataFrame:
        source = str(self.data_cfg.get('source', 'yfinance')).lower()
        start_date = self.data_cfg['train_start']
        end_date = self.data_cfg.get('end_date')

        if source == 'yfinance':
            return self._fetch_yfinance_data(tickers, start_date=start_date, end_date=end_date)
        if source == 'wrds':
            return self._fetch_wrds_data(tickers, start_date=start_date, end_date=end_date)

        raise ValueError(f"Unsupported data.source='{source}'")

    def _fetch_yfinance_data(
        self,
        tickers: List[str],
        *,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        all_dfs = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(df) < 250:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing = [col for col in keep_cols if col not in df.columns]
                if missing:
                    logger.warning(f"Skipping {ticker}: missing yfinance columns {missing}")
                    continue

                df = df[keep_cols].copy()
                df['Ticker'] = ticker
                df = df.reset_index()
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")

        if not all_dfs:
            raise RuntimeError("No yfinance data collected.")

        return pd.concat(all_dfs, ignore_index=True)

    def _fetch_wrds_data(
        self,
        tickers: List[str],
        *,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        username = self.wrds_cfg.get('username')
        username_env = self.wrds_cfg.get('username_env')
        if username_env and not username:
            username = os.getenv(str(username_env))

        password = self.wrds_cfg.get('password')
        password_env = self.wrds_cfg.get('password_env')
        if password_env and not password:
            password = os.getenv(str(password_env))

        client = WRDSClient(
            wrds_username=username,
            wrds_password=password,
            autoconnect=True,
            verbose=bool(self.wrds_cfg.get('verbose', False)),
        )
        try:
            source = WRDSEquitySource(client, self.config)
            return source.fetch_daily_ohlcv(tickers, start_date=start_date, end_date=end_date)
        finally:
            client.close()

    def _engineer_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'}
        missing = required_cols.difference(raw_df.columns)
        if missing:
            raise RuntimeError(f"Raw market data is missing required columns: {sorted(missing)}")

        raw_df = raw_df.copy()
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        raw_df = raw_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        all_dfs = []
        for ticker, ticker_df in raw_df.groupby('Ticker', sort=False):
            if len(ticker_df) < 250:
                logger.info(f"Skipping {ticker}: only {len(ticker_df)} raw rows available.")
                continue

            try:
                featured_df = self.engineer.generate_features(ticker_df)
                if not featured_df.empty:
                    all_dfs.append(featured_df)
            except Exception as e:
                logger.warning(f"Feature generation failed for {ticker}: {e}")

        if not all_dfs:
            raise RuntimeError("No feature-engineered ticker frames were produced.")

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        return full_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    def _split_scale_and_save(self, full_df: pd.DataFrame) -> Dict[str, int]:
        train_end = self.data_cfg['train_end']
        val_end = self.data_cfg['val_end']

        train_df = full_df[full_df['Date'] < train_end].copy()
        val_df = full_df[(full_df['Date'] >= train_end) & (full_df['Date'] < val_end)].copy()
        test_df = full_df[full_df['Date'] >= val_end].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise RuntimeError(
                "At least one split is empty after feature engineering. "
                "Adjust the date range, sequence horizon, or data source coverage."
            )

        logger.info("Fitting Scaler on Training set...")
        self.engineer.fit_scaler(train_df)

        logger.info("Scaling datasets...")
        train_df = self.engineer.transform(train_df)
        val_df = self.engineer.transform(val_df)
        test_df = self.engineer.transform(test_df)

        train_df.to_parquet(self.data_path / 'train.parquet')
        val_df.to_parquet(self.data_path / 'validation.parquet')
        test_df.to_parquet(self.data_path / 'test.parquet')

        logger.info(
            "Data Collection Complete. Train: %s, Val: %s, Test: %s",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return {
            'train_rows': int(len(train_df)),
            'validation_rows': int(len(val_df)),
            'test_rows': int(len(test_df)),
        }

    def _write_dataset_manifest(self, tickers: List[str], split_counts: Dict[str, int]) -> None:
        manifest = {
            'source': str(self.data_cfg.get('source', 'yfinance')).lower(),
            'dataset_cache_version': str(self.data_cfg.get('dataset_cache_version', 'v1')),
            'universe_mode': str(
                self.wrds_cfg.get('universe_mode') or self.data_cfg.get('universe_mode', 'nasdaq100_current_members')
            ).lower(),
            'tickers': tickers,
            'split_counts': split_counts,
            'train_start': self.data_cfg.get('train_start'),
            'train_end': self.data_cfg.get('train_end'),
            'val_end': self.data_cfg.get('val_end'),
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = self.data_path / 'dataset_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def collect_data(self):
        """Main pipeline: raw fetch -> feature engineering -> split -> scale -> save."""
        tickers = self.resolve_tickers()
        logger.info(
            "Collecting data for %s tickers from source='%s'...",
            len(tickers),
            self.data_cfg.get('source', 'yfinance'),
        )

        raw_df = self.fetch_raw_market_data(tickers)
        full_df = self._engineer_features(raw_df)
        split_counts = self._split_scale_and_save(full_df)
        self._write_dataset_manifest(tickers, split_counts)


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
    if 'wrds' in config and config['wrds'].get('raw_cache_dir'):
        config['wrds']['raw_cache_dir'] = str(project_root / config['wrds']['raw_cache_dir'])

    collector = DataCollector(config)
    collector.collect_data()


if __name__ == "__main__":
    main()
