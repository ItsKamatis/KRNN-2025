import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.wrds_client import WRDSClient
from src.data.wrds_equity_source import WRDSEquitySource


class FakeWRDSClient:
    def __init__(self, tables):
        self.tables = tables

    def table_exists(self, schema, table):
        return (schema, table) in self.tables

    def table_cols(self, schema, table):
        return self.tables[(schema, table)]["columns"]

    def pick_table(self, schema, candidates):
        for table in candidates:
            if self.table_exists(schema, table):
                return table
        return None

    def pick_col(self, columns, candidates):
        return WRDSClient.pick_col(columns, candidates)

    def quote_sql_string(self, value):
        return WRDSClient.quote_sql_string(value)

    def sql_string_list(self, values):
        return WRDSClient.sql_string_list(values)

    def sql_int_list(self, values):
        return WRDSClient.sql_int_list(values)

    def read_sql(self, sql, params=None, **kwargs):
        sql_lower = sql.lower()
        for (schema, table), payload in self.tables.items():
            if f"from {schema}.{table}".lower() in sql_lower:
                return payload["data"].copy()
        raise AssertionError(f"Unexpected SQL: {sql}")


def test_wrds_equity_source_normalizes_legacy_crsp_layout(tmp_path):
    tables = {
        ("crsp", "stocknames"): {
            "columns": ["permno", "ticker", "namedt", "nameenddt"],
            "data": pd.DataFrame(
                {
                    "permno": [10101],
                    "ticker": ["AAPL"],
                    "start_date": ["2010-01-01"],
                    "end_date": ["2099-12-31"],
                }
            ),
        },
        ("crsp", "dsf"): {
            "columns": ["permno", "date", "askhi", "bidlo", "prc", "vol", "ret", "retx", "cfacpr", "cfacshr"],
            "data": pd.DataFrame(
                {
                    "permno": [10101, 10101],
                    "date": ["2024-01-02", "2024-01-03"],
                    "high": [101.0, 102.0],
                    "low": [99.0, 98.5],
                    "close": [-100.5, -101.25],
                    "volume": [1_000_000, 1_100_000],
                    "ret": [0.01, 0.02],
                    "retx": [0.01, 0.02],
                    "cum_fac_pr": [2.0, 1.0],
                    "cum_fac_shr": [2.0, 1.0],
                }
            ),
        },
    }
    config = {
        "paths": {"data": str(tmp_path / "data")},
        "wrds": {"raw_cache_dir": str(tmp_path / "raw")},
    }

    source = WRDSEquitySource(FakeWRDSClient(tables), config)
    df = source.fetch_daily_ohlcv(["AAPL"], start_date="2024-01-01", end_date="2024-01-31")

    assert list(df.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    assert df["Ticker"].tolist() == ["AAPL", "AAPL"]
    assert df["Open"].tolist() == df["Close"].tolist()
    assert df["Close"].tolist() == [50.25, 101.25]
    assert df["Volume"].tolist() == [2_000_000, 1_100_000]
    assert (tmp_path / "raw" / "wrds_equity_daily.parquet").exists()

    manifest = json.loads((tmp_path / "raw" / "wrds_extract_manifest.json").read_text(encoding="utf-8"))
    assert manifest["daily_table"] == "dsf"
    assert manifest["names_table"] == "stocknames"


def test_wrds_equity_source_prefers_dsf_v2_with_cumulative_adjustment(tmp_path):
    tables = {
        ("crsp", "stocknames"): {
            "columns": ["permno", "ticker", "namedt", "nameenddt"],
            "data": pd.DataFrame(
                {
                    "permno": [20202],
                    "ticker": ["MSFT"],
                    "start_date": ["2010-01-01"],
                    "end_date": ["2099-12-31"],
                }
            ),
        },
        ("crsp", "dsf_v2"): {
            "columns": [
                "permno",
                "dlycaldt",
                "dlyopen",
                "dlyhigh",
                "dlylow",
                "dlyclose",
                "dlyvol",
                "dlycumfacpr",
                "dlycumfacshr",
            ],
            "data": pd.DataFrame(
                {
                    "permno": [20202, 20202],
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [400.0, 101.0],
                    "high": [404.0, 103.0],
                    "low": [396.0, 99.0],
                    "close": [402.0, 100.5],
                    "volume": [1_000_000, 1_100_000],
                    "cum_fac_pr": [4.0, 1.0],
                    "cum_fac_shr": [4.0, 1.0],
                }
            ),
        },
        ("crsp", "stkdlysecuritydata"): {
            "columns": ["permno", "dlycaldt", "dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlyvol"],
            "data": pd.DataFrame(
                {
                    "permno": [20202],
                    "date": ["2024-01-02"],
                    "open": [999.0],
                    "high": [999.0],
                    "low": [999.0],
                    "close": [999.0],
                    "volume": [1],
                }
            ),
        },
    }
    config = {
        "paths": {"data": str(tmp_path / "data")},
        "wrds": {"raw_cache_dir": str(tmp_path / "raw")},
    }

    source = WRDSEquitySource(FakeWRDSClient(tables), config)
    df = source.fetch_daily_ohlcv(["MSFT"], start_date="2024-01-01", end_date="2024-01-31")

    assert df.iloc[0]["Open"] == 100.0
    assert df.iloc[0]["High"] == 101.0
    assert df.iloc[0]["Low"] == 99.0
    assert df.iloc[0]["Close"] == 100.5
    assert df.iloc[0]["Volume"] == 4_000_000

    manifest = json.loads((tmp_path / "raw" / "wrds_extract_manifest.json").read_text(encoding="utf-8"))
    assert manifest["daily_table"] == "dsf_v2"


def test_wrds_equity_source_derives_cumulative_factor_from_event_factor(tmp_path):
    tables = {
        ("crsp", "stocknames"): {
            "columns": ["permno", "ticker", "namedt", "nameenddt"],
            "data": pd.DataFrame(
                {
                    "permno": [30303],
                    "ticker": ["ORLY"],
                    "start_date": ["2010-01-01"],
                    "end_date": ["2099-12-31"],
                }
            ),
        },
        ("crsp", "stkdlysecuritydata"): {
            "columns": [
                "permno",
                "dlycaldt",
                "dlyopen",
                "dlyhigh",
                "dlylow",
                "dlyclose",
                "dlyvol",
                "dlyfacprc",
            ],
            "data": pd.DataFrame(
                {
                    "permno": [30303, 30303, 30303],
                    "date": ["2025-06-09", "2025-06-10", "2025-06-11"],
                    "open": [1375.0, 90.0, 91.71],
                    "high": [1376.215, 92.12, 91.8716],
                    "low": [1332.3, 89.61, 89.53],
                    "close": [1348.1, 91.71, 90.01],
                    "volume": [368_578, 5_166_035, 5_754_549],
                    "event_fac_pr": [1.0, 15.0, 1.0],
                }
            ),
        },
    }
    config = {
        "paths": {"data": str(tmp_path / "data")},
        "wrds": {
            "raw_cache_dir": str(tmp_path / "raw"),
            "daily_table_candidates": ["stkdlysecuritydata"],
        },
    }

    source = WRDSEquitySource(FakeWRDSClient(tables), config)
    df = source.fetch_daily_ohlcv(["ORLY"], start_date="2025-06-09", end_date="2025-06-11")

    assert df["Close"].round(3).tolist() == [89.873, 91.71, 90.01]
    assert df["Open"].round(3).tolist() == [91.667, 90.0, 91.71]
    assert df["Volume"].tolist() == [5_528_670, 5_166_035, 5_754_549]
