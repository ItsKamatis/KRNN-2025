from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.wrds_client import WRDSClient


logger = logging.getLogger(__name__)


class WRDSEquitySource:
    """Fetch and normalize daily equity OHLCV data from WRDS/CRSP."""

    DAILY_TABLE_CANDIDATES = ("dsf_v2", "dsf", "stkdlysecuritydata")
    NAMES_TABLE_CANDIDATES = ("stksecurityinfohist", "stocknames", "dsenames")

    DAILY_COLUMN_CANDIDATES = {
        "permno": ("permno", "lpermno"),
        "date": ("dlycaldt", "date", "caldt", "datadate"),
        "open": ("dlyopen", "openprc", "open"),
        "high": ("dlyhigh", "askhi", "high"),
        "low": ("dlylow", "bidlo", "low"),
        "close": ("dlyclose", "dlyprc", "prc", "close"),
        "volume": ("dlyvol", "vol", "volume"),
        "ret": ("dlyret", "ret"),
        "retx": ("dlyretx", "retx"),
        "shares_out": ("dlyshrout", "shrout", "shares_out"),
        "cum_fac_pr": ("dlycumfacpr", "cfacpr", "cumfacpr"),
        "cum_fac_shr": ("dlycumfacshr", "cfacshr", "cumfacshr"),
        "event_fac_pr": ("dlyfacprc", "facpr", "split_factor"),
    }

    NAMES_COLUMN_CANDIDATES = {
        "permno": ("permno", "lpermno"),
        "ticker": ("ticker", "tsymbol", "ticker_symbol"),
        "start_date": (
            "namedt",
            "secinfostartdt",
            "securitybegdt",
            "begindate",
            "effdt",
            "startdt",
            "from_date",
        ),
        "end_date": (
            "nameenddt",
            "nameendt",
            "secinfoenddt",
            "securityenddt",
            "enddate",
            "enddt",
            "thrudt",
            "to_date",
        ),
    }

    def __init__(self, client: WRDSClient, config: dict[str, Any]) -> None:
        self.client = client
        self.config = config
        self.wrds_cfg = config.get("wrds", {})
        self.schema = str(self.wrds_cfg.get("schema", "crsp"))

        raw_cache_dir = self.wrds_cfg.get("raw_cache_dir")
        self.raw_cache_dir = Path(raw_cache_dir) if raw_cache_dir else Path(config["paths"]["data"]) / "raw"

        self.daily_table_candidates = tuple(
            self.wrds_cfg.get("daily_table_candidates", self.DAILY_TABLE_CANDIDATES)
        )
        self.names_table_candidates = tuple(
            self.wrds_cfg.get("names_table_candidates", self.NAMES_TABLE_CANDIDATES)
        )

    def fetch_daily_ohlcv(
        self,
        tickers: Sequence[str],
        *,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        clean_tickers = sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
        if not clean_tickers:
            raise ValueError("WRDS equity source requires at least one ticker.")

        names_df, names_table = self._fetch_name_history(clean_tickers, start_date=start_date, end_date=end_date)
        if names_df.empty:
            raise RuntimeError("No WRDS name history rows matched the requested ticker universe.")

        permnos = sorted(names_df["Permno"].dropna().astype(int).unique().tolist())
        daily_df, daily_table = self._fetch_daily_rows(permnos, start_date=start_date, end_date=end_date)
        if daily_df.empty:
            raise RuntimeError("No WRDS daily price rows matched the requested identifier/date range.")

        merged = self._attach_tickers(daily_df, names_df, requested_tickers=clean_tickers)
        if merged.empty:
            raise RuntimeError("WRDS daily rows were fetched, but none could be mapped back to the ticker universe.")

        self._write_raw_cache(
            names_df=names_df,
            daily_df=daily_df,
            merged_df=merged,
            names_table=names_table,
            daily_table=daily_table,
            start_date=start_date,
            end_date=end_date,
        )

        return merged[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()

    def _fetch_name_history(
        self,
        tickers: Sequence[str],
        *,
        start_date: str,
        end_date: str | None,
    ) -> tuple[pd.DataFrame, str]:
        table = self._resolve_table(self.names_table_candidates, "name-history")
        columns = self.client.table_cols(self.schema, table)

        permno_col = self.client.pick_col(columns, self.NAMES_COLUMN_CANDIDATES["permno"])
        ticker_col = self.client.pick_col(columns, self.NAMES_COLUMN_CANDIDATES["ticker"])
        start_col = self._pick_optional_col(columns, self.NAMES_COLUMN_CANDIDATES["start_date"])
        end_col = self._pick_optional_col(columns, self.NAMES_COLUMN_CANDIDATES["end_date"])

        select_parts = [
            f"{permno_col} as permno",
            f"upper(trim({ticker_col})) as ticker",
            f"{start_col} as start_date" if start_col else "NULL::date as start_date",
            f"{end_col} as end_date" if end_col else "NULL::date as end_date",
        ]

        where_parts = [f"upper(trim({ticker_col})) in {self.client.sql_string_list(tickers)}"]
        if start_col and end_date:
            where_parts.append(f"{start_col} <= DATE {self.client.quote_sql_string(end_date)}")
        if end_col:
            where_parts.append(
                f"coalesce({end_col}, DATE '2099-12-31') >= DATE {self.client.quote_sql_string(start_date)}"
            )

        sql = f"""
            select {", ".join(select_parts)}
            from {self.schema}.{table}
            where {" and ".join(where_parts)}
        """
        df = self.client.read_sql(sql)
        if df.empty:
            return df, table

        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df["start_date"] = df["start_date"].fillna(pd.Timestamp("1900-01-01"))
        df["end_date"] = df["end_date"].fillna(pd.Timestamp("2099-12-31"))

        df = df.dropna(subset=["permno", "ticker"]).copy()
        df = df.rename(
            columns={
                "permno": "Permno",
                "ticker": "Ticker",
                "start_date": "Name_Start",
                "end_date": "Name_End",
            }
        )
        df = df.sort_values(["Ticker", "Permno", "Name_Start", "Name_End"]).drop_duplicates()
        return df.reset_index(drop=True), table

    def _fetch_daily_rows(
        self,
        permnos: Sequence[int],
        *,
        start_date: str,
        end_date: str | None,
    ) -> tuple[pd.DataFrame, str]:
        table = self._resolve_table(self.daily_table_candidates, "daily price")
        columns = self.client.table_cols(self.schema, table)

        permno_col = self.client.pick_col(columns, self.DAILY_COLUMN_CANDIDATES["permno"])
        date_col = self.client.pick_col(columns, self.DAILY_COLUMN_CANDIDATES["date"])
        close_col = self.client.pick_col(columns, self.DAILY_COLUMN_CANDIDATES["close"])
        volume_col = self.client.pick_col(columns, self.DAILY_COLUMN_CANDIDATES["volume"])
        open_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES["open"])
        high_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES["high"])
        low_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES["low"])
        cum_fac_pr_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES["cum_fac_pr"])
        cum_fac_shr_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES["cum_fac_shr"])
        event_fac_pr_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES["event_fac_pr"])

        select_parts = [
            f"{permno_col} as permno",
            f"{date_col} as date",
            f"{close_col} as close",
            f"{volume_col} as volume",
        ]

        if open_col:
            select_parts.append(f"{open_col} as open")
        if high_col:
            select_parts.append(f"{high_col} as high")
        if low_col:
            select_parts.append(f"{low_col} as low")
        if cum_fac_pr_col:
            select_parts.append(f"{cum_fac_pr_col} as cum_fac_pr")
        if cum_fac_shr_col:
            select_parts.append(f"{cum_fac_shr_col} as cum_fac_shr")
        if event_fac_pr_col:
            select_parts.append(f"{event_fac_pr_col} as event_fac_pr")

        for label in ("ret", "retx", "shares_out"):
            optional_col = self._pick_optional_col(columns, self.DAILY_COLUMN_CANDIDATES[label])
            if optional_col:
                select_parts.append(f"{optional_col} as {label}")

        where_parts = [
            f"{permno_col} in {self.client.sql_int_list(permnos)}",
            f"{date_col} >= DATE {self.client.quote_sql_string(start_date)}",
        ]
        if end_date:
            where_parts.append(f"{date_col} <= DATE {self.client.quote_sql_string(end_date)}")

        sql = f"""
            select {", ".join(select_parts)}
            from {self.schema}.{table}
            where {" and ".join(where_parts)}
        """
        df = self.client.read_sql(sql)
        if df.empty:
            return df, table

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")

        for price_col in ("open", "high", "low", "close"):
            if price_col in df.columns:
                df[price_col] = pd.to_numeric(df[price_col], errors="coerce").abs()

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["open"] = df["open"] if "open" in df.columns else df["close"]
        df["high"] = df["high"] if "high" in df.columns else df["close"]
        df["low"] = df["low"] if "low" in df.columns else df["close"]
        if "cum_fac_pr" in df.columns:
            df["cum_fac_pr"] = pd.to_numeric(df["cum_fac_pr"], errors="coerce")
        if "cum_fac_shr" in df.columns:
            df["cum_fac_shr"] = pd.to_numeric(df["cum_fac_shr"], errors="coerce")
        if "event_fac_pr" in df.columns:
            df["event_fac_pr"] = pd.to_numeric(df["event_fac_pr"], errors="coerce")

        df = df.dropna(subset=["permno", "date", "close", "volume"]).copy()
        df["volume"] = df["volume"].clip(lower=0)
        df = self._apply_split_adjustments(df)

        daily = pd.DataFrame(
            {
                "Permno": df["permno"].astype(int),
                "Date": df["date"],
                "Open": df["open"],
                "High": df["high"],
                "Low": df["low"],
                "Close": df["close"],
                "Volume": df["volume"],
            }
        )
        daily = daily.sort_values(["Permno", "Date"]).drop_duplicates(subset=["Permno", "Date"], keep="last")
        return daily.reset_index(drop=True), table

    def _apply_split_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["permno", "date"]).copy()

        if "cum_fac_pr" in df.columns and df["cum_fac_pr"].notna().any():
            price_factor = df["cum_fac_pr"].copy()
        elif "event_fac_pr" in df.columns and df["event_fac_pr"].notna().any():
            price_factor = df.groupby("permno", group_keys=False)["event_fac_pr"].apply(
                self._derive_cumulative_factor_from_event
            )
        else:
            price_factor = pd.Series(1.0, index=df.index, dtype="float64")

        if "cum_fac_shr" in df.columns and df["cum_fac_shr"].notna().any():
            share_factor = df["cum_fac_shr"].copy()
        else:
            share_factor = price_factor.copy()

        price_factor = pd.to_numeric(price_factor, errors="coerce").fillna(1.0).clip(lower=1e-12)
        share_factor = pd.to_numeric(share_factor, errors="coerce").fillna(1.0).clip(lower=1e-12)

        for col in ("open", "high", "low", "close"):
            df[col] = df[col] / price_factor

        df["volume"] = df["volume"] * share_factor
        return df

    @staticmethod
    def _derive_cumulative_factor_from_event(event_factor: pd.Series) -> pd.Series:
        shifted = pd.to_numeric(event_factor, errors="coerce").fillna(1.0).shift(-1).fillna(1.0)
        reversed_cumprod = shifted.iloc[::-1].cumprod().iloc[::-1]
        return reversed_cumprod.astype("float64")

    def _attach_tickers(
        self,
        daily_df: pd.DataFrame,
        names_df: pd.DataFrame,
        *,
        requested_tickers: Sequence[str],
    ) -> pd.DataFrame:
        merged = daily_df.merge(names_df, how="left", on="Permno")
        active_mask = (merged["Date"] >= merged["Name_Start"]) & (merged["Date"] <= merged["Name_End"])
        merged = merged[active_mask].copy()
        if merged.empty:
            return merged

        merged = merged[merged["Ticker"].isin(requested_tickers)].copy()
        merged = merged.sort_values(["Ticker", "Permno", "Date", "Name_Start"])
        merged = merged.drop_duplicates(subset=["Ticker", "Date"], keep="last")
        merged = merged[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Permno"]]
        merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        return merged

    def _resolve_table(self, candidates: Sequence[str], label: str) -> str:
        table = self.client.pick_table(self.schema, candidates)
        if table is None:
            raise RuntimeError(
                f"No WRDS {label} table visible under schema '{self.schema}'. "
                f"Tried: {list(candidates)}"
            )
        return table

    @staticmethod
    def _pick_optional_col(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
        lower_map = {str(col).lower(): str(col) for col in columns}
        for candidate in candidates:
            match = lower_map.get(candidate.lower())
            if match is not None:
                return match
        return None

    def _write_raw_cache(
        self,
        *,
        names_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        names_table: str,
        daily_table: str,
        start_date: str,
        end_date: str | None,
    ) -> None:
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

        names_path = self.raw_cache_dir / "wrds_symbol_map.parquet"
        daily_path = self.raw_cache_dir / "wrds_equity_daily.parquet"
        merged_path = self.raw_cache_dir / "wrds_equity_ohlcv.parquet"
        manifest_path = self.raw_cache_dir / "wrds_extract_manifest.json"

        names_df.to_parquet(names_path, index=False)
        daily_df.to_parquet(daily_path, index=False)
        merged_df.to_parquet(merged_path, index=False)

        manifest = {
            "source": "wrds_crsp",
            "schema": self.schema,
            "names_table": names_table,
            "daily_table": daily_table,
            "start_date": start_date,
            "end_date": end_date,
            "symbol_rows": int(len(names_df)),
            "daily_rows": int(len(daily_df)),
            "ohlcv_rows": int(len(merged_df)),
            "tickers": sorted(merged_df["Ticker"].astype(str).unique().tolist()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
