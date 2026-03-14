from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Sequence

import pandas as pd

try:
    import wrds
except ImportError:  # pragma: no cover - covered by runtime dependency checks
    wrds = None


logger = logging.getLogger(__name__)


class WRDSClient:
    """Thin wrapper around the installed WRDS Python client."""

    def __init__(
        self,
        wrds_username: str | None = None,
        wrds_password: str | None = None,
        *,
        autoconnect: bool = True,
        verbose: bool = False,
    ) -> None:
        self.wrds_username = wrds_username or os.getenv("WRDS_USERNAME")
        self.wrds_password = wrds_password or os.getenv("WRDS_PASSWORD")
        self.verbose = verbose
        self._connection = None

        if autoconnect:
            self.connect()

    def connect(self):
        if self._connection is None:
            if wrds is None:
                raise ImportError(
                    "The 'wrds' package is required for WRDS-backed data collection. "
                    "Install it from requirements.txt before using data.source='wrds'."
                )

            kwargs = {"autoconnect": True, "verbose": self.verbose}
            if self.wrds_username:
                kwargs["wrds_username"] = self.wrds_username
            if self.wrds_password:
                kwargs["wrds_password"] = self.wrds_password

            logger.info("Opening WRDS connection...")
            self._connection = wrds.Connection(**kwargs)

        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def read_sql(self, sql: str, params: dict[str, Any] | None = None, **kwargs) -> pd.DataFrame:
        return self.connect().raw_sql(sql, params=params, **kwargs)

    def table_exists(self, schema: str, table: str) -> bool:
        sql = """
            select 1
            from information_schema.tables
            where table_schema = %(schema)s
              and table_name = %(table)s
            limit 1
        """
        df = self.read_sql(sql, params={"schema": schema, "table": table})
        return not df.empty

    def table_cols(self, schema: str, table: str) -> list[str]:
        sql = """
            select column_name
            from information_schema.columns
            where table_schema = %(schema)s
              and table_name = %(table)s
            order by ordinal_position
        """
        df = self.read_sql(sql, params={"schema": schema, "table": table})
        return df["column_name"].astype(str).tolist()

    def pick_table(self, schema: str, candidates: Sequence[str]) -> str | None:
        for table in candidates:
            if self.table_exists(schema, table):
                return table
        return None

    @staticmethod
    def pick_col(columns: Sequence[str], candidates: Sequence[str]) -> str:
        lower_map = {str(col).lower(): str(col) for col in columns}
        for candidate in candidates:
            match = lower_map.get(candidate.lower())
            if match is not None:
                return match

        raise KeyError(f"None of {list(candidates)} found in available columns: {list(columns)}")

    @staticmethod
    def quote_sql_string(value: Any) -> str:
        text = str(value).replace("'", "''")
        return f"'{text}'"

    @classmethod
    def sql_string_list(cls, values: Iterable[Any]) -> str:
        cleaned = [cls.quote_sql_string(v) for v in values if str(v).strip()]
        if not cleaned:
            raise ValueError("Cannot build SQL list from an empty string sequence.")
        return "(" + ", ".join(cleaned) + ")"

    @staticmethod
    def sql_int_list(values: Iterable[Any]) -> str:
        cleaned: list[str] = []
        for value in values:
            if pd.isna(value):
                continue
            cleaned.append(str(int(value)))

        if not cleaned:
            raise ValueError("Cannot build SQL list from an empty integer sequence.")

        return "(" + ", ".join(cleaned) + ")"
