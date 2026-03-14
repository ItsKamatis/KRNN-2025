# KRNN WRDS Migration Context

As of March 13, 2026.

This note records the current KRNN project context, what is reusable from the imported WRDS reference files, and the plan for replacing the current market-data source with WRDS while keeping the rest of the research pipeline stable.

## 1. What KRNN Is

Primary project goal:

- Predict next-day stock returns with a probabilistic KRNN that outputs `(mu, sigma)`.
- Standardize residuals and measure tail risk with EVT plus Naumova-style discrete conditional moment bounds.
- Build a tail-aware portfolio with mean-CVaR optimization.
- Evaluate the resulting portfolio out of sample.

Research report reviewed:

- `KRNN Risk Management_ Robust Portfolio Construction.pdf`
- Report date: December 22, 2025

Main report conclusions:

- The predictive model contributed little alpha on the reported run: `R^2 ~= -0.0285`.
- The project still produced a sensible risk-managed portfolio because the tail-risk layer, not the predictor, drove most of the value.
- Reported out-of-sample results for the 2024+ test period:
  - Strategy cumulative return: `32.52%`
  - Equal-weight benchmark cumulative return: `42.47%`
  - Strategy annualized volatility: `20.56%`
  - Benchmark annualized volatility: `22.69%`
  - Strategy Sharpe: `0.8053`
  - Benchmark Sharpe: `0.9527`

The project is fundamentally a risk-management and portfolio-construction pipeline, not a pure forecasting project.

## 2. Current KRNN Data Flow

Observed orchestration:

- Entry point: `main_pipeline.py`
- Data bootstrap: `initialize_data_pipeline()`
- Data collection boundary: `src/data/data_collector_v5.py`
- Feature engineering and scaling: `src/data/features_v5.py`
- Sequence dataset contract: `src/data/dataset_v5.py`

Actual current source logic:

1. `DataCollector.get_nasdaq100_tickers()` scrapes the current Nasdaq-100 list from Wikipedia.
2. `DataCollector.collect_data()` downloads daily OHLCV with `yfinance` starting at `config_v5.yaml:data.train_start`.
3. `FeatureEngineer.generate_features()` creates:
   - `Log_Return`
   - `RSI`
   - `MACD_Rel`
   - `MACD_Sig_Rel`
   - `BB_Width`
   - `BB_Pos`
   - `ATR_Rel`
   - `Log_Volume`
   - `Target` as next-day log return via `shift(-1)`
4. Data is split by time:
   - train: before `2023-01-01`
   - validation: `2023-01-01` to `2023-12-31`
   - test: `2024-01-01` onward
5. `StandardScaler` is fit on train only and applied only to the engineered feature columns.
6. The pipeline writes:
   - `data/train.parquet`
   - `data/validation.parquet`
   - `data/test.parquet`

Important current data contract:

- Downstream code expects per-row stock observations with at least:
  - `Date`
  - `Ticker`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Volume`
  - engineered features
  - `Target`
- `StockDataset` groups by `Ticker`, sorts by `Date`, and builds rolling windows.
- The safest migration is to keep the parquet schema and split filenames unchanged.

Observed parquet schema today:

- Columns:
  - `Date`
  - `Close`
  - `High`
  - `Low`
  - `Open`
  - `Volume`
  - `Ticker`
  - `RSI`
  - `MACD_Rel`
  - `MACD_Sig_Rel`
  - `BB_Pos`
  - `BB_Width`
  - `ATR_Rel`
  - `Log_Volume`
  - `Log_Return`
  - `Target`
- Current split coverage:
  - train: 99 tickers, `2018-02-20` to `2022-12-30`
  - validation: 101 tickers, `2023-01-03` to `2023-12-29`
  - test: 101 tickers, `2024-01-02` to `2025-12-19`

Notes:

- The feature columns are scaled.
- Raw OHLCV columns remain unscaled.
- `Target` remains unscaled.
- The current source path has survivorship bias because it uses a current Wikipedia membership list and backfills history for those names.

## 3. What The Imported WRDS Files Actually Are

Imported reference files:

- `wrdsquerying.ipynb`
- `PROJECT_DATA_FLOW_AUDIT.md`

These files are from another project. They are not KRNN-native and they are not directly portable as domain logic.

### 3.1 `wrdsquerying.ipynb`

What it does:

- Connects to WRDS via `wrds.Connection()`.
- Uses config-anchored paths.
- Defines reusable helpers:
  - `read_sql()`
  - `table_exists()`
  - `table_cols()`
  - `pick_col()`
- Pulls OptionMetrics auxiliary data:
  - `secprd`
  - `zerocd`
  - `distrd`
- Writes CSV artifacts to an aux directory.

What is reusable for KRNN:

- Connection setup pattern.
- Parameterized SQL execution.
- Schema introspection instead of hard-coding every column assumption.
- Config-driven paths.
- Small, auditable extraction functions per dataset.
- Saving raw extracts separately from downstream processed files.

What is not reusable as-is:

- OptionMetrics-specific table names and logic.
- `secid`-based identification.
- Zero-curve and dividend-window logic.
- Anything related to options carry, expiries, or auxiliary option pricing inputs.

### 3.2 `PROJECT_DATA_FLOW_AUDIT.md`

What it does:

- Documents another project's full data lineage from WRDS OptionMetrics ingestion to IV surfaces and C++ pricing benchmarks.

What is reusable for KRNN:

- The discipline of writing down:
  - source tables
  - fields pulled
  - transformations
  - output artifacts
  - assumptions
  - risks and known shortcuts
- The idea that the data layer should be auditable field by field.

What is not reusable as-is:

- The pipeline content itself. The audit is about options, SVI surfaces, carry construction, and pricing engines, none of which belong to KRNN.

Conclusion:

- We should reuse the WRDS extraction pattern and the documentation style.
- We should not reuse the options-specific business logic.

## 4. Recommended WRDS Source For KRNN

KRNN needs daily equity OHLCV for a stock universe, not options data.

Preferred WRDS source:

- CRSP daily stock data on WRDS.

Preferred implementation strategy:

- Prefer the newer CRSP CIZ-style daily security data if the account exposes a table with native daily OHLCV fields.
- If that exact table is not available in the account, fall back to legacy CRSP daily stock files plus a name-history mapping table.
- Use the same schema-introspection pattern as the notebook so the code adapts to the exact table names available in the user's WRDS environment.

Why CRSP is the correct fit:

- It is the appropriate WRDS-equity source for historical daily stock data.
- It is much more suitable than the imported OptionMetrics notebook for this project.
- It avoids dependence on Yahoo data quality, Wikipedia scraping, and current-membership survivorship assumptions.

Official WRDS references reviewed while planning:

- WRDS Python package and connection workflow: https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replication-mfa-support/
- WRDS overview for the Python package: https://wrds-www.wharton.upenn.edu/pages/grid-items/wrds-python-package/
- WRDS note on CRSP access changes / Flat File 2.0 transition: https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/crsp/changes-to-crsp-data/

## 5. Target Architecture For The Migration

The key design rule:

- Replace only the raw market-data ingestion layer first.
- Preserve the downstream KRNN training, risk, optimization, and reporting interfaces.

Recommended target flow:

1. Read WRDS credentials from environment or WRDS local configuration.
2. Query a WRDS equity history source for a stock universe and date range.
3. Normalize the raw extract to the current KRNN contract:
   - `Date`
   - `Ticker`
   - `Open`
   - `High`
   - `Low`
   - `Close`
   - `Volume`
4. Run the existing `FeatureEngineer`.
5. Keep the existing chronological split logic.
6. Keep the current output files:
   - `train.parquet`
   - `validation.parquet`
   - `test.parquet`

Recommended internal files to add during implementation:

- `src/data/wrds_client.py`
  - own the WRDS connection and generic SQL helpers
- `src/data/wrds_equity_source.py`
  - own CRSP-specific universe lookup and daily price pulls
- refactor `src/data/data_collector_v5.py`
  - become an orchestrator that dispatches by source type

Recommended config additions:

- `data.source: wrds | yfinance`
- `wrds.username_env`
- `wrds.use_crsp_ciz_first: true`
- `wrds.raw_cache_dir`
- `wrds.universe_mode`
  - `static_ticker_list`
  - `nasdaq100_current_members`
  - future option: `historical_index_membership`
- `wrds.start_date`
- `wrds.end_date`

## 6. Data Contract To Preserve

To avoid breaking the rest of the project, the WRDS-backed collector should still emit the same downstream columns.

Minimum preserved output:

- `Date`
- `Ticker`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `RSI`
- `MACD_Rel`
- `MACD_Sig_Rel`
- `BB_Pos`
- `BB_Width`
- `ATR_Rel`
- `Log_Volume`
- `Log_Return`
- `Target`

Recommended additions that are safe to include:

- `Permno`
- `Ret`
- `Retx`
- `Shares_Out`
- `Source`

These extra columns should not be used by `StockDataset` unless intentionally added to the feature set.

## 7. Practical Migration Plan

### Phase A. Introduce a WRDS extraction layer

Implement a reusable WRDS helper layer modeled after the imported notebook:

- connect once
- run parameterized SQL
- detect available schema/table/column names
- write raw extracts to disk before feature engineering

Suggested raw artifacts:

- `data/raw/wrds_equity_daily.parquet`
- `data/raw/wrds_symbol_map.parquet`
- `data/raw/wrds_extract_manifest.json`

### Phase B. Normalize WRDS data to the current KRNN schema

Normalize whichever CRSP table is available into:

- `Date`
- `Ticker`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Required QA checks:

- unique key on `(Ticker, Date)`
- monotonic date order within ticker
- no duplicate rows after joins
- numeric types on OHLCV
- no missing `Close`

### Phase C. Keep existing feature logic and parquet outputs

Do not change:

- `FeatureEngineer.generate_features()`
- train/validation/test split dates
- train-only scaling
- `StockDataset` sequence creation

This keeps the migration bounded to the data source instead of turning it into a full project refactor.

### Phase D. Add reproducibility and audit artifacts

Borrow the audit mindset from the other project and write:

- WRDS source tables used
- columns pulled
- filters applied
- date range pulled
- universe definition
- extraction timestamp
- row counts before and after cleaning

### Phase E. Compare WRDS output with current yfinance output

Before deleting the old path, compare:

- ticker counts
- date counts
- split sizes
- feature summary stats
- basic return distribution stats
- final pipeline behavior on a small pilot universe

## 8. Recommended First Implementation Scope

The first implementation should be narrow:

1. Keep the current static stock-project behavior.
2. Swap only the raw data pull from `yfinance` to WRDS.
3. Preserve current parquet file names and downstream schema.
4. Add a raw extract cache and a manifest.
5. Leave universe sophistication for later.

This is the lowest-risk path.

## 9. Important Open Questions

These should be answered before coding the full migration:

1. Which exact WRDS entitlements are available in this environment?
   - CRSP only?
   - CRSP CIZ tables?
   - legacy CRSP tables only?
2. Do we want to preserve the current survivorship-biased current-membership universe for comparability, or upgrade immediately to a historically correct universe?
3. Do we want to store permanent identifiers such as `permno` in the processed parquet files?
4. If a WRDS table does not provide native `Open`, do we require a CIZ-style source, or do we accept a reduced/derived OHLCV mapping?
5. Should the old `yfinance` path remain as a fallback for local development?

## 10. Recommendation

Recommended next action:

- Implement WRDS ingestion with a source selector and keep all downstream KRNN interfaces unchanged.

Concrete coding priority:

1. Add a WRDS client/helper module.
2. Add a CRSP-based equity extractor with schema introspection.
3. Refactor `DataCollector` to support `source='wrds'`.
4. Cache raw WRDS extracts.
5. Rebuild the existing parquet splits and compare them against the current `yfinance` output.

That gives KRNN a better institutional data source without disturbing the KRNN, EVT, DCMP, optimizer, or report-generation layers.
