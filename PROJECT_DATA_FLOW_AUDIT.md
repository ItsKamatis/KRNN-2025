# Project Data Flow Audit

As-of code state: March 4, 2026  
Scope: end-to-end data flow from OptionMetrics ingestion to IV surfaces and C++ pricing benchmarks.

## 1. Executive Summary

This project runs a research pipeline that:

1. Pulls OptionMetrics chain + auxiliary market data from WRDS.
2. Cleans and segments option quotes.
3. Builds carry term structures (parity or WRDS-aux, by asset profile).
4. Inverts implied volatility using Black-76 on forwards.
5. Fits per-expiry raw SVI slices in total variance.
6. Builds a full surface, runs diagnostics, and optionally repairs.
7. Exports filtered quote baskets to C++ pricers.
8. Benchmarks pricing engines and surface sources with aggregate reporting.

The implementation is consistent with `MATHEMATICAL_APPROACH_REFERENCE.md` and `AGENT_MASTER_REFERENCE.md`.  
`QUANT_RIGOR_AUDIT.md` findings are also reflected in current benchmark decision logic (mean-RMSE-first with limited paired/tail gating).

## 2. Primary Orchestration

Top-level orchestrator: `run_pipeline.py`

Canonical script sequence for each `(ticker, date, contract_family)`:

1. `data/datacleaner.py`
2. `src/calculate_iv.py`
3. `src/fit_surface.py`
4. Optional: `src/repair_european_svi_params.py` (European research mode)
5. Optional: `src/fit_american_time_value_surface.py` (American research mode)
6. `src/vol_surface_model.py`
7. `src/generate_cpp_inputs.py`
8. Optional benchmark pricing comparison: `src/compare_pricing_engines.py`

Benchmark mode then aggregates outputs and writes report artifacts, including M4 quote-quality diagnostics and worst-date review packs.

## 3. Data Inputs and What Is Pulled

WRDS fetch script: `src/fetch_optionmetrics_data.py`

### 3.1 Option chain fields pulled

Core pricing/liquidity fields:

- `date`, `exdate`, `cp_flag`, `strike_price`
- `best_bid`, `best_offer`
- `volume`, `open_interest`

Additional vendor fields also pulled:

- `impl_volatility`
- `delta`, `gamma`, `vega`, `theta`
- `optionid`
- `am_settlement`, `expiry_indicator`, `root`, `suffix`
- other optional metadata

### 3.2 Auxiliary files pulled

- Security price: `secprd_*`
- Zero curve: `zerocd_*`
- Dividend schedule: `divs_*`

These support WRDS-aux carry construction for American-style assets and diagnostics/fallback for others.

## 4. Cleaning and Segmentation

Cleaner: `data/datacleaner.py`

Transforms:

1. Parse dates.
2. Convert strike scale (`strike_price / 1000`).
3. Compute `mid_price = 0.5*(bid+ask)` and `spread = ask-bid`.
4. Filter invalid quotes (`bid >= 0`, `ask > 0`, `bid <= ask`, `mid >= min_mid`).
5. Keep calls/puts only.
6. Optional liquidity filter: `(volume > 0) OR (open_interest > 0)`.
7. Compute `rel_spread = spread / max(mid_price, eps)` and filter by threshold.
8. Compute `T = (ex_date - valuation_date).days / 365.25`.
9. Deduplicate per `(ex_date, strike, cp_flag)` by tight spread then high OI.

Segmentation:

- Contract families assigned in `src/contract_family.py`:
  - `pooled`
  - `monthly_am`
  - `weekly_pm`
  - `other_mixed`

## 5. Carry and IV Inversion

Main script: `src/calculate_iv.py`

### 5.1 Carry source by asset profile

From `config.yaml` via `src/market_inputs.py`:

- European index names (SPX/NDX/RUT): `carry_source=parity`, `exercise_style=european`
- American names (default/SPY): `carry_source=wrds_aux`, `exercise_style=american`

### 5.2 Parity-based term structure (European canonical)

For each expiry, on paired strikes:

- Fit weighted linear model:
  - `C - P = alpha + beta*K`
- Map to carry:
  - `D = -beta`
  - `F = alpha / D`
  - `r = -ln(D)/T`
- Pair weights:
  - `w = 1 / pair_spread`
  - `pair_spread = spread_call + spread_put`

### 5.3 WRDS-aux term structure (American canonical)

Built from:

- inferred spot (`chain` first, then `secprd`)
- zero curve from `zerocd`
- PV of dividends from `divs`

Per expiry:

- `D = exp(-rT)` from zero curve
- `F = (spot - PV(dividends_to_expiry)) / D`
- implied `q` backed out from `F, r, S`

### 5.4 OTM switch policy

After carry fit/merge:

- Keep put if `K < F(T)`
- Keep call if `K > F(T)`
- Keep both sides in near-ATM band (`|K-F|/F <= atm_band`)

This preserves parity information before switching.

### 5.5 IV inversion

Uses Black-76 on forward:

- Price model: `Black(F, K, T, D, sigma)`
- Root solve: Brent (`scipy.root_scalar`, `brentq`)
- Enforces option price bounds before inversion.

## 6. Surface Calibration and Construction

### 6.1 Baseline SVI fit

Script: `src/fit_surface.py`

Model per expiry:

- Raw SVI total variance:
  - `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`

Objective:

- Weighted least squares in total variance:
  - `sum weight * (w_model - w_mkt)^2`
- Weight proxy:
  - `weight = (vega / half_spread)^2` (clipped)
- Solver:
  - multi-start L-BFGS-B with parameter bounds

### 6.2 Surface build

Script: `src/vol_surface_model.py`

Construction:

1. Forward/discount curves over maturity in log-space (`logF`, `logD`) with PCHIP.
2. At fixed `k`, interpolate total variance across `T` (PCHIP in-range, slope-guarded extrapolation).
3. Convert to vol:
   - `sigma(K,T) = sqrt(w(k,T)/T)` with `k = ln(K/F(T))`.

### 6.3 Diagnostics and repairs

Diagnostics include:

- Slice butterfly (`g(k)`) violations
- Surface butterfly violations
- Calendar monotonicity violations in total variance
- Lee-style wing slope checks

Optional repairs:

- Grid-level calendar isotonic projection
- Row-wise butterfly projection via constrained optimizer (`trust-constr`)

Research-only alternatives:

- `param-repaired` SVI params (`src/repair_european_svi_params.py`)
- `american-tv-fit` params (`src/fit_american_time_value_surface.py`)

## 7. C++ Export and Basket Selection

Exporter: `src/generate_cpp_inputs.py`

### 7.1 Candidate filtering

From cleaned chain:

- `T in [min_T, max_T]`
- liquidity OR gate:
  - `volume >= min_volume` OR `open_interest >= min_open_interest`
- `rel_spread <= max_rel_spread`

Defaults in `config.yaml`:

- `n_options = 50`
- `min_T = 0.02`
- `max_T = 0.25`
- `min_volume = 10`
- `max_rel_spread = 0.40`

### 7.2 Basket construction

1. Compute `k = ln(K/F(T))`.
2. Rank by liquidity score (`volume + 0.01*open_interest`).
3. Stratify with `qcut` bins on `k`.
4. Take top-per-bin then fill remainder by liquidity.

### 7.3 Carry convention at export

Supported European export carry modes:

- `surface_parity` (canonical coherent mode)
- `legacy_mixed_aux` (legacy replay)

WRDS-aux assets use aux carry when available.

Export row format written for C++:

- `S,K,T,r,q,sigma,isCall,mktPrice` (headerless CSV)

## 8. C++ Pricing, Black-Scholes, and Greeks

C++ entrypoint: `ProjectB_Cpp/main.cpp`

Supported engines:

- `black-scholes` (European closed form)
- `crr` (European/American binomial tree)
- `fd-implicit` (European PDE)
- `fd-cn` (European Crank-Nicolson)
- `fd-cn-psor` (American CN + PSOR + Rannacher)

### 8.1 Black-Scholes usage

- Python side: Black-76 for IV inversion.
- C++ side: Black-Scholes in `(S,r,q,sigma)` for European pricing.

### 8.2 Greeks

Greeks are computed in C++ only when `--greeks` is set:

- Delta/Gamma: spot bumps
- Vega: vol bump
- Rho: rate bump
- Theta: time bump

Method: finite differences around base model price.

Important current behavior:

- Python benchmark wrapper (`src/compare_pricing_engines.py`) does not pass `--greeks`.
- So default benchmark outputs are primarily price-error metrics, not risk-metric validation.

## 9. What OptionMetrics Data Is Used vs Not Used

### 9.1 Used directly in core pipeline

- `best_bid`, `best_offer`, `cp_flag`, `strike_price`, `date`, `exdate`
- `volume`, `open_interest`
- segmentation metadata (`am_settlement`, `expiry_indicator`, `root`, `suffix`)
- auxiliary `secprd/zerocd/divs` (for WRDS-aux carry)

### 9.2 Pulled but not used as primary calibration targets

- Vendor `impl_volatility`
- Vendor Greeks `delta/gamma/vega/theta`

These are available for future QA and cross-checks but are not current primary fit inputs.

## 10. Current Shortcuts and Assumptions

1. Mid-price centric calibration and benchmarking.
2. Daycount hardcoded as `365.25`.
3. Export basket intentionally small and liquidity-biased.
4. Baseline SVI is slice-wise independent (not jointly constrained across maturities).
5. Static-arbitrage is diagnosed and optionally repaired post-fit; not guaranteed by baseline parameterization.
6. Benchmark decision logic is currently mean-RMSE-first.

## 11. Known Methodology Risks (Aligned with Quant Audit)

From `QUANT_RIGOR_AUDIT.md`, current policy gaps include:

1. Mean-first promotion logic without paired superiority gates.
2. Insufficient tail-risk gating for challenger surfaces.
3. Partial mismatch between diagnostics and pricing-failure behavior.
4. Evaluation representativeness concerns due to export-basket sampling.

This is why repair variants remain research-only in current documentation/policy.

## 12. Argument Surfaces

### 12.1 `run_pipeline.py`

Modes:

- `snapshot`
- `backfill`
- `benchmark`

Notable arguments:

- dates/tickers ranges
- `--contract-families`
- `--surface-source` or `--surface-sources`
- `--compare-engines`, `--compare-top`, `--compare-pricer-workers`
- `--european-export-carry-mode`
- `--american-tv-fit`
- `--strict-cpp`
- `--no-fetch`, `--refresh-fetch`, `--wrds-username`

### 12.2 `src/generate_cpp_inputs.py`

- `--ticker`
- `--n`
- `--surface-source`
- `--european-export-carry-mode`
- `--strict`
- `--no-project-copy`

### 12.3 `src/compare_pricing_engines.py`

- `--input`, `--output-dir`, `--pricer-exe`
- `--engines`
- tree/FD parameters
- `--pricer-workers`
- `--top`

### 12.4 C++ `VolSurf_Pricer`

- `--input`
- `--engine`
- `--style`
- tree/FD controls
- `--workers`
- `--report` / `--output-dir`
- `--top`
- `--greeks` and bump parameters

## 13. Practical Answer to "Do We Need Greeks?"

For current milestone objective (surface-source and engine pricing error comparison): not strictly required.

For production risk usage or hedge quality validation: yes, Greeks should be computed and benchmarked, and ideally compared against independent references (analytic where available, or stable numerical baselines where not).

## 14. Suggested Next Documentation Additions

To make this audit operational, add two follow-on artifacts:

1. Field-level lineage table (column-by-column source, transform, and sink).
2. Assumption register (carry convention, daycount, filters, basket policy, promotion gates).

