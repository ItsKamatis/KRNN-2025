# KRNN: Tail-Risk Parity Portfolio Construction

This repository implements an end-to-end risk management pipeline for constructing a **tail-aware portfolio** on the Nasdaq-100 universe.

The system is built around three technical pillars:
1. A probabilistic **KRNN** (K-parallel GRU encoder) that outputs **(μ, σ)** for next-day log returns.
2. Tail modeling on standardized residuals using **EVT (Hill/POT)** plus **moment-based robust bounds (Naumova-style DCMP)**.
3. A **Mean–CVaR optimizer** (Rockafellar–Uryasev linear program) that allocates capital to minimize tail loss, with optional long/short weights.

The main entrypoint is `main_pipeline.py`, which runs:

Data → Features → Time split → Scaling (train-only) → KRNN training (GNLL) → Candidate selection → EVT + DCMP tail risk → Scenario simulation → Mean–CVaR optimization → Out-of-sample evaluation → Report generation


---

## What this produces

Running the full pipeline generates:

- `reports/comprehensive_summary.txt`  
  A consolidated summary including per-ticker (μ, σ), Hill tail index (γ), EVT ES, DCMP worst-case ES, and final portfolio weights.

- Plots in `reports/` (filenames may vary slightly):
  - `3_allocation_logic.png` – weights vs tail index γ
  - `4_prediction_accuracy.png` – predicted μ vs realized target on the test set (R²)
  - `5_risk_bounds_gap.png` – EVT ES vs DCMP worst-case ES comparison

A typical run shows a “mean collapse” symptom: predicted μ and σ become nearly identical across many tickers, while the standardized residuals still reveal very different tail behavior—exactly what the EVT/DCMP stages are designed to exploit.

---

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

**TA-Lib note:** TA-Lib can be is an annoying dependency on Windows. If pip install fails, install TA-Lib via a platform-appropriate method (often easiest via conda-forge), then re-run the requirements install.

### 2) Run the full pipeline

From the repository root:

```bash
python main_pipeline.py
```

What happens if data files already exist?
- If `data/train.parquet`, `data/validation.parquet`, and `data/test.parquet` exist, the pipeline **skips downloads**.
- Otherwise, it downloads Nasdaq-100 tickers and historical OHLCV using `yfinance`.

### 3) Inspect results

- Text summary: `reports/comprehensive_summary.txt`
- Figures: `reports/*.png`

---

## Repository layout

```
KRNN/
├── main_pipeline.py              # full end-to-end run
├── config_v5.yaml                # experiment + data + training config
├── requirements.txt
├── data/                          # parquet splits + checkpoints (created)
├── reports/                       # plots + summary report (created)
└── src/
    ├── data/
    │   ├── data_collector_v5.py   # download → feature eng → split → scale → save
    │   ├── features_v5.py         # stationary features + StandardScaler
    │   └── dataset_v5.py          # sequence dataset (per-ticker windows) + DataModule
    ├── model/
    │   └── krnn_v5.py             # KRNNRegressor (μ, σ) + K-parallel encoder
    ├── risk/
    │   ├── evt.py                 # Hill estimator, EVT VaR/ES, scenario generation
    │   └── moment_bounds.py       # DCMP robust VaR/ES (Naumova-style anchor)
    ├── portfolio/
    │   └── optimizer.py           # Mean–CVaR linear program (long/short)
    └── utils/
        ├── trainer_v5.py          # GNLL training loop + early stopping
        └── generate_report.py     # plots + report writer
```

---

## Pipeline, step-by-step

### Phase 0 — Reproducibility and data availability

`main_pipeline.py` sets deterministic seeds (NumPy + PyTorch) and checks for existing parquet splits.

### Phase 1 — Data collection, feature engineering, and leakage control

Implemented in `src/data/data_collector_v5.py` + `src/data/features_v5.py`.

Process:
1. Download OHLCV using `yfinance` from `train_start` onward.
2. Build stationary features (see next section).
3. Build the supervised target:
   - `Log_Return_t = log(Close_t / Close_{t-1})`
   - `Target_t = Log_Return_{t+1}` (implemented via `shift(-1)`)
4. Time-aware split:
   - train: `Date < train_end`
   - validation: `train_end ≤ Date < val_end`
   - test: `Date ≥ val_end`
5. Fit `StandardScaler` **only on the training set**, then transform train/val/test.
   - This avoids look-ahead bias from feature normalization.

Saved outputs:
- `data/train.parquet`
- `data/validation.parquet`
- `data/test.parquet`

### Features used (v5)

Defined in `FeatureEngineer.generate_features()`:

- `Log_Return` : log(Close_t / Close_{t-1})
- `RSI` : 14-day RSI
- `MACD_Rel`, `MACD_Sig_Rel` : MACD and signal normalized by price
- `BB_Width`, `BB_Pos` : Bollinger bandwidth and position
- `ATR_Rel` : ATR normalized by price
- `Log_Volume` : log(1 + Volume)

Scaling:
- All 8 features above are standardized using `StandardScaler`.
- Fit happens on **train only**; transforms applied to train/val/test.

### Phase 2 — KRNN training (heteroscedastic regression)

Model: `src/model/krnn_v5.py` (`KRNNRegressor`)

KRNN structure:
- Create **K independent GRU encoders** (default `k_dups=3`).
- Run the same input sequence through each GRU.
- Take the last time-step hidden state from each.
- Mean-pool across the K representations to reduce variance.

Outputs:
- `μ` = predicted mean next-day return
- `σ` = predicted standard deviation (enforced positive via `softplus`)

Loss: Gaussian Negative Log-Likelihood (GNLL), implemented by PyTorch’s `GaussianNLLLoss` in `src/utils/trainer_v5.py`.

Per-sample GNLL (up to constants):
\[
\mathcal{L}(\mu,\sigma;y) = \tfrac12 \log(\sigma^2) + \tfrac12\frac{(y-\mu)^2}{\sigma^2}
\]

### Phase 3 — Candidate selection + EVT and DCMP risk analysis

After training, `main_pipeline.py` scans each ticker in the validation set:
- builds per-ticker sequences
- runs inference to collect `μ_t`, `σ_t`, and targets `y_t`
- computes per-ticker averages:
  - `Pred_Return = mean(μ_t)`
  - `Pred_Vol = mean(σ_t)`
- computes standardized residuals:
\[
Z_t = \frac{y_t - \mu_t}{\sigma_t + \epsilon}
\]

Then:
- Select top-N candidates by `|Pred_Return|` (long/short potential).
- For each candidate, estimate tail behavior on residuals `Z`.

EVT (Hill/POT), in `src/risk/evt.py`:
- Work on left tail (losses): `Loss = -Z`
- Use tail fraction (default 10%) and Hill estimator:
\[
\hat{\gamma} = \frac{1}{k}\sum_{i=1}^k \left(\log L_i - \log L_{(k)}\right)
\]
- EVT VaR/ES (in Z-units) at confidence \(p\) (default 0.99):
\[
VaR_p = u\left(\frac{k}{n(1-p)}\right)^{\gamma}, \qquad ES_p=\frac{VaR_p}{1-\gamma}\quad(\gamma<1)
\]

DCMP (moment-based robust bound), in `src/risk/moment_bounds.py`:
- Solve a linear program over a discrete support \(z\in[-10,10]\) to find a worst-case tail distribution consistent with empirical moments.
- Adds a conditional “tail anchor” constraint (Naumova-style) to prevent the solution from dumping all mass at an extreme grid point.

Outputs per ticker:
- Hill tail index γ
- EVT Expected Shortfall (ES) in Z-units
- DCMP worst-case ES in Z-units

### Phase 4 — Scenario simulation + Mean–CVaR optimization

Scenario simulation (EVT + KRNN forecasts), in `src/risk/evt.py`:
- Simulate standardized shocks \(Z\) as:
  - Gaussian if γ is near 0
  - Student‑t with \(df \approx 1/\gamma\) otherwise
- Convert to return scenarios:
\[
R = \mu + \sigma Z
\]

Optimization (Rockafellar–Uryasev) in `src/portfolio/optimizer.py`:
- Solve for weights \(w\) minimizing portfolio CVaR at level α (default 0.95).
- Allows long/short weights \(w_i\in[-1,1]\) with net exposure constraint \(\sum w_i = 1\).

Objective:
\[
\min \ \gamma + \frac{1}{(1-\alpha)T}\sum_{t=1}^T z_t
\]

Constraints (simplified):
- \(z_t \ge -w^\top r^{(t)} - \gamma\)
- \(w^\top \mu \ge R_{\text{target}}\)
- \(\sum_i w_i = 1\)

Target return logic in this repo:
- uses a conservative function of the top-quartile of \(|\mu|\) estimates from validation.

### Phase 5 — Out-of-sample evaluation and portfolio backtest

Two related evaluations happen:

1) **Prediction diagnostics** on the test loader:
- R² between predicted μ and realized target.

2) **Portfolio backtest** on the test period:
- Build a Date×Ticker matrix using `Log_Return` from the parquet.
- Apply the optimized weights (renormalized if tickers are missing).

Important note about return scaling:
- The parquet splits contain **scaled features** (StandardScaler applied), including `Log_Return`.
- `main_pipeline.py` therefore multiplies the pivoted `Log_Return` matrix by `0.02` as a rough conversion back to “percent-ish” units.
- For a fully correct backtest, keep an **unscaled return column** (raw log return) before normalization and use that for performance metrics.

---

## Configuration

Edit `config_v5.yaml`:

- `data.window_size`: sequence length per ticker (default 60)
- `data.batch_size`: training batch size
- `data.train_start`, `data.train_end`, `data.val_end`: time splits
- `model.hidden_dim`, `model.num_layers`, `model.dropout`: KRNN capacity
- `training.epochs`, `training.learning_rate`, early stopping settings
- `portfolio.top_n_assets`, `portfolio.risk_confidence`, `portfolio.n_simulations`

---

## Notes on look-ahead bias

This pipeline is designed to avoid common “quiet leaks”:

- Time splits are strictly chronological.
- Feature scaling is **fit on train only** then applied to val/test.
- The target is next-day return via `shift(-1)` (features at time t predict t+1).
- Portfolio selection uses **validation** only; the **test** period is held out for evaluation.

Remaining realism gaps (expected in a research repo):
- transaction costs, borrow fees, and slippage are not modeled
- “close-to-close” targets imply the signal is available after the close; live deployment needs a clear trade timestamp and execution assumption

---

## Running tests

There is a `tests/` folder, but some tests reference older APIs that no longer match the current `v5` modules (ongoing refactor). If running tests, expect to update the tests to match the current `FeatureEngineer` and `StockDataset` implementations.

---

## Troubleshooting

- TA-Lib install fails: install TA-Lib via a platform-appropriate method, then install the rest with pip.
- No data collected: some tickers may fail downloads; re-run, or reduce the universe for debugging.
- Not enough sequences: reduce `data.window_size` or extend the training date range.
- Optimizer infeasible: lower the target return or relax constraints; confirm scenarios and expected returns have reasonable scale.
- Backtest volatility looks absurd: you are likely using standardized returns; use raw returns for the backtest.

---

## License / attribution

This is a research codebase built for risk modeling and portfolio construction experiments. If adapting it for live trading, add:
- a strict data audit trail,
- robust time-stamping assumptions,
- cost/slippage modeling,
- and a proper walk-forward evaluation framework.
