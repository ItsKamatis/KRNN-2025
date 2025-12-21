# KRNN Risk Management Pipeline (Tail-Risk Parity Portfolio)

A full end-to-end risk management pipeline for constructing portfolios that stay robust in crash regimes where Gaussian assumptions fail.

Core idea: learn conditional mean/volatility with a heteroscedastic KRNN, then explicitly model and penalize tail risk using EVT + robust moment bounds, and finally allocate capital using Mean-CVaR optimization.

---

## What this does (high-level)

1. **Data + Features**
   - Pulls historical OHLCV data (e.g., via `yfinance`)
   - Builds technical features (RSI, MACD, ATR, Bollinger bands, log returns, etc.)
   - Builds next-day return targets

2. **Model: K-Parallel RNN (KRNN)**
   - K parallel GRUs encode the same sequence
   - Hidden representations are averaged (variance reduction / ensemble effect)
   - Outputs **(μ, σ)** for next-day return (heteroscedastic regression)

3. **Training**
   - Optimizes **Gaussian Negative Log Likelihood (GNLL)** instead of MSE
   - Early stopping + LR scheduling + gradient clipping

4. **Risk Layer**
   - Converts prediction errors into standardized residuals (z-scores)
   - Fits tail heaviness via EVT (POT + Hill estimator)
   - Computes tail metrics (VaR / ES on standardized residual scale)
   - Computes robust worst-case bounds via Discrete Conditional Moment Problem (DCMP)

5. **Portfolio Construction**
   - Simulates heavy-tailed return scenarios per asset (parametric bootstrap using γ + σ)
   - Solves **Mean-CVaR** optimization (Rockafellar & Uryasev LP)

---

## Mathematical backbone

### 1) Heteroscedastic training objective (GNLL)
Model predicts mean and volatility:  
- prediction: \( \mu_t, \sigma_t > 0 \)

Loss (Gaussian NLL):
\[
\mathcal{L}(\mu,\sigma;y) = \frac{(y-\mu)^2}{2\sigma^2} + \frac{1}{2}\log(\sigma^2)
\]
This penalizes errors relative to predicted uncertainty and prevents treating all errors equally (unlike MSE).

### 2) EVT tail estimation (POT + Hill estimator)
Standardized residuals:
\[
z_t = \frac{y_t - \mu_t}{\sigma_t}
\]
Focus on left-tail crashes using losses \( \ell = -z \).  
Tail index via Hill estimator (using top-k extremes):
\[
\hat{\gamma} = \frac{1}{k}\sum_{i=1}^{k}\left(\log \ell_i - \log \ell_{k+1}\right)
\]
Interpretation:
- small \( \gamma \): thinner tail
- large \( \gamma \): heavier tail, more extreme crashes

### 3) Mean-CVaR optimization (LP form)
Minimize CVaR at level \( \alpha \) over scenario returns:
\[
\min_{w,\phi,z} \ \phi + \frac{1}{(1-\alpha)T}\sum_{t=1}^T z_t
\]
Subject to:
\[
z_t \ge -w^\top r_t - \phi,\quad z_t \ge 0,\quad \sum_i w_i = 1,\quad w_i \ge 0,\quad w^\top \mu \ge R_{target}
\]
A dynamic target rule is used to avoid infeasibility in bearish regimes (target relaxes to the universe mean when needed).

### 4) DCMP (robust worst-case CVaR bounds)
Given only empirical moments (mean/variance/skewness) and a conditional tail anchor, solve a discrete LP to find a **model-free worst-case CVaR bound** (Naumova-style tightening).  
This is used as a robustness check against EVT-based parametric estimates.

---

## Repository structure (typical)

> Adjust names if yours differ, but keep the mapping clear.

- `main_pipeline.py`  
  Runs the end-to-end pipeline: train → scan → EVT → optimize → evaluate

- `src/data/`  
  Data ingestion, caching, dataset windows, dataloaders

- `src/models/`  
  KRNN / K-parallel GRU encoder, heteroscedastic head

- `src/training/` or `src/utils/`  
  Trainer (GNLL loss, early stopping, scheduler, grad clipping)

- `src/risk/`
  - EVT tail analysis (POT / Hill)
  - moment bounds (DCMP / robust worst-case CVaR)

- `src/portfolio/`  
  Mean-CVaR optimizer + scenario generator

- `outputs/` (optional)
  - saved plots like `test_performance.png`
  - cached parquet files / artifacts

---

## Installation

```bash
# recommended
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
````

TA-Lib is used for indicators, make sure it is installed correctly on your OS.

---

## Quickstart

```bash
python main_pipeline.py
```

Typical phases:

* load cached train/val/test datasets
* train KRNN with GNLL
* scan universe and rank by predicted μ
* compute EVT tail indices from z-scores
* generate heavy-tailed scenarios
* solve Mean-CVaR weights
* evaluate on test set + save plots

---

## Example output (what to expect)

You should see:

* training logs with Train/Val GNLL and early stopping
* “Top 10 candidates” with predicted μ and σ
* EVT tail index γ and ES(Z) per asset
* final Mean-CVaR weights and portfolio CVaR(95)
* test diagnostics and saved performance plot

---

## Common issues / design notes

* **Mean collapse (μ, σ nearly identical for many assets)**
  This can happen when the feature set provides limited alpha signal and the model learns a “safe constant” mean while focusing on volatility regimes. EVT on standardized residuals still differentiates tail risk across names.

* **Look-ahead bias avoidance**

  * targets are next-day returns (shifted)
  * time-based splits (train/val/test by date)
  * scalers fit on training set only
  * sequence windows never include future targets

* **Infeasible optimization in bearish regimes**
  Dynamic target return prevents empty feasible regions when all predicted μ are negative.

---

## Reproducibility

* Set random seeds (PyTorch / NumPy)
* Use cached datasets (parquet) for consistent splits
* Record config + commit hash in logs (recommended)

---

## Citation / references

* Rockafellar & Uryasev (2000): CVaR optimization and LP formulation
* EVT POT framework + Hill estimator (tail index estimation)
* Naumova (2015): conditional moment constraints for tighter robust bounds

