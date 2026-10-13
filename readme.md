# KRNN Portfolio Risk Research

This is an earlier Python/PyTorch project where I combined return forecasts, tail-risk estimates, and portfolio optimization.

The model uses parallel GRU encoders to estimate next-day return and volatility. I use EVT/Hill estimates and moment-based bounds to study tail losses, generate Monte Carlo return scenarios, and solve a mean-CVaR linear program with SciPy/HiGHS. The optimizer allows long and short positions.

Training, validation, and test data are split by date. Feature scaling is fitted on the training data only. The backtest uses the unscaled next-day return column, `Target`.

## Run

From the repo root:

```sh
python -m pip install -r requirements.txt
python main_pipeline.py
```

Settings are in `config_v5.yaml`. The current default data source is WRDS, which needs your own access. Set `data.source` to `yfinance` to use the alternate source. Cached data are reused when the dataset manifest matches the selected source and cache version.

The run writes a summary and plots to `reports/`. The code is split between data preparation, the GRU model, risk calculations, and portfolio optimization under `src/`.

TA-Lib may need a separate installation depending on your platform. Some tests still use older interfaces and need updating for the current regression model.

## Notes

This is a research example, not evidence of a trading edge. The current-membership universe can introduce survivorship bias, and the backtest omits trading costs, slippage, and borrow fees.

Assets are simulated independently, so the scenarios do not capture joint market moves. Student-t shocks are not scaled to unit variance, so the volatility input does not equal the simulated standard deviation. The backtest also fills missing returns with zero and uses weighted log returns, which limits how its performance figures should be interpreted.

See the [data-source notes](KRNN_WRDS_MIGRATION_CONTEXT.md) for more context. My newer [portfolio construction project](https://github.com/ItsKamatis/constrained-portfolio-construction) focuses on constraints, solver checks, and comparisons between construction methods.
