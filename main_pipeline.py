import torch
import numpy as np
import pandas as pd
import yaml
import logging
import random
import os
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from src.data.data_collector_v5 import DataCollector
from src.data.dataset_v5 import DataModule, DataConfig, StockDataset
from src.model.krnn_v5 import KRNNRegressor, ModelConfig
from src.utils.trainer_v5 import Trainer
from src.risk.evt import EVTEngine
from src.risk.moment_bounds import DiscreteConditionalMomentSolver
from src.portfolio.optimizer import MeanCVaROptimizer
from src.utils.generate_report import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 2304571, deterministic: bool = False):
    """Seed RNGs and optionally force deterministic cuDNN.

    For speed on RTX 4090, keep deterministic=False so we can enable:
      - cuDNN benchmark (autotuner)
      - TF32 matmul (fast fp32)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    # 4090: TF32 is a big speed win for RNN matmuls with minimal accuracy loss.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # PyTorch 2.x: nudges matmul kernels toward faster implementations
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger.info(f"Global Seed set to {seed} (deterministic={deterministic})")


def initialize_data_pipeline(config_path: str = "config_v5.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_dir = Path(config['paths']['data'])
    required_files = ['train.parquet', 'validation.parquet', 'test.parquet']
    if all((data_dir / f).exists() for f in required_files):
        logger.info("Data files found. Skipping download.")
        return config
    logger.info("Data files missing. Starting Data Collection...")
    collector = DataCollector(config)
    collector.collect_data()
    return config


def run_project_pipeline():
    # --- Step 0: Robustness & Init ---
    print("\n[Phase 0] Initializing Pipeline & Checking Data...")
    config_dict = initialize_data_pipeline("config_v5.yaml")

    # Speed vs reproducibility toggle
    deterministic = bool(config_dict.get('training', {}).get('deterministic', False))
    set_seed(42, deterministic=deterministic)

    reporter = ReportGenerator(output_dir="./reports")

    # --- Step 1: Data Loading ---
    print("\n[Phase 1] Loading Data Module (RAM Cache)...")
    data_config = DataConfig(
        sequence_length=config_dict['data']['window_size'],
        batch_size=config_dict['data']['batch_size'],
        num_workers=int(config_dict.get('hardware', {}).get('num_workers', 0)),
        pin_memory=bool(config_dict.get('hardware', {}).get('pin_memory', True)),
    )
    data_module = DataModule(data_config)
    data_module.setup(Path(config_dict['paths']['data']))
    dataloaders = data_module.get_dataloaders()

    # --- Step 2: Model Training ---
    print("\n[Phase 2] Training K-Parallel KRNN (Heteroscedastic - Overfit Mode)...")
    feature_dim = len(data_module.train_dataset.feature_cols)
    model_config = ModelConfig(
        input_dim=feature_dim,
        hidden_dim=config_dict['model']['hidden_dim'],
        k_dups=3,
        output_dim=2,
        dropout=config_dict['model']['dropout'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = KRNNRegressor(model_config)

    # Optional: torch.compile can help reduce Python overhead (PyTorch 2.x)
    if bool(config_dict.get('training', {}).get('compile', False)) and model_config.device.startswith('cuda'):
        try:
            model = torch.compile(model, mode="max-autotune")
            logger.info("torch.compile enabled (mode=max-autotune)")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without it: {e}")

    trainer = Trainer(model, config=config_dict)
    training_metrics = trainer.train(train_loader=dataloaders['train'], val_loader=dataloaders['val'])
    print(f"Training Complete. Final Val GNLL: {training_metrics.get('loss', 'N/A')}")

    # --- Step 3: Systematic Portfolio Selection ---
    print("\n[Phase 3] Running Systematic Portfolio Selection & Robust Risk Analysis...")

    val_df_path = Path(config_dict['paths']['data']) / 'validation.parquet'
    val_df = pd.read_parquet(val_df_path)
    all_tickers = val_df['Ticker'].unique()

    candidates = []
    # Configs
    top_n = config_dict.get('portfolio', {}).get('top_n_assets', 10)
    risk_conf = config_dict.get('portfolio', {}).get('risk_confidence', 0.95)
    n_sims = config_dict.get('portfolio', {}).get('n_simulations', 1000)

    model.eval()

    # Inference settings (higher batch -> more GPU memory -> higher throughput)
    infer_bs = int(config_dict.get('data', {}).get('inference_batch_size', max(1024, data_config.batch_size * 16)))
    use_amp_infer = bool(config_dict.get('training', {}).get('use_amp', True)) and model_config.device.startswith('cuda')
    amp_dtype = str(config_dict.get('training', {}).get('amp_dtype', 'bf16')).lower()
    infer_dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16

    # --- 3a. Inference & Alpha Scan ---
    for ticker in all_tickers:
        ticker_df = val_df[val_df['Ticker'] == ticker].copy()
        if len(ticker_df) < data_config.sequence_length + 20:
            continue
        ticker_ds = StockDataset(ticker_df, data_config.sequence_length)
        ticker_loader = DataLoader(
            ticker_ds,
            batch_size=infer_bs,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=(data_config.num_workers > 0),
        )

        # Keep everything on GPU during the loop; transfer to CPU once per ticker.
        device = model_config.device
        sum_mu = torch.zeros((), device=device)
        sum_sigma = torch.zeros((), device=device)
        count = 0
        residual_chunks = []

        with torch.inference_mode():
            for feat, targ in ticker_loader:
                feat = feat.to(device, non_blocking=True)
                targ = targ.to(device, non_blocking=True).squeeze(-1).float()

                if use_amp_infer and device.startswith('cuda'):
                    with torch.autocast(device_type='cuda', dtype=infer_dtype):
                        mu, sigma = model(feat)
                else:
                    mu, sigma = model(feat)

                sum_mu += mu.sum()
                sum_sigma += sigma.sum()
                count += mu.numel()

                # Standardized residuals Z = (y - mu)/sigma
                residual_chunks.append(((targ - mu) / (sigma + 1e-6)).detach())

        if count == 0:
            continue

        avg_pred_return = float((sum_mu / count).item())
        avg_pred_volatility = float((sum_sigma / count).item())
        residuals = torch.cat(residual_chunks, dim=0).float().cpu().numpy()

        candidates.append({
            'Ticker': ticker,
            'Pred_Return': avg_pred_return,
            'Pred_Vol': avg_pred_volatility,
            'Residuals': residuals
        })

    # Sort by Absolute Return potential (since we can short)
    candidates.sort(key=lambda x: abs(x['Pred_Return']), reverse=True)
    top_candidates = candidates[:top_n]
    print(f"   Selected Top {len(top_candidates)} Candidates (Long/Short Potential).")

    # --- 3b. Robust Risk Analysis ---
    print(f"\n3. Risk Filter: Analyzing Heavy Tails (EVT) & Bounds (Naumova DCMP)...")

    scenarios_list = []
    final_tickers = []
    expected_returns_vec = []
    evt_engine = EVTEngine(tail_fraction=0.10)
    dcmp_solver = DiscreteConditionalMomentSolver(n_points=500, support_range=(-10.0, 10.0))

    tail_data = {}
    candidates_report_data = []
    bounds_report_data = []

    for cand in top_candidates:
        ticker = cand['Ticker']
        residuals = cand['Residuals']

        evt_params = evt_engine.analyze_tails(residuals)
        gamma = evt_params['gamma']
        evt_metrics = evt_engine.calculate_risk_metrics(evt_params)

        dcmp_result = dcmp_solver.solve_dcmp(residuals, alpha=0.05, use_conditional=True)
        risk_gap = dcmp_result.wc_cvar - evt_metrics['ES_0.99']

        tail_data[ticker] = {'gamma': gamma, 'residuals': residuals}

        scenarios = evt_engine.generate_scenarios(
            n_simulations=n_sims,
            gamma=gamma,
            volatility=cand['Pred_Vol'],
            expected_return=cand['Pred_Return']
        )
        scenarios_list.append(scenarios)
        final_tickers.append(ticker)
        expected_returns_vec.append(cand['Pred_Return'])

        candidates_report_data.append({
            'Ticker': ticker,
            'Mu': cand['Pred_Return'],
            'Sigma': cand['Pred_Vol'],
            'Gamma': gamma,
            'ES': evt_metrics['ES_0.99'],
            'WC_ES': dcmp_result.wc_cvar,
            'Gap': risk_gap
        })

        bounds_report_data.append({
            'Ticker': ticker, 'EVT_CVaR': evt_metrics['ES_0.99'],
            'DMP_CVaR': dcmp_result.wc_cvar, 'Gamma': gamma
        })

        print(f"   - {ticker}: Gamma={gamma:.2f} | Gap={risk_gap:.2f}")

    if hasattr(reporter, 'plot_risk_bounds_comparison'):
        reporter.plot_risk_bounds_comparison(bounds_report_data)

    print("\n4. Optimization: Mean-CVaR (Long/Short)...")
    portfolio_report_data = []
    opt_metrics = {}

    if len(scenarios_list) > 1:
        scenarios_matrix = np.column_stack(scenarios_list)
        expected_returns_vec = np.array(expected_returns_vec)
        optimizer = MeanCVaROptimizer(confidence_level=risk_conf)

        # Target Return Logic for Long/Short
        # We can target positive returns even if components are negative (via shorting)
        # Set target to top quartile of absolute predicted returns
        abs_returns = np.abs(expected_returns_vec)
        target_ret = np.percentile(abs_returns, 75) * 0.5  # Conservative target

        result = optimizer.optimize(expected_returns_vec, scenarios_matrix, target_ret)

        if result:
            final_weights = result['weights']
            opt_metrics['Target_Ret'] = target_ret
            opt_metrics['CVaR'] = result['CVaR_Optimal']

            allocations = sorted(zip(final_tickers, final_weights), key=lambda x: abs(x[1]), reverse=True)
            for ticker, weight in allocations:
                if abs(weight) > 0.001:
                    portfolio_report_data.append({
                        'Ticker': ticker, 'Weight': weight, 'Gamma': tail_data[ticker]['gamma']
                    })

            gammas = [tail_data[t]['gamma'] for t in final_tickers]
            reporter.plot_allocation_vs_risk(final_tickers, final_weights, gammas)
        else:
            print("Optimizer failed.")

    # --- Phase 4: Evaluation ---
    print("\n[Phase 4] Out-of-Sample Evaluation...")
    test_loader = dataloaders['test']
    model.eval()
    all_mus, all_targets = [], []
    with torch.no_grad():
        for feat, targ in test_loader:
            feat = feat.to(model_config.device)
            mu, _ = model(feat)
            all_mus.append(mu.cpu().numpy())
            all_targets.append(targ.cpu().numpy().flatten())

    all_mus = np.concatenate(all_mus)
    all_targets = np.concatenate(all_targets)
    r2 = reporter.generate_diagnostics(all_mus, all_targets)

    # Strategy Simulation (Long/Short)
    # If Mu > 0: Long (1 * Return)
    # If Mu < 0: Short (-1 * Return)
    # Note: Shorting adds volatility drag/cost, simplified here.
    strategy_returns = np.sign(all_mus) * all_targets

    cum_market = np.cumsum(all_targets)
    cum_strategy = np.cumsum(strategy_returns)

    test_metrics = {'R2': r2, 'Market_Cum': cum_market[-1], 'Strategy_Cum': cum_strategy[-1]}

    # --- Phase 5: Portfolio Backtest (The Pivot) ---
    print("\n[Phase 5] Running Out-of-Sample Portfolio Backtest...")

    # 1. Load Test Data directly (for proper Ticker alignment)
    test_file = Path(config_dict['paths']['data']) / 'test.parquet'
    test_df = pd.read_parquet(test_file)

    # 2. Pivot to get Returns Matrix (Date x Ticker)
    # We use 'Log_Return' which is the target we engineered
    returns_matrix = test_df.pivot(index='Date', columns='Ticker', values='Log_Return')

    # --- POLISH: Rescale Z-Scores to Percentage Returns ---
    # The data is currently in Z-scores (Std ~ 1.0).
    # To generate realistic report metrics (e.g., Volatility ~20% instead of ~800%),
    # we approximate the original scale by multiplying by a typical daily volatility (2%).
    returns_matrix = returns_matrix * 0.02
    # ------------------------------------------------------

    # 3. Align Weights with Test Data
    # final_tickers and final_weights come from Phase 4
    if 'final_weights' in locals() and len(final_weights) > 0:
        # Create a Series for easy mapping
        weight_map = pd.Series(final_weights, index=final_tickers)

        # Filter returns matrix to only include our portfolio assets
        portfolio_assets = [t for t in final_tickers if t in returns_matrix.columns]
        aligned_returns = returns_matrix[portfolio_assets].fillna(0.0)

        # Re-normalize weights if any assets were missing in test data
        aligned_weights = weight_map[portfolio_assets]
        aligned_weights = aligned_weights / aligned_weights.sum()

        # 4. Calculate Portfolio Performance
        # Daily Portfolio Return = Sum(Weight_i * Return_i)
        portfolio_daily_rets = aligned_returns.dot(aligned_weights)

        # 5. Calculate Benchmarks
        # Equal Weight Index (The "Market" of these assets)
        equal_weights = np.ones(len(portfolio_assets)) / len(portfolio_assets)
        market_daily_rets = aligned_returns.dot(equal_weights)

        # 6. Cumulative Returns
        portfolio_cum = np.cumsum(portfolio_daily_rets)
        market_cum = np.cumsum(market_daily_rets)

        # 7. Metrics
        total_port_ret = portfolio_cum.iloc[-1] if len(portfolio_cum) > 0 else 0.0
        total_mkt_ret = market_cum.iloc[-1] if len(market_cum) > 0 else 0.0

        # Volatility (Annualized)
        port_vol = portfolio_daily_rets.std() * np.sqrt(252)
        mkt_vol = market_daily_rets.std() * np.sqrt(252)

        # Sharpe Ratio (assuming 0 risk-free for simplicity)
        port_sharpe = (portfolio_daily_rets.mean() * 252) / (port_vol + 1e-6)
        mkt_sharpe = (market_daily_rets.mean() * 252) / (mkt_vol + 1e-6)

        print("\n=== OUT-OF-SAMPLE RESULTS (2024+) ===")
        print(f"Strategy:    Tail-Risk Parity (Min CVaR)")
        print(f"Benchmark:   Equal-Weight Index")
        print("-" * 45)
        print(f"{'Metric':<15} {'Strategy':<12} {'Benchmark':<12}")
        print("-" * 45)
        print(f"{'Total Return':<15} {total_port_ret * 100:<10.2f}%  {total_mkt_ret * 100:<10.2f}%")
        print(f"{'Volatility':<15} {port_vol * 100:<10.2f}%  {mkt_vol * 100:<10.2f}%")
        print(f"{'Sharpe Ratio':<15} {port_sharpe:<10.4f}  {mkt_sharpe:<10.4f}")
        print("-" * 45)

        # Update Report Dictionary
        test_metrics = {
            'Strategy_Return': total_port_ret,
            'Market_Return': total_mkt_ret,
            'Strategy_Sharpe': port_sharpe,
            'Market_Sharpe': mkt_sharpe
        }

        # Generate Comparison Plot
        if hasattr(reporter, 'plot_backtest_comparison'):
            reporter.plot_backtest_comparison(portfolio_cum, market_cum)

    else:
        print("[Warning] No portfolio weights found. Skipping backtest.")
        test_metrics = {}

    # Save Final Report
    reporter.save_comprehensive_report(
        candidates=candidates_report_data,
        portfolio=portfolio_report_data,
        opt_metrics=opt_metrics,
        test_metrics=test_metrics,
        bounds_data=bounds_report_data
    )
    print(f"\nPipeline Complete. Reports generated in {reporter.output_dir}")


if __name__ == "__main__":
    run_project_pipeline()