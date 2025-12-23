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


def set_seed(seed=2304571):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global Seed set to {seed}")


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
    set_seed(42)
    print("\n[Phase 0] Initializing Pipeline & Checking Data...")
    config_dict = initialize_data_pipeline("config_v5.yaml")
    reporter = ReportGenerator(output_dir="./reports")

    # --- Step 1: Data Loading ---
    print("\n[Phase 1] Loading Data Module (RAM Cache)...")
    data_config = DataConfig(
        sequence_length=config_dict['data']['window_size'],
        batch_size=config_dict['data']['batch_size']
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

    # --- 3a. Inference & Alpha Scan ---
    for ticker in all_tickers:
        ticker_df = val_df[val_df['Ticker'] == ticker].copy()
        if len(ticker_df) < data_config.sequence_length + 20:
            continue
        ticker_ds = StockDataset(ticker_df, data_config.sequence_length)
        ticker_loader = DataLoader(ticker_ds, batch_size=64, shuffle=False)
        mus, sigmas, targets = [], [], []
        with torch.no_grad():
            for feat, targ in ticker_loader:
                feat = feat.to(model_config.device)
                mu, sigma = model(feat)
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
                targets.append(targ.cpu().numpy())
        mus = np.concatenate(mus)
        sigmas = np.concatenate(sigmas)
        targets = np.concatenate(targets)

        avg_pred_return = np.mean(mus)
        avg_pred_volatility = np.mean(sigmas)
        residuals = (targets.flatten() - mus.flatten()) / (sigmas.flatten() + 1e-6)

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
    dcmp_solver = DiscreteConditionalMomentSolver(n_points=801, support_range=(-15.0, 15.0))

    tail_data = {}
    candidates_report_data = []
    bounds_report_data = []

    for cand in top_candidates:
        ticker = cand['Ticker']
        residuals = cand['Residuals']

        evt_params = evt_engine.analyze_tails(residuals)
        gamma = evt_params['gamma']
        evt_metrics = evt_engine.calculate_risk_metrics(evt_params, confidence_level=risk_conf)

        # Use consistent confidence across EVT and DCMP
        alpha = 1.0 - float(risk_conf)

        # DCMP (tail-anchored) bound on standardized residuals Z
        dcmp_result = dcmp_solver.solve_dcmp(
        residuals,
        alpha=alpha,
        use_conditional=True
        )

        # Convert Z-loss metrics to RETURN-loss metrics: L_R = -mu + sigma * L_Z
        p = float(risk_conf)
        evt_es_z = float(evt_metrics.get(f"ES_{p}", 0.0))
        evt_var_z = float(evt_metrics.get(f"VaR_{p}", 0.0))

        wc_es_z = float(dcmp_result.wc_cvar)
        wc_var_z = float(dcmp_result.wc_var)

        mu_r = float(cand['Pred_Return'])
        sig_r = float(cand['Pred_Vol'])

        evt_cvar_ret = (-mu_r) + (sig_r * evt_es_z)
        evt_var_ret = (-mu_r) + (sig_r * evt_var_z)

        wc_cvar_ret = (-mu_r) + (sig_r * wc_es_z)
        wc_var_ret = (-mu_r) + (sig_r * wc_var_z)

        risk_gap = wc_cvar_ret - evt_cvar_ret


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
            'Mu': mu_r,
            'Sigma': sig_r,
            'Gamma': gamma,
            # Return-loss units (comparable to optimizer CVaR)
            'VaR': evt_var_ret,
            'ES': evt_cvar_ret,
            'WC_VaR': wc_var_ret,
            'WC_ES': wc_cvar_ret,
            'Gap': risk_gap,
            # Diagnostics in Z-loss units
            'VaR_Z': evt_var_z,
            'ES_Z': evt_es_z,
            'WC_VaR_Z': wc_var_z,
            'WC_ES_Z': wc_es_z,
            'Risk_Conf': float(risk_conf),
            'DMP_Type': getattr(dcmp_result, 'dmp_type', 'DCMP')
        })

        bounds_report_data.append({
            'Ticker': ticker,
            'EVT_CVaR': evt_cvar_ret,
            'DMP_CVaR': wc_cvar_ret,
            'Gamma': gamma,
            'EVT_CVaR_Z': evt_es_z,
            'DMP_CVaR_Z': wc_es_z
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

    test_metrics = {'R2': r2}

    # --- Phase 5: Portfolio Backtest (The Pivot) ---
    print("\n[Phase 5] Running Out-of-Sample Portfolio Backtest...")

    # 1. Load Test Data directly (for proper Ticker alignment)
    test_file = Path(config_dict['paths']['data']) / 'test.parquet'
    test_df = pd.read_parquet(test_file)

    # 2. Pivot to get Returns Matrix (Date x Ticker)
    # We use 'Target' which is the realized next-day log return (raw scale)
    returns_matrix = test_df.pivot(index='Date', columns='Ticker', values='Target')

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
        test_metrics.update({
            'Strategy_Return': total_port_ret,
            'Market_Return': total_mkt_ret,
            'Strategy_Sharpe': port_sharpe,
            'Market_Sharpe': mkt_sharpe,
            # keep keys used by the report summary
            'Strategy_Cum': total_port_ret,
            'Market_Cum': total_mkt_ret,
            'Strategy_Vol': port_vol,
            'Market_Vol': mkt_vol
        })

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
