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


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
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
    collector.collect_data(start_date=config['data']['train_start'])
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
    print("\n[Phase 2] Training K-Parallel KRNN (Heteroscedastic)...")
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

        # Calculate residuals (Z-scores)
        residuals = (targets.flatten() - mus.flatten()) / (sigmas.flatten() + 1e-6)  # Avoid div/0

        candidates.append({
            'Ticker': ticker,
            'Pred_Return': avg_pred_return,
            'Pred_Vol': avg_pred_volatility,
            'Residuals': residuals
        })

    candidates.sort(key=lambda x: x['Pred_Return'], reverse=True)
    top_candidates = candidates[:top_n]
    print(f"   Selected Top {len(top_candidates)} Candidates.")

    # --- 3b. Robust Risk Analysis (EVT + DCMP) ---
    print(f"\n3. Risk Filter: Analyzing Heavy Tails (EVT) & Bounds (Naumova DCMP)...")

    scenarios_list = []
    final_tickers = []
    expected_returns_vec = []

    evt_engine = EVTEngine(tail_fraction=0.10)
    # Instantiate the new Naumova Solver
    dcmp_solver = DiscreteConditionalMomentSolver(n_points=500, support_range=(-10.0, 10.0))

    tail_data = {}
    candidates_report_data = []
    bounds_report_data = []  # For the "Risk Gap" chart

    for cand in top_candidates:
        ticker = cand['Ticker']
        residuals = cand['Residuals']

        # 1. EVT Analysis (Parametric Tail)
        evt_params = evt_engine.analyze_tails(residuals)
        gamma = evt_params['gamma']
        evt_metrics = evt_engine.calculate_risk_metrics(evt_params)
        evt_cvar = evt_metrics['ES_0.99']

        # 2. DCMP Analysis (Robust Bound with Conditional Constraints)
        # We pass use_conditional=True to use Naumova's method
        dcmp_result = dcmp_solver.solve_dcmp(residuals, alpha=0.05, use_conditional=True)
        wc_cvar = dcmp_result.wc_cvar

        # Calculate Risk Gap
        risk_gap = wc_cvar - evt_cvar

        tail_data[ticker] = {'gamma': gamma, 'residuals': residuals}

        # Generate Scenarios (Using EVT logic for the optimizer input)
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
            'ES': evt_cvar,
            'WC_ES': wc_cvar,  # Add for report
            'Gap': risk_gap
        })

        bounds_report_data.append({
            'Ticker': ticker,
            'EVT_CVaR': evt_cvar,
            'DMP_CVaR': wc_cvar,
            'Gamma': gamma
        })

        print(
            f"   - {ticker}: Gamma={gamma:.2f} | EVT-CVaR={evt_cvar:.2f} | DCMP-CVaR={wc_cvar:.2f} | Gap={risk_gap:.2f}")

    reporter.plot_tail_comparison(tail_data)
    # Make sure your reporter class has the plot_risk_bounds_comparison method we added in thoughts
    if hasattr(reporter, 'plot_risk_bounds_comparison'):
        reporter.plot_risk_bounds_comparison(bounds_report_data)

    print("\n4. Optimization: Minimizing Portfolio CVaR...")
    portfolio_report_data = []
    opt_metrics = {}

    if len(scenarios_list) > 1:
        scenarios_matrix = np.column_stack(scenarios_list)
        expected_returns_vec = np.array(expected_returns_vec)
        optimizer = MeanCVaROptimizer(confidence_level=risk_conf)
        # --- FIX TARGET LOGIC HERE ---
        avg_mu = np.mean(expected_returns_vec)
        # If returns are negative, don't multiply by 0.8 (which makes them 'higher' closer to 0)
        # Instead, just pass the raw average as the baseline
        target_ret = avg_mu * config_dict.get('portfolio', {}).get('target_return_factor', 0.8)

        result = optimizer.optimize(expected_returns_vec, scenarios_matrix, target_ret)

        if result:
            final_weights = result['weights']
            opt_metrics['Target_Ret'] = target_ret
            opt_metrics['CVaR'] = result['CVaR_Optimal']

            allocations = sorted(zip(final_tickers, final_weights), key=lambda x: x[1], reverse=True)
            for ticker, weight in allocations:
                if weight > 0.001:
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

    # Simple strategy simulation
    cum_market = np.cumsum(all_targets)
    cum_strategy = np.cumsum(np.where(all_mus > 0, all_targets, 0))
    test_metrics = {'R2': r2, 'Market_Cum': cum_market[-1], 'Strategy_Cum': cum_strategy[-1]}

    # Pass the new Naumova bounds data to the report
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