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
from src.portfolio.optimizer import MeanCVaROptimizer
from src.utils.generate_report import ReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Ensures reproducibility across the entire pipeline."""
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
        output_dim=2,  # Mu and Sigma
        dropout=config_dict['model']['dropout'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = KRNNRegressor(model_config)
    trainer = Trainer(model, config=config_dict)

    # Train
    training_metrics = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val']
    )
    print(f"Training Complete. Final Val GNLL: {training_metrics.get('loss', 'N/A')}")

    # --- Step 3: The Systematic Funnel ---
    print("\n[Phase 3] Running Systematic Portfolio Selection...")

    val_df_path = Path(config_dict['paths']['data']) / 'validation.parquet'
    val_df = pd.read_parquet(val_df_path)
    all_tickers = val_df['Ticker'].unique()
    print(f"1. Universe Scan: Found {len(all_tickers)} tickers.")

    print("2. Alpha Filter: Ranking assets by Predicted Return...")
    candidates = []

    # Configs
    top_n = config_dict.get('portfolio', {}).get('top_n_assets', 10)
    risk_conf = config_dict.get('portfolio', {}).get('risk_confidence', 0.95)
    n_sims = config_dict.get('portfolio', {}).get('n_simulations', 1000)

    model.eval()

    all_regime_mus = []
    all_regime_sigmas = []

    for ticker in all_tickers:
        ticker_df = val_df[val_df['Ticker'] == ticker].copy()
        if len(ticker_df) < data_config.sequence_length + 20:
            continue

        ticker_ds = StockDataset(ticker_df, data_config.sequence_length)
        ticker_loader = DataLoader(ticker_ds, batch_size=64, shuffle=False)

        mus = []
        sigmas = []
        targets = []
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

        all_regime_mus.append(avg_pred_return)
        all_regime_sigmas.append(avg_pred_volatility)

        residuals = (targets.flatten() - mus.flatten()) / sigmas.flatten()

        candidates.append({
            'Ticker': ticker,
            'Pred_Return': avg_pred_return,
            'Pred_Vol': avg_pred_volatility,
            'Residuals': residuals
        })

    reporter.plot_regime_clustering(all_regime_mus, all_regime_sigmas)

    candidates.sort(key=lambda x: x['Pred_Return'], reverse=True)
    top_candidates = candidates[:top_n]

    print(f"   Selected Top {len(top_candidates)} Candidates.")

    print(f"\n3. Risk Filter: Analyzing Heavy Tails (Gamma of Z-scores)...")

    scenarios_list = []
    final_tickers = []
    expected_returns_vec = []
    evt_engine = EVTEngine(tail_fraction=0.10)

    tail_data = {}

    # Data for Report
    candidates_report_data = []

    for cand in top_candidates:
        ticker = cand['Ticker']
        evt_params = evt_engine.analyze_tails(cand['Residuals'])
        gamma = evt_params['gamma']

        tail_data[ticker] = {
            'gamma': gamma,
            'residuals': cand['Residuals']
        }

        scenarios = evt_engine.generate_scenarios(
            n_simulations=n_sims,
            gamma=gamma,
            volatility=cand['Pred_Vol'],
            expected_return=cand['Pred_Return']
        )

        scenarios_list.append(scenarios)
        final_tickers.append(ticker)
        expected_returns_vec.append(cand['Pred_Return'])

        # Calculate ES for Report
        risk_metrics = evt_engine.calculate_risk_metrics(evt_params)

        candidates_report_data.append({
            'Ticker': ticker,
            'Mu': cand['Pred_Return'],
            'Sigma': cand['Pred_Vol'],
            'Gamma': gamma,
            'ES': risk_metrics['ES_0.99']
        })

        print(f"   - {ticker}: Gamma={gamma:.4f}")

    reporter.plot_tail_comparison(tail_data)

    print("\n4. Optimization: Minimizing Portfolio CVaR...")

    portfolio_report_data = []
    opt_metrics = {}

    if len(scenarios_list) > 1:
        scenarios_matrix = np.column_stack(scenarios_list)
        expected_returns_vec = np.array(expected_returns_vec)

        optimizer = MeanCVaROptimizer(confidence_level=risk_conf)
        target_ret = np.mean(expected_returns_vec) * config_dict.get('portfolio', {}).get('target_return_factor', 0.8)

        result = optimizer.optimize(expected_returns_vec, scenarios_matrix, target_ret)

        if result:
            final_weights = result['weights']
            opt_metrics['Target_Ret'] = target_ret
            opt_metrics['CVaR'] = result['CVaR_Optimal']

            print("\n=== FINAL HETEROSCEDASTIC PORTFOLIO ===")
            allocations = sorted(zip(final_tickers, final_weights), key=lambda x: x[1], reverse=True)
            for ticker, weight in allocations:
                if weight > 0.001:
                    print(f"{ticker:<6}: {weight * 100:.2f}%")

                # Store for Report
                portfolio_report_data.append({
                    'Ticker': ticker,
                    'Weight': weight,
                    'Gamma': tail_data[ticker]['gamma']
                })

            # Use original unsorted lists for plotting alignment
            gammas = [tail_data[t]['gamma'] for t in final_tickers]
            reporter.plot_allocation_vs_risk(final_tickers, final_weights, gammas)

        else:
            print("Optimizer failed.")

    # --- Phase 4: Evaluation & Diagnostics ---
    print("\n[Phase 4] Out-of-Sample Evaluation...")

    test_loader = dataloaders['test']
    model.eval()

    all_mus = []
    all_targets = []

    with torch.no_grad():
        for feat, targ in test_loader:
            feat = feat.to(model_config.device)
            mu, sigma = model(feat)
            all_mus.append(mu.cpu().numpy())
            all_targets.append(targ.cpu().numpy().flatten())

    all_mus = np.concatenate(all_mus)
    all_targets = np.concatenate(all_targets)

    # Run Diagnostics and get R2
    r2 = reporter.generate_diagnostics(all_mus, all_targets)

    # Calculate Cumulative Returns for Report
    cum_market = np.cumsum(all_targets)
    cum_strategy = np.cumsum(np.where(all_mus > 0, all_targets, 0))  # Simple Long-Only Logic

    test_metrics = {
        'R2': r2,
        'Market_Cum': cum_market[-1],
        'Strategy_Cum': cum_strategy[-1]
    }

    # SAVE ALL TEXT DATA
    reporter.save_comprehensive_report(
        candidates=candidates_report_data,
        portfolio=portfolio_report_data,
        opt_metrics=opt_metrics,
        test_metrics=test_metrics
    )

    print(f"\nPipeline Complete. Reports generated in {reporter.output_dir}")


if __name__ == "__main__":
    run_project_pipeline()