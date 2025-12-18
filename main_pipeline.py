import torch
import numpy as np
import pandas as pd
import yaml
import logging
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from src.data.data_collector_v5 import DataCollector
from src.data.dataset_v5 import DataModule, DataConfig, StockDataset
from src.model.krnn_v5 import KRNNRegressor, ModelConfig
from src.utils.trainer_v5 import Trainer
from src.risk.evt import EVTEngine, get_standardized_residuals
from src.portfolio.optimizer import MeanCVaROptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    collector.collect_data(start_date='2018-01-01')
    return config


def run_project_pipeline():
    # --- Step 0: Initialization ---
    print("\n[Phase 0] Initializing Pipeline & Checking Data...")
    config_dict = initialize_data_pipeline("config_v5.yaml")

    # --- Step 1: Data Loading (Training Cache) ---
    print("\n[Phase 1] Loading Data Module (RAM Cache)...")
    data_config = DataConfig(
        sequence_length=config_dict['data']['window_size'],
        batch_size=config_dict['data']['batch_size']
    )
    data_module = DataModule(data_config)
    data_module.setup(Path(config_dict['paths']['data']))
    dataloaders = data_module.get_dataloaders()

    # --- Step 2: Model Training (Probabilistic) ---
    print("\n[Phase 1] Training K-Parallel KRNN (Heteroscedastic)...")
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

    # A. Universe Scanning
    val_df_path = Path(config_dict['paths']['data']) / 'validation.parquet'
    val_df = pd.read_parquet(val_df_path)
    all_tickers = val_df['Ticker'].unique()
    print(f"1. Universe Scan: Found {len(all_tickers)} tickers.")

    # B. Alpha Filter (KRNN Prediction)
    print("2. Alpha Filter: Ranking assets by Predicted Return...")
    candidates = []

    top_n = config_dict.get('portfolio', {}).get('top_n_assets', 10)
    risk_conf = config_dict.get('portfolio', {}).get('risk_confidence', 0.95)
    n_sims = config_dict.get('portfolio', {}).get('n_simulations', 1000)

    model.eval()

    for ticker in all_tickers:
        # Create ephemeral dataset for inference
        ticker_df = val_df[val_df['Ticker'] == ticker].copy()
        if len(ticker_df) < data_config.sequence_length + 20:
            continue

        ticker_ds = StockDataset(ticker_df, data_config.sequence_length)
        ticker_loader = DataLoader(ticker_ds, batch_size=64, shuffle=False)

        # Inference
        mus = []
        sigmas = []
        targets = []
        with torch.no_grad():
            for feat, targ in ticker_loader:
                feat = feat.to(model_config.device)

                # Unpack Probabilistic Output
                mu, sigma = model(feat)

                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
                targets.append(targ.cpu().numpy())

        mus = np.concatenate(mus)
        sigmas = np.concatenate(sigmas)
        targets = np.concatenate(targets)

        # Metrics
        # We use the average Predicted Return (Alpha)
        avg_pred_return = np.mean(mus)

        # We also store the average Predicted Volatility (for the Scenario Generator)
        avg_pred_volatility = np.mean(sigmas)

        # Calculate STANDARDIZED Residuals for EVT
        # Z = (Target - Mu) / Sigma
        residuals = (targets.flatten() - mus.flatten()) / sigmas.flatten()

        candidates.append({
            'Ticker': ticker,
            'Pred_Return': avg_pred_return,
            'Pred_Vol': avg_pred_volatility,
            'Residuals': residuals  # These are now Z-scores
        })

    # Sort and Select Top N
    candidates.sort(key=lambda x: x['Pred_Return'], reverse=True)
    top_candidates = candidates[:top_n]

    print(f"   Selected Top {len(top_candidates)} Candidates:")
    for c in top_candidates:
        print(f"   - {c['Ticker']}: Mu={c['Pred_Return'] * 100:.4f}% | Sigma={c['Pred_Vol'] * 100:.4f}%")

    # C. Risk Filter (EVT Analysis)
    print(f"\n3. Risk Filter: Analyzing Heavy Tails (Gamma of Z-scores)...")

    scenarios_list = []
    final_tickers = []
    expected_returns_vec = []
    evt_engine = EVTEngine(tail_fraction=0.10)

    for cand in top_candidates:
        ticker = cand['Ticker']

        # EVT Analysis of Standardized Residuals
        evt_params = evt_engine.analyze_tails(cand['Residuals'])
        gamma = evt_params['gamma']

        # Generate Scenarios (Future Path)
        # We use the Model's Predicted Volatility + EVT Tail Shocks
        scenarios = evt_engine.generate_scenarios(
            n_simulations=n_sims,
            gamma=gamma,
            volatility=cand['Pred_Vol'],  # <--- NEW: Using KRNN Sigma
            expected_return=cand['Pred_Return']
        )

        scenarios_list.append(scenarios)
        final_tickers.append(ticker)
        expected_returns_vec.append(cand['Pred_Return'])

        # Metrics
        metrics = evt_engine.calculate_risk_metrics(evt_params)
        print(f"   - {ticker}: Gamma={gamma:.4f} | 99% ES (Z)={metrics['ES_0.99']:.4f}")

    # D. Optimization
    print("\n4. Optimization: Minimizing Portfolio CVaR...")

    if len(scenarios_list) > 1:
        scenarios_matrix = np.column_stack(scenarios_list)
        expected_returns_vec = np.array(expected_returns_vec)

        optimizer = MeanCVaROptimizer(confidence_level=risk_conf)

        # Target
        target_factor = config_dict.get('portfolio', {}).get('target_return_factor', 0.8)
        target_ret = np.mean(expected_returns_vec) * target_factor

        result = optimizer.optimize(expected_returns_vec, scenarios_matrix, target_ret)

        if result:
            print("\n=== FINAL HETEROSCEDASTIC PORTFOLIO ===")
            print(f"Objective: Minimize CVaR (95%) while earning > {target_ret * 100:.4f}% daily")
            print(f"Resulting Portfolio CVaR: {result['CVaR_Optimal']:.4f}")
            print("-" * 40)

            allocations = sorted(zip(final_tickers, result['weights']), key=lambda x: x[1], reverse=True)
            for ticker, weight in allocations:
                if weight > 0.001:
                    print(f"{ticker:<6}: {weight * 100:.2f}%")
            print("-" * 40)
        else:
            print("Optimizer failed to find a feasible solution.")
    else:
        print("Not enough candidates for optimization.")


if __name__ == "__main__":
    run_project_pipeline()