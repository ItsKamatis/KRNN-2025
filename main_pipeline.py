# main_pipeline.py
import torch
import numpy as np
import pandas as pd
import yaml
import logging
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from src.data.data_collector_v5 import DataCollector
from src.data.dataset_v5 import DataModule, DataConfig
from src.model.krnn_v5 import KRNNRegressor, ModelConfig
from src.model.train_v5 import Trainer
from src.risk.evt import calculate_portfolio_risk, get_residuals
from src.portfolio.optimizer import MeanCVaROptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_data_pipeline(config_path: str = "config_v5.yaml"):
    """
    Checks if data exists. If not, runs the DataCollector.
    This replaces the assumption that data is just 'there'.
    """
    # Load config to get paths
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config['paths']['data'])
    required_files = ['train.parquet', 'validation.parquet', 'test.parquet']

    # Check if all files exist
    if all((data_dir / f).exists() for f in required_files):
        logger.info("Data files found. Skipping download.")
        return config

    logger.info("Data files missing. Starting Data Collection...")

    # Initialize Collector with config
    collector = DataCollector(config)

    # Run Collection (Downloads NASDAQ-100, Calc Features, Saves Parquet)
    # Using a 5-year lookback for good EVT tail estimation
    collector.collect_data(start_date='2018-01-01')

    return config


def run_project_pipeline():
    # --- Step 0: Initialization & Configuration ---
    print("\n[Phase 0] Initializing Pipeline & Checking Data...")
    # This ensures data exists before we crash
    config_dict = initialize_data_pipeline("config_v5.yaml")

    # --- Step 1: Data Loading ---
    print("\n[Phase 1] Loading Data Module...")
    # FIX: changed 'window_size' to 'sequence_length' to match DataConfig definition
    data_config = DataConfig(
        sequence_length=config_dict['data']['window_size'],
        batch_size=config_dict['data']['batch_size']
    )

    data_module = DataModule(data_config)
    data_module.setup(Path(config_dict['paths']['data']))
    dataloaders = data_module.get_dataloaders()

    # --- Step 2: Model Training (The "KRNN" Regression) ---
    print("\n[Phase 1] Training K-Parallel KRNN Regressor...")
    feature_dim = len(data_module.train_dataset.feature_cols)

    model_config = ModelConfig(
        input_dim=feature_dim,
        hidden_dim=config_dict['model']['hidden_dim'],
        k_dups=3,  # The "K" in KRNN (Parallel Ensembles)
        output_dim=1,  # Regression (Predicting Log Returns)
        dropout=config_dict['model']['dropout'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = KRNNRegressor(model_config)

    # Train
    # We pass the full config dict for hyperparameters
    trainer = Trainer(model, config=config_dict)

    # Train the model (Assuming trainer.train handles the loop)
    # Note: Ensure src/model/train_v5.py Trainer class is updated for Regression (MSELoss)
    training_metrics = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        experiment=None  # logic for experiment tracking if needed
    )
    print(f"Training Complete. Final Val Loss: {training_metrics.get('loss', 'N/A')}")

    # --- Step 3: Risk Analysis (The "EVT" Engine) ---
    print("\n[Phase 2] Analyzing Residuals & Heavy Tails (EVT)...")

    # In a real scenario, we loop through tickers.
    # Here we show the logic using the model's actual validation performance.

    # 1. Get Real Residuals from the trained model
    # (residuals = y_true - y_pred)
    val_residuals = get_residuals(model, dataloaders['val'], model_config.device)

    # 2. Analyze Tails
    # We use these residuals to estimate the 'Gamma' (Tail Index)
    # A generic analysis for the whole market (simplified)
    from src.risk.evt import EVTEngine
    evt_engine = EVTEngine(tail_fraction=0.10)
    evt_params = evt_engine.analyze_tails(val_residuals)
    risk_metrics = evt_engine.calculate_risk_metrics(evt_params)

    print(f"Market Tail Index (Gamma): {evt_params['gamma']:.4f}")
    print(f"99% VaR (EVT): {risk_metrics['VaR_0.99']:.5f}")
    print(f"99% ES (EVT):  {risk_metrics['ES_0.99']:.5f}")

    # --- Step 4: Portfolio Optimization (Mean-CVaR) ---
    print("\n[Phase 3] Optimizing Portfolio Weights (Mean-CVaR)...")

    # Demonstration: Using 3 hypothetical assets derived from our model's statistics
    # In production, you would run the model 3 times for 3 diff tickers.
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Simulate scenarios based on the heavy tails we just found
    # We generate random returns using the actual Gamma we found (Student-t approx)
    df_params = 1.0 / evt_params['gamma']  # Gamma approx inverse of DOF

    scenarios_list = []
    expected_returns = []

    for _ in tickers:
        # Predict: Model expects +0.05% return
        pred_ret = 0.0005

        # Risk: Heavy tailed noise based on our EVT findings
        noise = np.random.standard_t(df=df_params, size=250) * np.std(val_residuals)

        scenarios_list.append(pred_ret + noise)
        expected_returns.append(pred_ret)

    scenarios_matrix = np.column_stack(scenarios_list)
    expected_returns_vec = np.array(expected_returns)

    optimizer = MeanCVaROptimizer(confidence_level=0.95)
    target_ret = 0.0003  # Target: 0.03% daily return

    result = optimizer.optimize(expected_returns_vec, scenarios_matrix, target_ret)

    if result:
        print("\n=== OPTIMAL PORTFOLIO ALLOCATION ===")
        print(f"Target Return: {target_ret * 100:.2f}%")
        print(f"Min CVaR (95%): {result['CVaR_Optimal']:.4f}")
        print("-" * 30)
        for ticker, weight in zip(tickers, result['weights']):
            print(f"{ticker}: {weight * 100:.2f}%")
        print("-" * 30)
    else:
        print("Optimization failed to find a solution.")


if __name__ == "__main__":
    run_project_pipeline()