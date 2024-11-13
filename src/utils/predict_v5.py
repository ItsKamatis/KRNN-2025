# predict_v5.py
import torch
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
import yaml

from src.model.krnn_v5 import KRNN, ModelConfig
from src.data.features_v5 import FeatureEngineer, FeatureConfig
from src.utils.experiment_v5 import MemoryOptimizer

logger = logging.getLogger(__name__)


class Predictor:
    """Efficient stock movement predictor."""

    def __init__(self, model_path: str, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.feature_engineer = FeatureEngineer(FeatureConfig())

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration file."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _load_model(self, model_path: str) -> KRNN:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = KRNN(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for stock data."""
        with torch.no_grad(), MemoryOptimizer():
            # Calculate features
            features = self.feature_engineer.calculate_features(data)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features.values).unsqueeze(0)
            features_tensor = features_tensor.to(self.device)

            # Get predictions
            probabilities = self.model.predict(features_tensor)
            predictions = probabilities.argmax(dim=1).cpu().numpy()

            # Create results DataFrame
            results = pd.DataFrame({
                'Date': data['Date'],
                'Predicted_Movement': predictions + 1,  # Adjust to 1-based indexing
                'Confidence': probabilities.max(dim=1)[0].cpu().numpy()
            })

            return results

    def predict_batch(self, data_path: Union[str, Path], output_path: Optional[str] = None) -> pd.DataFrame:
        """Process multiple stock files in a directory."""
        data_path = Path(data_path)
        all_predictions = []

        for file in data_path.glob('*.csv'):
            try:
                # Load and predict
                data = pd.read_csv(file)
                predictions = self.predict(data)
                predictions['Ticker'] = file.stem
                all_predictions.append(predictions)

            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")
                continue

        # Combine results
        results = pd.concat(all_predictions, ignore_index=True)

        # Save if output path provided
        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")

        return results


def main():
    """CLI interface for predictions."""
    import argparse
    parser = argparse.ArgumentParser(description='Stock Movement Prediction')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--input', required=True, help='Path to input data file or directory')
    parser.add_argument('--output', help='Path to save predictions')
    args = parser.parse_args()

    # Initialize predictor
    predictor = Predictor(args.model, args.config)

    # Process data
    input_path = Path(args.input)
    if input_path.is_file():
        data = pd.read_csv(input_path)
        predictions = predictor.predict(data)
    else:
        predictions = predictor.predict_batch(input_path, args.output)

    if args.output and input_path.is_file():
        predictions.to_csv(args.output, index=False)
        logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()