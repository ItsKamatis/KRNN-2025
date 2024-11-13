# tests/test_krnn_v5.py
import pytest
import torch
import numpy as np
from pathlib import Path

from src.model.krnn_v5 import KRNN, ModelConfig, KRNNPredictor


@pytest.fixture
def model_config():
    """Basic model configuration for testing."""
    return ModelConfig(
        feature_dim=10,
        hidden_dim=32,
        num_classes=3,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        device='cpu'  # Use CPU for testing
    )


@pytest.fixture
def model(model_config):
    """Initialize model for testing."""
    return KRNN(model_config)


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    batch_size, seq_len = 4, 10
    features = torch.randn(batch_size, seq_len, model_config().feature_dim)
    targets = torch.randint(0, 3, (batch_size,))
    return features, targets


class TestKRNN:
    """Test KRNN model functionality."""

    @staticmethod
    def test_model_initialization(model):
        """Test model creates with correct architecture."""
        assert isinstance(model, KRNN)
        assert model.rnn.input_size == model.config.hidden_dim
        assert model.rnn.hidden_size == model.config.hidden_dim
        assert model.rnn.num_layers == model.config.num_layers
        assert model.rnn.bidirectional == model.config.bidirectional

    @staticmethod
    def test_forward_pass(model, sample_batch):
        """Test forward pass returns correct shapes."""
        features, _ = sample_batch
        logits, attention = model(features)

        # Check output shapes
        batch_size = features.size(0)
        assert logits.size() == (batch_size, model.config.num_classes)
        assert attention.size() == (batch_size, features.size(1), features.size(1))

        # Check output values
        assert not torch.isnan(logits).any()
        assert not torch.isnan(attention).any()
        assert (attention >= 0).all() and (attention <= 1).all()

    @staticmethod
    def test_prediction(model, sample_batch):
        """Test prediction generates valid probabilities."""
        features, _ = sample_batch
        with torch.no_grad():
            predictions = model.predict(features)

        # Check probability properties
        assert predictions.size() == (features.size(0), model.config.num_classes)
        assert torch.allclose(predictions.sum(dim=1), torch.ones(features.size(0)))
        assert (predictions >= 0).all() and (predictions <= 1).all()

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_sizes(self, model, batch_size):
        """Test model handles different batch sizes."""
        seq_len = 10
        features = torch.randn(batch_size, seq_len, model.config.feature_dim)
        logits, attention = model(features)
        assert logits.size(0) == batch_size

    @staticmethod
    def test_save_load(model, tmp_path):
        """Test model save and load functionality."""
        # Save model
        save_path = tmp_path / "model.pt"
        torch.save({
            'model_state': model.state_dict(),
            'config': model.config
        }, save_path)

        # Load model
        checkpoint = torch.load(save_path)
        new_model = KRNN(checkpoint['config'])
        new_model.load_state_dict(checkpoint['model_state'])

        # Compare outputs
        with torch.no_grad():
            features = torch.randn(1, 10, model.config.feature_dim)
            original_out = model(features)[0]
            loaded_out = new_model(features)[0]
            assert torch.allclose(original_out, loaded_out)


class TestKRNNPredictor:
    """Test KRNNPredictor functionality."""

    @pytest.fixture
    def predictor(self, model):
        return KRNNPredictor(model)

    @staticmethod
    def test_train_step(predictor, sample_batch):
        """Test training step."""
        features, targets = sample_batch
        loss, logits = predictor.train_step(features, targets)

        assert isinstance(loss, float)
        assert logits.size() == (features.size(0), predictor.model.config.num_classes)
        assert not torch.isnan(torch.tensor(loss))

    @staticmethod
    def test_validation(predictor, sample_batch):
        """Test validation step."""
        features, targets = sample_batch
        with torch.no_grad():
            loss, logits = predictor.validate(features, targets)

        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    @staticmethod
    def test_checkpoint_save_load(predictor, tmp_path):
        """Test checkpoint functionality."""
        save_path = tmp_path / "checkpoint.pt"
        predictor.save_checkpoint(str(save_path))

        # Create new predictor and load checkpoint
        new_predictor = KRNNPredictor(KRNN(predictor.model.config))
        new_predictor.load_checkpoint(str(save_path))

        # Verify loaded state
        assert torch.equal(
            next(predictor.model.parameters()),
            next(new_predictor.model.parameters())
        )


if __name__ == "__main__":
    pytest.main([__file__])