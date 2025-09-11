"""Tests for MLP trainer early stopping functionality."""

from unittest.mock import MagicMock

from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.train_util import check_early_stopping


class TestMLPTrainerEarlyStopping:
    """Test early stopping functionality in MLP trainer."""

    def test_early_stopping_check_function_improving_scores(self):
        """Test that check_early_stopping function works correctly with improving scores."""
        # Mock logger with improving scores
        mock_logger = MagicMock()
        mock_logger.best_model_score = None  # No previous best score

        # First call - should reset patience
        patience_counter, should_stop = check_early_stopping(
            current_score=0.8,
            log_manager=mock_logger,
            patience_counter=0,
            patience_limit=3,
            epoch=0,
        )

        assert patience_counter == 0
        assert should_stop is False

        # Second call with improvement - should still reset patience
        mock_logger.best_model_score = 0.8
        patience_counter, should_stop = check_early_stopping(
            current_score=0.9,  # Improvement
            log_manager=mock_logger,
            patience_counter=patience_counter,
            patience_limit=3,
            epoch=1,
        )

        assert patience_counter == 0
        assert should_stop is False

    def test_early_stopping_check_function_no_improvement(self):
        """Test that check_early_stopping function works correctly with no improvement."""
        # Mock logger with a previous best score
        mock_logger = MagicMock()
        mock_logger.best_model_score = 0.9

        # First call with no improvement - should increment patience
        patience_counter, should_stop = check_early_stopping(
            current_score=0.8,  # No improvement
            log_manager=mock_logger,
            patience_counter=0,
            patience_limit=3,
            epoch=0,
        )

        assert patience_counter == 1
        assert should_stop is False

        # Second call with no improvement - should increment patience
        patience_counter, should_stop = check_early_stopping(
            current_score=0.7,  # Still no improvement
            log_manager=mock_logger,
            patience_counter=patience_counter,
            patience_limit=3,
            epoch=1,
        )

        assert patience_counter == 2
        assert should_stop is False

        # Third call with no improvement - should trigger early stopping
        patience_counter, should_stop = check_early_stopping(
            current_score=0.6,  # Still no improvement
            log_manager=mock_logger,
            patience_counter=patience_counter,
            patience_limit=3,
            epoch=2,
        )

        assert patience_counter == 3
        assert should_stop is True

    def test_early_stopping_check_function_mixed_scores(self):
        """Test early stopping with mixed scores (improvement then decline)."""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger.best_model_score = 0.5

        # First call with improvement - should reset patience
        patience_counter, should_stop = check_early_stopping(
            current_score=0.8,  # Improvement
            log_manager=mock_logger,
            patience_counter=2,  # Previous patience
            patience_limit=3,
            epoch=0,
        )

        assert patience_counter == 0  # Reset due to improvement
        assert should_stop is False

        # Update best score for next test
        mock_logger.best_model_score = 0.8

        # Second call with no improvement - should increment patience
        patience_counter, should_stop = check_early_stopping(
            current_score=0.7,  # No improvement
            log_manager=mock_logger,
            patience_counter=patience_counter,
            patience_limit=3,
            epoch=1,
        )

        assert patience_counter == 1
        assert should_stop is False

    def test_mlp_param_early_stopping_config(self):
        """Test that MLPParam correctly configures early stopping parameters."""
        # Test default early stopping patience
        prm = MLPParam()
        assert hasattr(prm, "early_stopping_patience")
        assert isinstance(prm.early_stopping_patience, int)
        assert prm.early_stopping_patience > 0

        # Test custom early stopping patience
        custom_patience = 10
        prm_custom = MLPParam(early_stopping_patience=custom_patience)
        assert prm_custom.early_stopping_patience == custom_patience

    def test_mlp_param_training_config(self):
        """Test that MLPParam correctly configures training parameters."""
        # Test default training parameters
        prm = MLPParam()
        assert hasattr(prm, "n_epochs")
        assert isinstance(prm.n_epochs, int)
        assert prm.n_epochs > 0

        # Test custom training parameters
        custom_epochs = 20
        prm_custom = MLPParam(n_epochs=custom_epochs)
        assert prm_custom.n_epochs == custom_epochs
