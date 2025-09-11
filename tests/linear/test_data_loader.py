"""Tests for linear data loader."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from torch.utils.data import DataLoader

from scxpand.data_util.dataset import CellsDataset
from scxpand.linear.linear_trainer import LinearBatchPredictor


class TestLinearBatchPredictor:
    """Test the LinearBatchPredictor class."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock CellsDataset."""
        dataset = MagicMock(spec=CellsDataset)
        dataset.y = MagicMock()
        dataset.y.numpy = MagicMock(return_value=np.array([0, 1, 0, 1, 1]))
        dataset.obs_df = pd.DataFrame({"expansion": [0, 1, 0, 1, 1]})
        return dataset

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock DataLoader."""
        dataloader = MagicMock(spec=DataLoader)
        # Mock the iteration to return some sample data
        dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "x": MagicMock(numpy=MagicMock(return_value=np.random.rand(10, 5))),
                        "y": MagicMock(numpy=MagicMock(return_value=np.random.randint(0, 2, 10))),
                    },
                    {
                        "x": MagicMock(numpy=MagicMock(return_value=np.random.rand(8, 5))),
                        "y": MagicMock(numpy=MagicMock(return_value=np.random.randint(0, 2, 8))),
                    },
                ]
            )
        )
        return dataloader

    def test_predict_batch_logistic(self, mock_dataset, mock_dataloader):
        """Test batch prediction for logistic regression."""
        predictor = LinearBatchPredictor(mock_dataset, mock_dataloader)

        # Create a mock logistic model
        model = MagicMock()
        model.loss = "log_loss"

        # Mock predict_proba to return proper shape
        predict_proba_return = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        model.predict_proba = MagicMock(return_value=predict_proba_return)

        X_batch = np.random.rand(3, 5)

        probs = predictor.predict_batch(model, X_batch)

        assert len(probs) == 3
        assert all(isinstance(p, int | float | np.floating) for p in probs)
        model.predict_proba.assert_called_once_with(X_batch)

    def test_predict_batch_svm(self, mock_dataset, mock_dataloader):
        """Test batch prediction for SVM using decision function."""
        predictor = LinearBatchPredictor(mock_dataset, mock_dataloader)

        # Create a mock SVM model
        model = MagicMock()
        model.loss = "hinge"

        # Mock decision function to return 1D array
        decision_values = np.array([1.5, -0.5, 0.8])
        model.decision_function = MagicMock(return_value=decision_values)

        # Remove predict_proba attribute to simulate SVM
        if hasattr(model, "predict_proba"):
            delattr(model, "predict_proba")

        X_batch = np.random.rand(3, 5)

        probs = predictor.predict_batch(model, X_batch)

        assert len(probs) == 3
        assert all(0 <= p <= 1 for p in probs)
        model.decision_function.assert_called_once_with(X_batch)

    def test_predict_batch_multiclass_decision_function(self, mock_dataset, mock_dataloader):
        """Test batch prediction for multiclass using decision function."""
        predictor = LinearBatchPredictor(mock_dataset, mock_dataloader)

        # Create a mock model with multiclass decision function
        model = MagicMock()
        model.loss = "hinge"

        # Mock decision function to return 2D array (multiclass)
        decision_values = np.array([[1.5, -0.5], [0.8, 0.2], [-0.3, 1.1]])
        model.decision_function = MagicMock(return_value=decision_values)

        # Remove predict_proba attribute to simulate SVM
        if hasattr(model, "predict_proba"):
            delattr(model, "predict_proba")

        X_batch = np.random.rand(3, 5)

        probs = predictor.predict_batch(model, X_batch)

        assert len(probs) == 3
        assert all(0 <= p <= 1 for p in probs)
        model.decision_function.assert_called_once_with(X_batch)

    def test_predict_all(self, mock_dataset, mock_dataloader):
        """Test predicting on all samples."""
        predictor = LinearBatchPredictor(mock_dataset, mock_dataloader)

        # Create a mock model
        model = MagicMock()

        def mock_predict_proba(X):
            # Return probabilities for the batch size
            n_samples = X.shape[0]
            return np.column_stack([np.random.rand(n_samples), np.random.rand(n_samples)])

        model.predict_proba = MagicMock(side_effect=mock_predict_proba)

        all_probs = predictor.predict_all(model)

        # Should have predictions for all samples from both batches (10 + 8 = 18)
        assert len(all_probs) == 18
        assert all(isinstance(p, int | float | np.floating) for p in all_probs)
        assert model.predict_proba.call_count == 2  # Called for each batch
