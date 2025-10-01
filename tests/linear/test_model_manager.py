"""Tests for linear model manager."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.model_manager import ModelManager
from scxpand.util.model_constants import BEST_MODEL_INFO_FILE
from tests.test_utils import safe_context_manager


class TestModelManager:
    """Test the ModelManager class."""

    def test_initialize_logistic_model(self):
        """Test initialization of logistic regression model."""
        prm = LinearClassifierParam(model_type="logistic")
        y_train = np.array([0, 1, 0, 1, 1])

        model = ModelManager.initialize_model(prm=prm, y_train=y_train)

        assert isinstance(model, SGDClassifier)
        assert model.loss == "log_loss"
        assert model.penalty == "l2"
        assert model.alpha == 0.0001  # scikit-learn SGDClassifier default
        assert model.max_iter == 1
        assert model.warm_start is False  # scikit-learn SGDClassifier default

    def test_initialize_svm_model(self):
        """Test initialization of SVM model."""
        prm = LinearClassifierParam(model_type="svm")
        y_train = np.array([0, 1, 0, 1, 1])

        model = ModelManager.initialize_model(prm=prm, y_train=y_train)

        assert isinstance(model, SGDClassifier)
        assert model.loss == "hinge"
        assert model.penalty == "l2"

    def test_balanced_class_weight(self):
        """Test balanced class weight computation."""
        prm = LinearClassifierParam(class_weight="balanced")
        y_train = np.array([0, 0, 0, 1, 1])  # Imbalanced: 3 zeros, 2 ones

        model = ModelManager.initialize_model(prm=prm, y_train=y_train)

        # Should have computed balanced weights
        assert model.class_weight is not None

    def test_none_class_weight(self):
        """Test None class weight handling."""
        prm = LinearClassifierParam(class_weight="None")
        y_train = np.array([0, 1, 0, 1, 1])

        model = ModelManager.initialize_model(prm=prm, y_train=y_train)

        assert model.class_weight is None

    @pytest.fixture
    def mock_model(self):
        """Create a mock SGDClassifier with state."""
        model = MagicMock()
        model.coef_ = np.array([[1.0, 2.0, 3.0]])
        model.intercept_ = np.array([0.5])
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 3
        model.t_ = 100
        return model

    def test_save_model_state(self, mock_model):
        """Test saving model state."""
        state = ModelManager.save_model_state(
            model=mock_model,
            current_score=0.85,
            epoch=15,
            dev_set_metrics={"AUROC": 0.8, "F1": 0.7},
        )

        assert "coef_" in state
        assert "intercept_" in state
        assert "classes_" in state
        assert "n_features_in_" in state
        assert "t_" in state
        assert state["epoch"] == 15
        assert state["score"] == 0.85
        assert "dev_set_metrics" in state

    def test_load_model_state(self, mock_model):
        """Test loading model state."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)

            best_model_state = {
                "coef_": np.array([[4.0, 5.0, 6.0]]),
                "intercept_": np.array([1.0]),
                "classes_": np.array([0, 1]),
                "n_features_in_": 3,
                "t_": 200,
                "epoch": 20,
                "score": 0.9,
            }

            ModelManager.load_model_state(
                model=mock_model,
                best_model_state=best_model_state,
                base_save_dir=base_save_dir,
                score_metric="AUROC",
            )

            # Check that model attributes were updated
            np.testing.assert_array_equal(mock_model.coef_, best_model_state["coef_"])
            np.testing.assert_array_equal(
                mock_model.intercept_, best_model_state["intercept_"]
            )
            np.testing.assert_array_equal(
                mock_model.classes_, best_model_state["classes_"]
            )
            assert mock_model.n_features_in_ == best_model_state["n_features_in_"]
            assert mock_model.t_ == best_model_state["t_"]

            # Check that best model info file was created
            assert (base_save_dir / BEST_MODEL_INFO_FILE).exists()
