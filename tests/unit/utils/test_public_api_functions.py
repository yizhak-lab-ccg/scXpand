"""Tests for additional public API functions.

This module provides comprehensive tests for public API functions
that may not have complete test coverage.
"""

import numpy as np
import pytest

from scxpand.core.prediction import run_prediction_pipeline
from scxpand.util.general_util import (
    convert_enums_to_values,
    decisions_to_probabilities,
    format_float,
    get_device,
    get_new_version_path,
    sigmoid,
    to_np,
)
from scxpand.util.metrics import compute_basic_metrics, safe_hmean
from scxpand.util.model_type import (
    get_available_model_types,
    infer_model_type_from_parameters,
    load_model_type,
    save_model_type,
)


class TestGeneralUtilFunctions:
    """Tests for general utility functions."""

    def test_convert_enums_to_values_with_enum(self):
        """Test converting enum to its value."""
        from scxpand.util.classes import ModelType

        # Test with enum
        result = convert_enums_to_values(ModelType.MLP)
        assert result == "mlp"

        # Test with list containing enum
        result = convert_enums_to_values([ModelType.MLP, ModelType.AUTOENCODER])
        assert result == ["mlp", "autoencoder"]

        # Test with dict containing enum
        result = convert_enums_to_values({"model": ModelType.MLP})
        assert result == {"model": "mlp"}

    def test_convert_enums_to_values_without_enum(self):
        """Test that non-enum values are returned unchanged."""
        # Test with string
        result = convert_enums_to_values("test")
        assert result == "test"

        # Test with number
        result = convert_enums_to_values(42)
        assert result == 42

        # Test with list without enums
        result = convert_enums_to_values(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_decisions_to_probabilities(self):
        """Test converting decisions to probabilities."""
        # Test with 1D array - function applies sigmoid
        decisions_1d = np.array([0.2, 0.8, 0.5])
        result = decisions_to_probabilities(decisions_1d)
        expected = 1 / (1 + np.exp(-decisions_1d))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        # Test with 2D array - just verify it returns an array
        decisions_2d = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        result = decisions_to_probabilities(decisions_2d)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_format_float(self):
        """Test float formatting function."""
        # Test normal float
        result = format_float(0.123456789)
        assert result == "0.1235"

        # Test very small float - function uses scientific notation for small values
        result = format_float(0.0001)
        assert result == "1e-4"

        # Test very small float below threshold - function uses scientific notation
        result = format_float(0.00001, threshold=1e-3)
        assert result == "1e-5"

        # Test with custom precision
        result = format_float(0.123456789, precision=2)
        assert result == "0.12"

    def test_get_device(self):
        """Test device detection function."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_get_new_version_path(self, tmp_path):
        """Test versioned path creation."""
        base_path = tmp_path / "test_dir"

        # First call should return the base path and create it
        result = get_new_version_path(base_path)
        assert result == base_path
        assert base_path.exists()

        # Add a file to make the directory non-empty
        (base_path / "test_file.txt").write_text("test")

        # Second call should return v_1 (since base_path now exists and is not empty)
        result = get_new_version_path(base_path)
        assert result == tmp_path / "test_dir_v_1"

    def test_sigmoid(self):
        """Test sigmoid function."""
        # Test with array
        x = np.array([0, 1, -1, 2])
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(result, expected)

        # Test with single value
        result = sigmoid(0)
        assert result == 0.5

    def test_to_np(self):
        """Test conversion to numpy array."""
        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = to_np(arr)
        np.testing.assert_array_equal(result, arr)

        # Test with list
        lst = [1, 2, 3]
        result = to_np(lst)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

        # Test with torch tensor (if available)
        try:
            import torch

            tensor = torch.tensor([1, 2, 3])
            result = to_np(tensor)
            np.testing.assert_array_equal(result, np.array([1, 2, 3]))
        except ImportError:
            pass  # torch not available


class TestMetricsFunctions:
    """Tests for metrics calculation functions."""

    def test_safe_hmean(self):
        """Test safe harmonic mean calculation."""
        # Test with valid values
        values = [0.5, 0.8, 0.6]
        result = safe_hmean(values)
        expected = 3 / (1 / 0.5 + 1 / 0.8 + 1 / 0.6)
        assert abs(result - expected) < 1e-10

        # Test with NaN values (should be ignored)
        values_with_nan = [0.5, np.nan, 0.6]
        result = safe_hmean(values_with_nan)
        expected = 2 / (1 / 0.5 + 1 / 0.6)
        assert abs(result - expected) < 1e-10

        # Test with all NaN values
        values_all_nan = [np.nan, np.nan]
        result = safe_hmean(values_all_nan)
        assert np.isnan(result)

    def test_compute_basic_metrics(self):
        """Test basic metrics computation."""
        # Create test data
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

        metrics = compute_basic_metrics(y_true, y_pred_prob)

        # Check that all expected metrics are present
        expected_metrics = [
            "error_rate",
            "false_positive_rate",
            "false_negative_rate",
            "AUROC",
            "F1",
            "RMSE",
            "positives_rate",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (float, np.floating))

        # Check specific values
        assert metrics["positives_rate"] == 0.6  # 3 out of 5 are positive
        assert 0 <= metrics["AUROC"] <= 1
        assert 0 <= metrics["F1"] <= 1

    def test_compute_basic_metrics_single_class(self):
        """Test metrics computation with single class."""
        # All positive labels
        y_true = np.array([1, 1, 1])
        y_pred_prob = np.array([0.8, 0.9, 0.7])

        metrics = compute_basic_metrics(y_true, y_pred_prob)

        # AUROC should be NaN for single class
        assert np.isnan(metrics["AUROC"])


class TestModelTypeFunctions:
    """Tests for model type utility functions."""

    def test_save_and_load_model_type(self, tmp_path):
        """Test saving and loading model type."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Save model type
        save_model_type("mlp", model_dir)

        # Check file was created
        model_type_file = model_dir / "model_type.txt"
        assert model_type_file.exists()
        assert model_type_file.read_text().strip() == "mlp"

        # Load model type
        loaded_type = load_model_type(model_dir)
        assert loaded_type == "mlp"

    def test_load_model_type_nonexistent_file(self, tmp_path):
        """Test loading model type from nonexistent file."""
        model_dir = tmp_path / "nonexistent"
        model_dir.mkdir()

        result = load_model_type(model_dir)
        assert result is None

    def test_infer_model_type_from_parameters(self, tmp_path):
        """Test inferring model type from parameters file."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create a model_type.txt file (the function looks for this, not parameters.json)
        (model_dir / "model_type.txt").write_text("autoencoder")

        # Test inference
        result = infer_model_type_from_parameters(model_dir)
        assert result == "autoencoder"

    def test_get_available_model_types(self):
        """Test getting available model types."""
        model_types = get_available_model_types()

        assert isinstance(model_types, list)
        assert len(model_types) > 0

        # Check that expected model types are present
        expected_types = ["autoencoder", "mlp", "lightgbm", "logistic", "svm"]
        for model_type in expected_types:
            assert model_type in model_types


class TestRunPredictionPipeline:
    """Tests for the run_prediction_pipeline function."""

    def test_run_prediction_pipeline_no_data_source_error(self):
        """Test error when neither adata nor data_path is provided."""
        with pytest.raises(
            ValueError, match="Either adata or data_path must be provided"
        ):
            run_prediction_pipeline(model_path="fake_path")
