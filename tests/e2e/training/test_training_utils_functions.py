"""Tests for training utility functions.

This module provides comprehensive tests for the training utility functions
that are part of the public API.
"""

from unittest.mock import Mock

import pytest

from scxpand.core.model_types import ModelSpec
from scxpand.util.classes import ModelType
from scxpand.util.training_utils import (
    call_training_function,
    validate_and_setup_common,
)


class TestValidateAndSetupCommon:
    """Tests for the validate_and_setup_common function."""

    def test_validate_and_setup_common_valid_model_type(self):
        """Test validation with valid model type."""
        model_type_enum, model_spec = validate_and_setup_common("mlp")

        assert model_type_enum == ModelType.MLP
        assert isinstance(model_spec, ModelSpec)

    def test_validate_and_setup_common_enum_model_type(self):
        """Test validation with enum model type."""
        model_type_enum, model_spec = validate_and_setup_common(ModelType.AUTOENCODER)

        assert model_type_enum == ModelType.AUTOENCODER
        assert isinstance(model_spec, ModelSpec)

    def test_validate_and_setup_common_invalid_model_type(self):
        """Test validation with invalid model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            validate_and_setup_common("invalid_model")

    def test_validate_and_setup_common_empty_model_type(self):
        """Test validation with empty model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            validate_and_setup_common("")

    def test_validate_and_setup_common_none_model_type(self):
        """Test validation with None model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            validate_and_setup_common(None)

    def test_validate_and_setup_common_with_valid_data_path(self, tmp_path):
        """Test validation with valid data path."""
        # Create a test data file
        data_file = tmp_path / "test_data.h5ad"
        data_file.write_text("test data")

        model_type_enum, model_spec = validate_and_setup_common(
            "mlp", data_path=str(data_file)
        )

        assert model_type_enum == ModelType.MLP
        assert isinstance(model_spec, ModelSpec)

    def test_validate_and_setup_common_with_nonexistent_data_path(self):
        """Test validation with nonexistent data path."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            validate_and_setup_common("mlp", data_path="nonexistent_file.h5ad")

    def test_validate_and_setup_common_with_empty_data_path(self):
        """Test validation with empty data path."""
        with pytest.raises(ValueError, match="data_path cannot be empty"):
            validate_and_setup_common("mlp", data_path="")

    def test_validate_and_setup_common_with_directory_data_path(self, tmp_path):
        """Test validation with directory as data path."""
        # Create a directory instead of a file
        data_dir = tmp_path / "data_dir"
        data_dir.mkdir()

        with pytest.raises(ValueError, match="Data path is not a file"):
            validate_and_setup_common("mlp", data_path=str(data_dir))

    def test_validate_and_setup_common_with_valid_model_path(self, tmp_path):
        """Test validation with valid model path."""
        # Create a test model directory
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        model_type_enum, model_spec = validate_and_setup_common(
            "mlp", model_path=str(model_dir)
        )

        assert model_type_enum == ModelType.MLP
        assert isinstance(model_spec, ModelSpec)

    def test_validate_and_setup_common_with_nonexistent_model_path(self):
        """Test validation with nonexistent model path."""
        with pytest.raises(FileNotFoundError, match="Model path not found"):
            validate_and_setup_common("mlp", model_path="nonexistent_model")

    def test_validate_and_setup_common_with_file_model_path(self, tmp_path):
        """Test validation with file as model path."""
        # Create a file instead of a directory
        model_file = tmp_path / "model_file.txt"
        model_file.write_text("test")

        with pytest.raises(ValueError, match="Model path is not a directory"):
            validate_and_setup_common("mlp", model_path=str(model_file))

    def test_validate_and_setup_common_all_model_types(self):
        """Test validation with all supported model types."""
        supported_types = ["autoencoder", "mlp", "lightgbm", "logistic", "svm"]

        for model_type in supported_types:
            model_type_enum, model_spec = validate_and_setup_common(model_type)
            assert model_type_enum.value == model_type
            assert isinstance(model_spec, ModelSpec)

    def test_validate_and_setup_common_case_sensitivity(self):
        """Test that model types are case sensitive."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            validate_and_setup_common("MLP")  # Should be lowercase


class TestCallTrainingFunction:
    """Tests for the call_training_function function."""

    def test_call_training_function_basic(self):
        """Test basic functionality of call_training_function."""
        # Mock training function
        mock_train_fn = Mock()
        mock_train_fn.return_value = {"test": "result"}

        # Mock parameters
        mock_prm = Mock()

        # Call function
        result = call_training_function(
            model_type=ModelType.MLP,
            train_fn=mock_train_fn,
            data_path="test_data.h5ad",
            save_dir="test_save_dir",
            prm=mock_prm,
            resume=False,
            num_workers=4,
        )

        # Verify training function was called
        mock_train_fn.assert_called_once()

        # Verify call arguments - MLP uses base_save_dir, not save_dir
        call_args = mock_train_fn.call_args
        assert call_args[1]["data_path"] == "test_data.h5ad"
        assert call_args[1]["base_save_dir"] == "test_save_dir"
        assert call_args[1]["prm"] is mock_prm
        assert call_args[1]["resume"] is False
        assert call_args[1]["num_workers"] == 4

    def test_call_training_function_with_resume(self):
        """Test call_training_function with resume=True."""
        mock_train_fn = Mock()
        mock_train_fn.return_value = {"test": "result"}
        mock_prm = Mock()

        result = call_training_function(
            model_type=ModelType.AUTOENCODER,
            train_fn=mock_train_fn,
            data_path="test_data.h5ad",
            save_dir="test_save_dir",
            prm=mock_prm,
            resume=True,
            num_workers=2,
        )

        # Verify resume parameter was passed
        call_args = mock_train_fn.call_args
        assert call_args[1]["resume"] is True
        assert call_args[1]["num_workers"] == 2

    def test_call_training_function_all_model_types(self):
        """Test call_training_function with all model types."""
        model_types = [
            ModelType.AUTOENCODER,
            ModelType.MLP,
            ModelType.LIGHTGBM,
            ModelType.LOGISTIC,
            ModelType.SVM,
        ]

        for model_type in model_types:
            mock_train_fn = Mock()
            mock_train_fn.return_value = {"test": "result"}
            mock_prm = Mock()

            result = call_training_function(
                model_type=model_type,
                train_fn=mock_train_fn,
                data_path="test_data.h5ad",
                save_dir="test_save_dir",
                prm=mock_prm,
                resume=False,
                num_workers=4,
            )

            # Verify function was called for each model type
            mock_train_fn.assert_called_once()

    def test_call_training_function_training_function_exception(self):
        """Test call_training_function when training function raises exception."""
        mock_train_fn = Mock()
        mock_train_fn.side_effect = RuntimeError("Training failed")
        mock_prm = Mock()

        with pytest.raises(RuntimeError, match="Training failed"):
            call_training_function(
                model_type=ModelType.MLP,
                train_fn=mock_train_fn,
                data_path="test_data.h5ad",
                save_dir="test_save_dir",
                prm=mock_prm,
                resume=False,
                num_workers=4,
            )

    def test_call_training_function_return_value(self):
        """Test that call_training_function returns the result from training function."""
        expected_result = {"accuracy": 0.95, "loss": 0.1}
        mock_train_fn = Mock()
        mock_train_fn.return_value = expected_result
        mock_prm = Mock()

        result = call_training_function(
            model_type=ModelType.MLP,
            train_fn=mock_train_fn,
            data_path="test_data.h5ad",
            save_dir="test_save_dir",
            prm=mock_prm,
            resume=False,
            num_workers=4,
        )

        assert result == expected_result

    def test_call_training_function_parameter_passing(self):
        """Test that all parameters are correctly passed to training function."""
        mock_train_fn = Mock()
        mock_train_fn.return_value = {}
        mock_prm = Mock()

        # Test with various parameter values
        test_cases = [
            {"resume": True, "num_workers": 1},
            {"resume": False, "num_workers": 8},
            {"resume": True, "num_workers": 0},
        ]

        for test_case in test_cases:
            mock_train_fn.reset_mock()

            call_training_function(
                model_type=ModelType.MLP,
                train_fn=mock_train_fn,
                data_path="test_data.h5ad",
                save_dir="test_save_dir",
                prm=mock_prm,
                **test_case,
            )

            # Verify parameters were passed correctly - MLP uses base_save_dir, not save_dir
            call_args = mock_train_fn.call_args[1]
            assert call_args["resume"] == test_case["resume"]
            assert call_args["num_workers"] == test_case["num_workers"]
            assert call_args["data_path"] == "test_data.h5ad"
            assert call_args["base_save_dir"] == "test_save_dir"
            assert call_args["prm"] is mock_prm
