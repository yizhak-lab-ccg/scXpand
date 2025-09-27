"""Tests for training utility functions."""

from unittest.mock import MagicMock, patch

import pytest

from scxpand.util.classes import ModelType
from scxpand.util.training_utils import (
    call_training_function,
    validate_and_setup_common,
)


class TestValidateAndSetupCommon:
    """Test the validate_and_setup_common function."""

    def test_validate_model_type_string(self):
        """Test model type validation with string input."""
        with patch("scxpand.util.training_utils.MODEL_TYPES") as mock_registry:
            mock_spec = MagicMock()
            mock_registry.__contains__.return_value = True
            mock_registry.__getitem__.return_value = mock_spec

            with patch("scxpand.util.training_utils.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.is_file.return_value = True

                model_type, spec = validate_and_setup_common("mlp", "data/test.h5ad")

                assert model_type == ModelType.MLP
                assert spec == mock_spec

    def test_validate_model_type_enum(self):
        """Test model type validation with enum input."""
        with patch("scxpand.util.training_utils.MODEL_TYPES") as mock_registry:
            mock_spec = MagicMock()
            mock_registry.__contains__.return_value = True
            mock_registry.__getitem__.return_value = mock_spec

            with patch("scxpand.util.training_utils.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.is_file.return_value = True

                model_type, spec = validate_and_setup_common(
                    ModelType.MLP, "data/test.h5ad"
                )

                assert model_type == ModelType.MLP
                assert spec == mock_spec

    def test_invalid_model_type(self):
        """Test error for invalid model type."""
        with patch("scxpand.util.training_utils.MODEL_TYPES") as mock_registry:
            mock_registry.__contains__.return_value = False

            with pytest.raises(ValueError, match="model_type must be one of"):
                validate_and_setup_common("invalid", "data/test.h5ad")

    def test_missing_data_file(self):
        """Test error for missing data file."""
        with patch("scxpand.util.training_utils.MODEL_TYPES") as mock_registry:
            mock_spec = MagicMock()
            mock_registry.__contains__.return_value = True
            mock_registry.__getitem__.return_value = mock_spec

            with patch("scxpand.util.training_utils.Path") as mock_path:
                mock_path.return_value.exists.return_value = False

                with pytest.raises(FileNotFoundError, match="Data file not found"):
                    validate_and_setup_common("mlp", "missing/file.h5ad")


class TestCallTrainingFunction:
    """Test the call_training_function dispatcher."""

    def test_neural_network_models(self):
        """Test calling neural network training functions."""
        mock_fn = MagicMock()
        mock_fn.return_value = {"test": {"metric": 0.5}}

        for model_type in [ModelType.AUTOENCODER, ModelType.MLP]:
            result = call_training_function(
                model_type=model_type,
                train_fn=mock_fn,
                data_path="data/test.h5ad",
                save_dir="results/test",
                prm=MagicMock(),
                resume=True,
                num_workers=4,
            )

            mock_fn.assert_called_with(
                data_path="data/test.h5ad",
                base_save_dir="results/test",
                prm=mock_fn.call_args[1]["prm"],
                device=None,
                resume=True,
                num_workers=4,
            )
            assert result == {"test": {"metric": 0.5}}
            mock_fn.reset_mock()

    def test_lightgbm_model(self):
        """Test calling LightGBM training function."""
        mock_fn = MagicMock()
        mock_fn.return_value = {"test": {"metric": 0.8}}

        result = call_training_function(
            model_type=ModelType.LIGHTGBM,
            train_fn=mock_fn,
            data_path="data/test.h5ad",
            save_dir="results/test",
            prm=MagicMock(),
            resume=True,  # Should be ignored for LightGBM
            num_workers=4,  # Should be ignored for LightGBM
        )

        mock_fn.assert_called_once_with(
            base_save_dir="results/test",
            prm=mock_fn.call_args[1]["prm"],
            data_path="data/test.h5ad",
        )
        assert result == {"test": {"metric": 0.8}}

    def test_linear_models(self):
        """Test calling linear model training functions."""
        mock_fn = MagicMock()
        mock_fn.return_value = {"test": {"metric": 0.9}}

        for model_type in [ModelType.LOGISTIC, ModelType.SVM]:
            result = call_training_function(
                model_type=model_type,
                train_fn=mock_fn,
                data_path="data/test.h5ad",
                save_dir="results/test",
                prm=MagicMock(),
                resume=True,  # Should be ignored for linear models
                num_workers=4,
            )

            mock_fn.assert_called_with(
                base_save_dir="results/test",
                prm=mock_fn.call_args[1]["prm"],
                data_path="data/test.h5ad",
                num_workers=4,
            )
            assert result == {"test": {"metric": 0.9}}
            mock_fn.reset_mock()

    def test_unknown_model_type(self):
        """Test error for unknown model type."""
        # Create a fake model type that's not handled
        fake_model_type = "fake_model"

        with pytest.raises(ValueError, match="Unknown model type for training"):
            call_training_function(
                model_type=fake_model_type,
                train_fn=MagicMock(),
                data_path="data/test.h5ad",
                save_dir="results/test",
                prm=MagicMock(),
            )
