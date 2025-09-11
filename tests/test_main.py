"""Tests for the main CLI entry points."""

from unittest.mock import patch

import pytest

from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.core.model_types import ModelSpec
from scxpand.lightgbm.lightgbm_params import LightGBMParams
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.main import optimize, optimize_all, train
from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.classes import ModelType


class TestTrainCommand:
    """Tests for the train command."""

    @pytest.mark.parametrize(
        ("model_type", "param_class", "expected_save_dir"),
        [
            ("autoencoder", AutoEncoderParams, "results/autoencoder_v1"),
            ("mlp", MLPParam, "results/mlp_v1"),
            ("lightgbm", LightGBMParams, "results/lightgbm_v1"),
            ("logistic", LinearClassifierParam, "results/logistic_v1"),
            ("svm", LinearClassifierParam, "results/svm_v1"),
        ],
    )
    def test_train_models(
        self,
        model_type,
        param_class,
        expected_save_dir,
    ):
        """Test training different model types."""
        # Convert string model type to enum for registry
        model_type_enum = ModelType(model_type)

        # Create a mock ModelSpec
        mock_spec = ModelSpec(
            config_func=lambda _: {},
            param_class=param_class,
            runner=lambda *_, **__: {"test": {"metric": 0.5}},
            default_save_dir=f"results/{model_type}",
        )

        with (
            patch("scxpand.main.get_new_version_path") as mock_get_path,
            patch("scxpand.main.load_and_override_params") as mock_load_params,
            patch("scxpand.main.call_training_function") as mock_call_training,
            patch("scxpand.main.validate_and_setup_common") as mock_validate_setup,
            patch.dict(
                "scxpand.main.MODEL_TYPES",
                {model_type_enum: mock_spec},
                clear=True,
            ),
        ):
            # Set up mock return values
            mock_validate_setup.return_value = (model_type_enum, mock_spec)
            mock_get_path.return_value = expected_save_dir
            mock_load_params.return_value = param_class()
            mock_call_training.return_value = {"test": {"metric": 0.5}}

            train(
                model_type=model_type,
                data_path="data/test.h5ad",
                num_workers=2,
            )

            mock_get_path.assert_called_once_with(f"results/{model_type}")
            mock_load_params.assert_called_once()
            mock_call_training.assert_called_once()
            mock_validate_setup.assert_called_once()

    def test_train_invalid_model_type(self):
        """Test that training with invalid model type raises error."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            train(model_type="invalid_model")


class TestOptimizeCommand:
    """Tests for the optimize command."""

    @pytest.mark.parametrize(
        "model_type",
        ["autoencoder", "mlp", "lightgbm", "logistic", "svm"],
    )
    def test_optimize_models(self, model_type):
        """Test optimizing different model types."""
        # Convert string to ModelType enum
        model_type_enum = ModelType(model_type)

        with (
            patch("scxpand.main.HyperparameterOptimizer") as mock_optimizer_cls,
            patch("scxpand.main.validate_and_setup_common") as mock_validate_setup,
            patch.dict(
                "scxpand.main.MODEL_TYPES",
                {model_type_enum: None},  # Just need the key to exist
                clear=True,
            ),
        ):
            # Set up mock return value for validation function
            mock_validate_setup.return_value = (model_type_enum, None)

            mock_optimizer = mock_optimizer_cls.return_value
            mock_study = mock_optimizer.run_optimization.return_value

            optimize(
                model_type=model_type,
                data_path="data/test.h5ad",
                n_trials=2,
            )

            mock_optimizer_cls.assert_called_once()
            mock_optimizer.run_optimization.assert_called_once_with(n_trials=2)
            mock_optimizer.print_results.assert_called_once_with(mock_study)
            mock_validate_setup.assert_called_once()

    def test_optimize_invalid_model_type(self):
        """Test that optimizing with invalid model type raises error."""
        with (
            patch.dict(
                "scxpand.main.MODEL_TYPES",
                {},  # Empty registry
                clear=True,
            ),
        ):
            with pytest.raises(ValueError, match="model_type must be one of"):
                optimize(model_type="invalid_model")

    def test_optimize_with_force_new(self):
        """Test optimizing with force_new parameter."""
        with (
            patch("scxpand.main.HyperparameterOptimizer") as mock_optimizer_cls,
            patch("scxpand.main.validate_and_setup_common") as mock_validate_setup,
            patch.dict(
                "scxpand.main.MODEL_TYPES",
                {ModelType.AUTOENCODER: None},  # Just need the key to exist
                clear=True,
            ),
        ):
            # Set up mock return value for validation function
            mock_validate_setup.return_value = (ModelType.AUTOENCODER, None)

            mock_optimizer = mock_optimizer_cls.return_value
            mock_study = mock_optimizer.run_optimization.return_value

            optimize(
                model_type="autoencoder",
                data_path="data/test.h5ad",
                n_trials=2,
                resume=False,
            )

            mock_optimizer_cls.assert_called_once()
            # Check that resume=False was passed to the optimizer (force_new=True means resume=False)
            call_args = mock_optimizer_cls.call_args
            assert call_args[1]["resume"] is False
            mock_optimizer.run_optimization.assert_called_once_with(n_trials=2)
            mock_optimizer.print_results.assert_called_once_with(mock_study)
            mock_validate_setup.assert_called_once()


class TestOptimizeAllCommand:
    """Tests for the optimize-all command."""

    def test_optimize_all_models(self):
        """Test optimizing all model types."""
        model_types = ["autoencoder", "mlp", "lightgbm"]
        model_type_enums = [ModelType(mt) for mt in model_types]

        with (
            patch("scxpand.main.optimize") as mock_optimize,
            patch("scxpand.main.Path") as mock_path_cls,
            patch.dict(
                "scxpand.main.MODEL_TYPES",
                dict.fromkeys(model_type_enums),  # Just need the keys to exist
                clear=True,
            ),
        ):
            # Set up file path mocking for validation
            mock_data_file = mock_path_cls.return_value
            mock_data_file.exists.return_value = True
            mock_data_file.is_file.return_value = True

            optimize_all(
                data_path="data/test.h5ad",
                n_trials=2,
                num_workers=4,
            )

            assert mock_optimize.call_count == len(model_type_enums)
            for model_type_enum in model_type_enums:
                mock_optimize.assert_any_call(
                    model_type=model_type_enum,
                    data_path="data/test.h5ad",
                    n_trials=2,
                    study_name=model_type_enum.value,
                    storage_path="results/optuna_studies",
                    score_metric="harmonic_avg/AUROC",
                    resume=True,
                    num_workers=4,
                )
