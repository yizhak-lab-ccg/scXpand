"""Tests for hyperopt system consistency and parameter validation."""

import inspect
from unittest.mock import MagicMock

import optuna
import pytest

from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.core.model_types import MODEL_TYPES
from scxpand.hyperopt.param_grids import (
    configure_ae_trial_params,
    configure_lightgbm_trial_params,
    configure_logistic_trial_params,
    configure_mlp_trial_params,
    configure_svm_trial_params,
)
from scxpand.lightgbm.lightgbm_params import LightGBMParams
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.mlp.mlp_params import MLPParam


class TestHyperoptParameterConsistency:
    """Test that all parameter classes and parameter grids follow consistent patterns."""

    def test_all_parameter_classes_have_random_seed(self):
        """Test that all parameter classes used in hyperopt have random_seed attribute."""
        param_classes = [
            MLPParam,
            LinearClassifierParam,
            LightGBMParams,
            AutoEncoderParams,
        ]

        for param_class in param_classes:
            # Check that the class has random_seed field
            assert hasattr(param_class, "__dataclass_fields__"), (
                f"{param_class.__name__} is not a dataclass"
            )
            fields = param_class.__dataclass_fields__
            assert "random_seed" in fields, (
                f"{param_class.__name__} missing 'random_seed' field"
            )

            # Check that random_seed has correct type annotation
            field_info = fields["random_seed"]
            assert field_info.type is int, (
                f"{param_class.__name__}.random_seed should be type 'int', got {field_info.type}"
            )

            # Check that default value is reasonable
            instance = param_class()
            assert isinstance(instance.random_seed, int), (
                f"{param_class.__name__}.random_seed default is not int"
            )
            assert instance.random_seed >= 0, (
                f"{param_class.__name__}.random_seed should be non-negative"
            )

    def test_parameter_classes_do_not_use_random_state(self):
        """Test that parameter classes use random_seed, not random_state (consistency with hyperopt)."""
        param_classes = [
            MLPParam,
            LinearClassifierParam,
            LightGBMParams,
            AutoEncoderParams,
        ]

        for param_class in param_classes:
            fields = param_class.__dataclass_fields__
            assert "random_state" not in fields, (
                f"{param_class.__name__} should use 'random_seed', not 'random_state' "
                "for consistency with hyperopt system"
            )

    @pytest.mark.parametrize(
        "config_func",
        [
            configure_mlp_trial_params,
            configure_logistic_trial_params,
            configure_svm_trial_params,
            configure_lightgbm_trial_params,
            configure_ae_trial_params,
        ],
    )
    def test_parameter_grids_work_correctly(self, config_func):
        """Test that all parameter grid functions work correctly and return valid dictionaries."""
        # Create a mock trial
        mock_trial = MagicMock()

        # Setup mock methods for all possible parameter suggestions
        # Return appropriate values based on parameter name to avoid enum validation errors
        def mock_suggest_categorical(name, choices):
            # Return the first valid choice for each parameter type
            if "sampler_type" in name:
                return "random"  # Valid SamplerType value
            elif "lr_scheduler_type" in name:
                return "ReduceLROnPlateau"  # Valid LRSchedulerType value
            elif (
                "use_log_transform" in name
                or "use_soft_loss" in name
                or "use_zscore_norm" in name
            ):
                return True  # Boolean values
            elif "train_batch_size" in name or "batch_size" in name:
                return 2048  # Batch size values
            elif "penalty" in name:
                return "l2"  # Penalty values
            elif "class_weight" in name:
                return "balanced"  # Class weight values
            elif "learning_rate" in name:
                return "optimal"  # Learning rate values
            elif (
                "warm_start" in name
                or "average" in name
                or "fit_intercept" in name
                or "shuffle" in name
            ):
                return True  # Boolean values
            elif "boosting_type" in name:
                return "gbdt"  # Boosting type values
            elif "objective" in name:
                return "binary"  # Objective values
            elif "metric" in name:
                return "binary_logloss"  # Metric values
            elif "verbose" in name:
                return -1  # Verbose values
            elif "model_type" in name:
                return "standard"  # Model type values
            elif "loss_type" in name:
                return "mse"  # Loss type values
            elif "aux_categorical_types" in name:
                return "none"  # Aux categorical types
            elif "layer" in name and "units" in name:
                return 512  # Layer units values
            elif "latent_dim" in name:
                return 32  # Latent dimension values
            else:
                # Fallback to first choice
                return choices[0] if choices else "test_value"

        mock_trial.suggest_categorical.side_effect = mock_suggest_categorical
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.suggest_int.return_value = 10

        # Call the configuration function
        try:
            params = config_func(mock_trial)
        except Exception as e:
            pytest.fail(f"{config_func.__name__} failed with: {e}")

        assert isinstance(params, dict), f"{config_func.__name__} should return a dict"

        # Parameter grids should NOT include random_seed - that's set by the optimizer
        # This is the correct behavior since hyperopt_optimizer.py line 112 sets: param_dict["random_seed"] = trial_seed
        assert "random_seed" not in params, (
            f"{config_func.__name__} should not return 'random_seed' - this is set by the hyperopt optimizer"
        )

    def test_model_registry_consistency(self):
        """Test that MODEL_REGISTRY contains all expected models and their components are consistent."""
        expected_models = {"mlp", "logistic", "svm", "lightgbm", "autoencoder"}

        assert set(MODEL_TYPES.keys()) == expected_models, (
            f"MODEL_REGISTRY should contain {expected_models}, got {set(MODEL_TYPES.keys())}"
        )

        for model_name, spec in MODEL_TYPES.items():
            # Check that each spec has required components
            assert hasattr(spec, "config_func"), (
                f"{model_name} spec missing config_func"
            )
            assert hasattr(spec, "param_class"), (
                f"{model_name} spec missing param_class"
            )
            assert hasattr(spec, "runner"), f"{model_name} spec missing runner"

            # Check that config_func is callable
            assert callable(spec.config_func), (
                f"{model_name} config_func is not callable"
            )

            # Check that param_class is a class
            assert inspect.isclass(spec.param_class), (
                f"{model_name} param_class is not a class"
            )

            # Check that param_class has random_seed
            assert hasattr(spec.param_class, "__dataclass_fields__"), (
                f"{model_name} param_class is not a dataclass"
            )
            fields = spec.param_class.__dataclass_fields__
            assert "random_seed" in fields, (
                f"{model_name} param_class missing 'random_seed' field"
            )

    def test_hyperopt_optimizer_sets_random_seed(self):
        """Test that the hyperopt optimizer pattern sets random_seed correctly."""
        # This tests the pattern used in hyperopt_optimizer.py line 112: param_dict["random_seed"] = trial_seed

        for model_name, spec in MODEL_TYPES.items():
            # Create a mock trial
            mock_trial = MagicMock()

            # Use the same mock setup as the other test to avoid enum validation errors
            def mock_suggest_categorical(name, choices):
                # Return the first valid choice for each parameter type
                if "sampler_type" in name:
                    return "random"  # Valid SamplerType value
                elif "lr_scheduler_type" in name:
                    return "ReduceLROnPlateau"  # Valid LRSchedulerType value
                elif (
                    "use_log_transform" in name
                    or "use_soft_loss" in name
                    or "use_zscore_norm" in name
                ):
                    return True  # Boolean values
                elif "train_batch_size" in name or "batch_size" in name:
                    return 2048  # Batch size values
                elif "penalty" in name:
                    return "l2"  # Penalty values
                elif "class_weight" in name:
                    return "balanced"  # Class weight values
                elif "learning_rate" in name:
                    return "optimal"  # Learning rate values
                elif (
                    "warm_start" in name
                    or "average" in name
                    or "fit_intercept" in name
                    or "shuffle" in name
                ):
                    return True  # Boolean values
                elif "boosting_type" in name:
                    return "gbdt"  # Boosting type values
                elif "objective" in name:
                    return "binary"  # Objective values
                elif "metric" in name:
                    return "binary_logloss"  # Metric values
                elif "verbose" in name:
                    return -1  # Verbose values
                elif "model_type" in name:
                    return "standard"  # Model type values
                elif "loss_type" in name:
                    return "mse"  # Loss type values
                elif "aux_categorical_types" in name:
                    return "none"  # Aux categorical types
                elif "layer" in name and "units" in name:
                    return 512  # Layer units values
                elif "latent_dim" in name:
                    return 32  # Latent dimension values
                else:
                    # Fallback to first choice
                    return choices[0] if choices else "test_value"

            mock_trial.suggest_categorical.side_effect = mock_suggest_categorical
            mock_trial.suggest_float.return_value = 0.1
            mock_trial.suggest_int.return_value = 10

            # Get parameters from config function
            param_dict = spec.config_func(mock_trial)

            # Simulate what optimizer does: override random_seed
            trial_seed = 123
            param_dict["random_seed"] = trial_seed

            # Test that param_class can be instantiated with these parameters
            try:
                params = spec.param_class(**param_dict)
                assert params.random_seed == trial_seed, (
                    f"{model_name} param_class did not accept random_seed override"
                )
            except Exception as e:
                pytest.fail(
                    f"{model_name} param_class failed to instantiate with hyperopt parameters: {e}"
                )

    def test_parameter_grid_functions_accept_optuna_trial(self):
        """Test that all parameter grid functions accept an optuna.Trial object."""
        config_functions = [
            configure_mlp_trial_params,
            configure_logistic_trial_params,
            configure_svm_trial_params,
            configure_lightgbm_trial_params,
            configure_ae_trial_params,
        ]

        for config_func in config_functions:
            sig = inspect.signature(config_func)
            params = list(sig.parameters.keys())

            assert len(params) == 1, (
                f"{config_func.__name__} should accept exactly one parameter"
            )
            assert params[0] == "trial", (
                f"{config_func.__name__} should accept parameter named 'trial'"
            )

            # Check type annotation
            param_annotation = sig.parameters["trial"].annotation
            assert param_annotation == optuna.Trial, (
                f"{config_func.__name__} trial parameter should be annotated as optuna.Trial"
            )

    def test_parameter_default_values_are_reasonable(self):
        """Test that default random_seed values are reasonable across all parameter classes."""
        param_classes = [
            MLPParam,
            LinearClassifierParam,
            LightGBMParams,
            AutoEncoderParams,
        ]

        for param_class in param_classes:
            instance = param_class()

            # Check that random_seed is a reasonable default (commonly 42 or similar)
            assert instance.random_seed in [42, 0, 1, 123], (
                f"{param_class.__name__} has unusual default random_seed: {instance.random_seed}. "
                "Consider using a standard value like 42."
            )

    def test_lightgbm_special_case_random_state_mapping(self):
        """Test that LightGBM correctly maps random_seed to random_state for the backend."""
        # This is a special test for LightGBM since it uses random_state in the sklearn API
        # but we want random_seed in our parameter class for consistency

        params = LightGBMParams(random_seed=999)
        assert params.random_seed == 999

        # The mapping to random_state should happen in the run_lightgbm_training function
        # This test ensures our parameter class uses the consistent naming
        assert hasattr(params, "random_seed")
        assert not hasattr(params, "random_state"), (
            "LightGBMParams should not have random_state attribute"
        )


class TestParameterClassDefaults:
    """Test default values across parameter classes for consistency."""

    @pytest.mark.parametrize(
        "param_class",
        [
            MLPParam,
            LinearClassifierParam,
            LightGBMParams,
            AutoEncoderParams,
        ],
    )
    def test_use_log_transform_default(self, param_class):
        """Test that all parameter classes default to use_log_transform=True."""
        instance = param_class()
        assert instance.use_log_transform is True, (
            f"{param_class.__name__} should default to use_log_transform=True for gene expression data"
        )

    @pytest.mark.parametrize(
        "param_class",
        [
            MLPParam,
            AutoEncoderParams,
        ],
    )
    def test_neural_network_common_defaults(self, param_class):
        """Test that neural network models have consistent defaults."""
        instance = param_class()

        # Common neural network parameters
        assert hasattr(instance, "train_batch_size"), (
            f"{param_class.__name__} missing train_batch_size"
        )
        assert hasattr(instance, "dropout_rate"), (
            f"{param_class.__name__} missing dropout_rate"
        )
        assert hasattr(instance, "mask_rate"), (
            f"{param_class.__name__} missing mask_rate"
        )
        assert hasattr(instance, "noise_std"), (
            f"{param_class.__name__} missing noise_std"
        )

        # Check reasonable defaults
        assert instance.train_batch_size > 0, (
            f"{param_class.__name__} train_batch_size should be positive"
        )
        assert 0 <= instance.dropout_rate <= 1, (
            f"{param_class.__name__} dropout_rate should be in [0,1]"
        )
        assert instance.mask_rate >= 0, (
            f"{param_class.__name__} mask_rate should be non-negative"
        )
        assert instance.noise_std >= 0, (
            f"{param_class.__name__} noise_std should be non-negative"
        )
