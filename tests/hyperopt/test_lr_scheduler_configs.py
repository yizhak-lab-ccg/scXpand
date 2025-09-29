"""Tests for learning rate scheduler configurations and hyperparameter optimization.

This module tests all learning rate scheduler types and their hyperparameter optimization
ranges to ensure they are correctly implemented and follow best practices.
"""

import optuna
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scxpand.hyperopt.param_grids import (
    configure_ae_trial_params,
    configure_mlp_trial_params,
    create_optimized_lr_scheduler_config,
)
from scxpand.mlp.mlp_model import FC_Net
from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.train_util import get_lr_scheduler, get_optimizer


class TestLRSchedulerConfigurations:
    """Test learning rate scheduler configurations and hyperparameter optimization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return FC_Net(in_dim=100, out_dim=1, hid_layers=(64, 32))

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing."""
        dataset = TensorDataset(torch.randn(100, 10))
        return DataLoader(dataset, batch_size=16)

    def test_optimized_lr_scheduler_config_all_types(self):
        """Test create_optimized_lr_scheduler_config function with all scheduler types."""
        n_epochs = 30
        study = optuna.create_study()
        trial = study.ask()
        prefix = "test_"

        # Test ReduceLROnPlateau
        config = create_optimized_lr_scheduler_config(
            trial, "ReduceLROnPlateau", n_epochs, prefix
        )
        assert config["type"] == "ReduceLROnPlateau"
        assert "factor" in config
        assert "patience" in config
        assert "min_lr" in config
        assert 0.1 <= config["factor"] <= 0.7
        assert 3 <= config["patience"] <= 15
        assert 1e-7 <= config["min_lr"] <= 1e-5

        # Test OneCycleLR
        config = create_optimized_lr_scheduler_config(
            trial, "OneCycleLR", n_epochs, prefix
        )
        assert config["type"] == "OneCycleLR"
        assert "warmup_ratio" in config
        assert 0.05 <= config["warmup_ratio"] <= 0.3

        # Test StepLR
        config = create_optimized_lr_scheduler_config(trial, "StepLR", n_epochs, prefix)
        assert config["type"] == "StepLR"
        assert "step_size" in config
        assert "gamma" in config
        assert 5 <= config["step_size"] <= 20
        assert 0.1 <= config["gamma"] <= 0.7

        # Test CosineAnnealingLR
        config = create_optimized_lr_scheduler_config(
            trial, "CosineAnnealingLR", n_epochs, prefix
        )
        assert config["type"] == "CosineAnnealingLR"
        assert "T_max" in config
        assert "eta_min" in config
        assert n_epochs <= config["T_max"] <= n_epochs * 2
        assert 1e-7 <= config["eta_min"] <= 1e-5

        # Test ConstantLR
        config = create_optimized_lr_scheduler_config(
            trial, "ConstantLR", n_epochs, prefix
        )
        assert config["type"] == "ConstantLR"
        assert len(config) == 1  # Only type, no additional params

        # Test NoScheduler
        config = create_optimized_lr_scheduler_config(
            trial, "NoScheduler", n_epochs, prefix
        )
        assert config["type"] == "NoScheduler"
        assert len(config) == 1  # Only type, no additional params

    def test_mlp_trial_params_lr_scheduler_integration(self):
        """Test that MLP trial parameters include all LR scheduler types."""
        study = optuna.create_study()
        trial = study.ask()

        params = configure_mlp_trial_params(trial)

        # Check that LR scheduler type is included
        assert "lr_scheduler_type" in params
        assert "lr_scheduler_config" in params

        # Check that the scheduler type is one of the supported types
        supported_types = [
            "ReduceLROnPlateau",
            "OneCycleLR",
            "StepLR",
            "CosineAnnealingLR",
            "ConstantLR",
            "NoScheduler",
        ]
        assert params["lr_scheduler_type"] in supported_types

        # Check that config has correct type
        assert params["lr_scheduler_config"]["type"] == params["lr_scheduler_type"]

    def test_ae_trial_params_lr_scheduler_integration(self):
        """Test that AutoEncoder trial parameters include all LR scheduler types."""
        study = optuna.create_study()
        trial = study.ask()

        params = configure_ae_trial_params(trial)

        # Check that LR scheduler type is included
        assert "lr_scheduler_type" in params
        assert "lr_scheduler_config" in params

        # Check that the scheduler type is one of the supported types
        supported_types = [
            "ReduceLROnPlateau",
            "OneCycleLR",
            "StepLR",
            "CosineAnnealingLR",
            "ConstantLR",
            "NoScheduler",
        ]
        assert params["lr_scheduler_type"] in supported_types

        # Check that config has correct type
        assert params["lr_scheduler_config"]["type"] == params["lr_scheduler_type"]

    def test_lr_scheduler_creation_with_all_types(self, mock_model, mock_dataloader):
        """Test that all LR scheduler types can be created successfully."""
        n_epochs = 10
        init_learning_rate = 0.001

        scheduler_configs = {
            "ReduceLROnPlateau": {
                "type": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 10,
                "min_lr": 1e-6,
            },
            "OneCycleLR": {
                "type": "OneCycleLR",
                "warmup_ratio": 0.1,
            },
            "StepLR": {
                "type": "StepLR",
                "step_size": 10,
                "gamma": 0.5,
            },
            "CosineAnnealingLR": {
                "type": "CosineAnnealingLR",
                "T_max": n_epochs,
                "eta_min": 1e-6,
            },
            "ConstantLR": {
                "type": "ConstantLR",
            },
            "NoScheduler": {
                "type": "NoScheduler",
            },
        }

        for scheduler_type, scheduler_config in scheduler_configs.items():
            # Create parameter with this scheduler type
            param = MLPParam(
                n_epochs=n_epochs,
                early_stopping_patience=3,
                init_learning_rate=init_learning_rate,
                weight_decay=0.01,
                max_grad_norm=1.0,
                lr_scheduler_config=scheduler_config,
                optimizer_type="AdamW",
                adam_betas=(0.9, 0.999),
                train_batch_size=32,
                inference_batch_size=64,
                sampler_type="balanced",
                layer_units=(64, 32),
                dropout_rate=0.3,
                mask_rate=0.0,
                noise_std=0.0,
                soft_loss_beta=None,
                soft_loss_start_epoch=None,
                positives_weight=1.0,
                train_log_interval=10,
            )

            optimizer = get_optimizer(
                model=mock_model, optimizer_params=param.get_optimizer_params()
            )

            # This should not raise any exceptions
            lr_scheduler = get_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_params=param.get_lr_scheduler_params(),
                n_epochs=n_epochs,
                train_loader=mock_dataloader,
                init_learning_rate=init_learning_rate,
            )

            # Verify the scheduler type
            if scheduler_type == "NoScheduler":
                assert lr_scheduler is None
            else:
                assert lr_scheduler is not None
                # Check that it's the correct type
                expected_class = getattr(torch.optim.lr_scheduler, scheduler_type)
                assert isinstance(lr_scheduler, expected_class)

    def test_hyperparameter_ranges_are_reasonable(self):
        """Test that hyperparameter ranges are within reasonable bounds."""
        n_epochs = 30
        study = optuna.create_study()
        trial = study.ask()
        prefix = "test_"

        # Test ReduceLROnPlateau ranges
        config = create_optimized_lr_scheduler_config(
            trial, "ReduceLROnPlateau", n_epochs, prefix
        )
        assert 0.1 <= config["factor"] <= 0.7  # Reasonable factor range
        assert 3 <= config["patience"] <= 15  # Reasonable patience range
        assert 1e-7 <= config["min_lr"] <= 1e-5  # Reasonable min_lr range

        # Test OneCycleLR ranges
        config = create_optimized_lr_scheduler_config(
            trial, "OneCycleLR", n_epochs, prefix
        )
        assert 0.05 <= config["warmup_ratio"] <= 0.3  # Reasonable warmup range

        # Test StepLR ranges
        config = create_optimized_lr_scheduler_config(trial, "StepLR", n_epochs, prefix)
        assert 5 <= config["step_size"] <= 20  # Reasonable step_size range
        assert 0.1 <= config["gamma"] <= 0.7  # Reasonable gamma range

        # Test CosineAnnealingLR ranges
        config = create_optimized_lr_scheduler_config(
            trial, "CosineAnnealingLR", n_epochs, prefix
        )
        assert n_epochs <= config["T_max"] <= n_epochs * 2  # Reasonable T_max range
        assert 1e-7 <= config["eta_min"] <= 1e-5  # Reasonable eta_min range
