"""Tests for linear model hyperparameter optimization.

This module tests the hyperparameter optimization functions for logistic regression
and SVM models, including learning rate scheduler configurations.
"""

import numpy as np
import optuna

from scxpand.hyperopt.param_grids import (
    create_optimized_lr_scheduler_config,
)
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.model_manager import ModelManager


class TestLinearHyperopt:
    """Test linear model hyperparameter optimization."""

    def test_linear_lr_scheduler_configs(self):
        """Test learning rate scheduler configurations for linear models."""
        n_epochs = 30
        study = optuna.create_study()
        trial = study.ask()

        # Test logistic regression LR scheduler
        config = create_optimized_lr_scheduler_config(trial, "ReduceLROnPlateau", n_epochs, "logistic_")
        assert config["type"] == "ReduceLROnPlateau"
        assert "factor" in config
        assert "patience" in config
        assert "min_lr" in config

        # Test SVM LR scheduler
        config = create_optimized_lr_scheduler_config(trial, "CosineAnnealingLR", n_epochs, "svm_")
        assert config["type"] == "CosineAnnealingLR"
        assert "T_max" in config
        assert "eta_min" in config

    def test_linear_model_initialization_with_new_params(self):
        """Test that linear models can be initialized with the new parameters."""
        # Test logistic regression
        param = LinearClassifierParam(
            model_type="logistic",
            alpha=0.1,
            penalty="l2",
            n_epochs=30,
            class_weight="balanced",
            tol=1e-4,
            l1_ratio=0.15,
            random_seed=42,
            batch_size=1024,
            early_stopping_patience=5,
            eval_interval=1,
            train_log_interval=10,
            sampler_type="random",
            mask_rate=0.0,
            noise_std=0.0,
            soft_loss_beta=1.0,
            init_learning_rate=0.001,
            learning_rate="constant",
            eta0=0.01,
            power_t=0.5,
            lr_scheduler_type="ConstantLR",
            lr_scheduler_config={"type": "ConstantLR"},
            warm_start=True,
            average=False,
            n_iter_no_change=5,
            validation_fraction=0.1,
            fit_intercept=True,
            shuffle=True,
        )

        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model = ModelManager.initialize_model(param, y_train)

        assert model.loss == "log_loss"
        assert model.penalty == "l2"
        assert model.alpha == 0.1
        assert model.l1_ratio == 0.15
        assert model.max_iter == 1
        assert model.tol == 1e-4
        assert model.warm_start is True
        assert model.learning_rate == "constant"
        assert model.eta0 == 0.01
        assert model.power_t == 0.5
        assert model.average is False
        assert model.n_iter_no_change == 5
        assert model.validation_fraction == 0.1
        assert model.early_stopping is False  # Should always be False when using partial_fit()
        assert model.fit_intercept is True
        assert model.shuffle is True

        # Test SVM
        param.model_type = "svm"
        model = ModelManager.initialize_model(param, y_train)
        assert model.loss == "hinge"

    def test_linear_params_interface_methods(self):
        """Test that LinearClassifierParam has all required interface methods."""
        param = LinearClassifierParam()

        # Test dataset params
        dataset_params = param.get_dataset_params()
        assert hasattr(dataset_params, "mask_rate")
        assert hasattr(dataset_params, "noise_std")
        assert hasattr(dataset_params, "soft_loss_beta")

        # Test LR scheduler params
        lr_params = param.get_lr_scheduler_params()
        assert hasattr(lr_params, "lr_scheduler_type")
        assert hasattr(lr_params, "lr_scheduler_config")

        # Test optimizer params
        opt_params = param.get_optimizer_params()
        assert hasattr(opt_params, "optimizer_type")
        assert hasattr(opt_params, "adam_betas")
        assert hasattr(opt_params, "weight_decay")
        assert hasattr(opt_params, "max_grad_norm")
        assert hasattr(opt_params, "init_learning_rate")
