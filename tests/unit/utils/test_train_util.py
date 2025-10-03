import optuna
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scxpand.hyperopt.param_grids import create_optimized_lr_scheduler_config
from scxpand.mlp.mlp_model import FC_Net
from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.train_util import get_lr_scheduler, get_optimizer


class TestTrainUtil:
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return FC_Net(in_dim=100, out_dim=1, hid_layers=(64, 32))

    @pytest.fixture
    def mock_param(self):
        """Create a mock Param for testing."""
        return MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "OneCycleLR",
                "warmup_ratio": 0.1,
            },
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

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing."""
        dataset = TensorDataset(torch.randn(100, 10))
        return DataLoader(dataset, batch_size=16)

    def test_get_optimizer(self, mock_model, mock_param):
        """Test get_optimizer function."""
        optimizer = get_optimizer(
            model=mock_model, optimizer_params=mock_param.get_optimizer_params()
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == mock_param.init_learning_rate
        assert optimizer.param_groups[0]["weight_decay"] == mock_param.weight_decay
        assert optimizer.param_groups[0]["betas"] == mock_param.adam_betas
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_get_lr_scheduler_reducelr(self, mock_model, mock_dataloader):
        """Test get_lr_scheduler function with ReduceLROnPlateau."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 3,
                "min_lr": 1e-6,
            },
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

        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_params=param.get_lr_scheduler_params(),
            n_epochs=10,
            train_loader=mock_dataloader,
            init_learning_rate=param.init_learning_rate,
        )

        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert lr_scheduler.factor == 0.5
        assert lr_scheduler.patience == 3
        assert lr_scheduler.min_lrs[0] == 1e-6

    def test_get_lr_scheduler_steplr(self, mock_model, mock_dataloader):
        """Test get_lr_scheduler function with StepLR."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "StepLR",
                "step_size": 5,
                "gamma": 0.1,
            },
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

        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_params=param.get_lr_scheduler_params(),
            n_epochs=10,
            train_loader=mock_dataloader,
            init_learning_rate=param.init_learning_rate,
        )

        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)
        assert lr_scheduler.step_size == 5
        assert lr_scheduler.gamma == 0.1

    def test_get_lr_scheduler_cosine(self, mock_model, mock_dataloader):
        """Test get_lr_scheduler function with CosineAnnealingLR."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "CosineAnnealingLR",
                "T_max": 10,
                "eta_min": 1e-6,
            },
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

        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_params=param.get_lr_scheduler_params(),
            n_epochs=10,
            train_loader=mock_dataloader,
            init_learning_rate=param.init_learning_rate,
        )

        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert lr_scheduler.T_max == 10
        assert lr_scheduler.eta_min == 1e-6

    def test_get_lr_scheduler_constant(self, mock_model, mock_dataloader):
        """Test get_lr_scheduler function with ConstantLR."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "ConstantLR",
            },
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

        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_params=param.get_lr_scheduler_params(),
            n_epochs=10,
            train_loader=mock_dataloader,
            init_learning_rate=param.init_learning_rate,
        )

        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.ConstantLR)
        assert lr_scheduler.factor == 1.0
        assert lr_scheduler.total_iters == 0

    def test_get_lr_scheduler_none(self, mock_model, mock_dataloader):
        """Test get_lr_scheduler function with NoScheduler."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "NoScheduler",
            },
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

        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_params=param.get_lr_scheduler_params(),
            n_epochs=10,
            train_loader=mock_dataloader,
            init_learning_rate=param.init_learning_rate,
        )

        assert lr_scheduler is None

    def test_get_lr_scheduler_onecycle(self, mock_model, mock_dataloader):
        """Test get_lr_scheduler function with OneCycleLR."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "OneCycleLR",
                "warmup_ratio": 0.1,
            },
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

        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_params=param.get_lr_scheduler_params(),
            n_epochs=10,
            train_loader=mock_dataloader,
            init_learning_rate=param.init_learning_rate,
        )

        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)
        assert (
            lr_scheduler.optimizer.param_groups[0]["max_lr"] == param.init_learning_rate
        )

    def test_create_optimized_lr_scheduler_config_all_types(self):
        """Test create_optimized_lr_scheduler_config function with all scheduler types."""
        # Create a mock trial
        study = optuna.create_study()
        trial = study.ask()
        n_epochs = 30
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

    def test_invalid_lr_scheduler_type(self, mock_model, mock_dataloader):
        """Test that invalid learning rate scheduler types raise ValueError."""
        param = MLPParam(
            n_epochs=10,
            early_stopping_patience=3,
            init_learning_rate=0.001,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_config={
                "type": "InvalidScheduler",
            },
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

        with pytest.raises(
            ValueError, match="Unknown learning rate scheduler: InvalidScheduler"
        ):
            get_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_params=param.get_lr_scheduler_params(),
                n_epochs=10,
                train_loader=mock_dataloader,
                init_learning_rate=param.init_learning_rate,
            )
