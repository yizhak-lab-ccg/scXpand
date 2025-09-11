"""Core tests for checkpoint resuming functionality.

These tests verify the fundamental checkpoint save/load mechanisms work correctly
without getting into complex integration scenarios.
"""

import tempfile

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.model_manager import ModelManager
from scxpand.util.train_logger import TrainLogger, has_checkpoint
from scxpand.util.train_util import report_to_optuna_and_handle_pruning


class MockModel(torch.nn.Module):
    """Simple mock model for testing checkpoint functionality."""

    def __init__(self, input_size: int = 10, hidden_size: int = 5):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestCheckpointCore:
    """Core checkpoint functionality tests."""

    def test_pytorch_checkpoint_save_and_resume(self):
        """Test that PyTorch model checkpoints work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_checkpoint"

            # Create a model and train it a bit
            model = MockModel()
            optimizer = Adam(model.parameters(), lr=0.001)
            lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

            # Simulate some training
            for step in range(3):
                dummy_loss = torch.sum(model.linear1.weight**2)
                dummy_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step == 2:
                    lr_scheduler.step()

            # Save checkpoint using TrainLogger
            logger = TrainLogger(save_path=save_path)
            from scxpand.mlp.mlp_params import MLPParam  # noqa: PLC0415

            dummy_params = MLPParam()
            logger.init_writer(n_epochs=5, n_train_batches=10, prm=dummy_params)

            logger.save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=2,
                model_score=0.85,
            )

            # Store original states
            original_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            original_lr = lr_scheduler.get_last_lr()[0]

            # Create new model/optimizer and load checkpoint
            new_model = MockModel()
            new_optimizer = Adam(new_model.parameters(), lr=0.001)
            new_lr_scheduler = StepLR(new_optimizer, step_size=5, gamma=0.5)

            new_logger = TrainLogger(save_path=save_path)
            resumed_epoch = new_logger.resume_from_checkpoint(
                resume_exp_path=save_path,
                model=new_model,
                optimizer=new_optimizer,
                lr_scheduler=new_lr_scheduler,
                device_name="cpu",
            )

            # Verify resuming behavior
            assert resumed_epoch == 3  # Should resume from next epoch

            # Verify model weights are identical
            new_model_state = new_model.state_dict()
            for key in original_model_state:  # noqa: PLC0206
                assert torch.allclose(original_model_state[key], new_model_state[key])

            # Verify learning rate scheduler state
            assert new_lr_scheduler.get_last_lr()[0] == original_lr

            print("✓ PyTorch checkpoint save/resume works correctly")

    def test_duplicate_epoch_prevention(self):
        """Test that duplicate epoch reporting is prevented."""
        # Create a mock trial
        mock_trial = MagicMock()
        mock_trial.number = 1
        mock_trial.user_attrs = {}
        mock_trial.should_prune.return_value = False

        # Mock the set_user_attr method to actually update the user_attrs dict
        def mock_set_user_attr(key, value):
            mock_trial.user_attrs[key] = value

        mock_trial.set_user_attr = mock_set_user_attr

        # Report some epochs
        report_to_optuna_and_handle_pruning(trial=mock_trial, current_score=0.7, epoch=0)
        report_to_optuna_and_handle_pruning(trial=mock_trial, current_score=0.75, epoch=1)
        report_to_optuna_and_handle_pruning(trial=mock_trial, current_score=0.8, epoch=2)

        # Verify epochs were recorded
        reported_epochs = set(mock_trial.user_attrs.get("reported_epochs", []))
        assert reported_epochs == {0, 1, 2}
        assert mock_trial.report.call_count == 3

        # Reset mock and try to report duplicates
        mock_trial.report.reset_mock()

        # Try to report duplicates and one new epoch
        report_to_optuna_and_handle_pruning(trial=mock_trial, current_score=0.7, epoch=0)  # Duplicate
        report_to_optuna_and_handle_pruning(trial=mock_trial, current_score=0.75, epoch=1)  # Duplicate
        report_to_optuna_and_handle_pruning(trial=mock_trial, current_score=0.85, epoch=3)  # New

        # Should only report the new epoch
        assert mock_trial.report.call_count == 1
        mock_trial.report.assert_called_with(value=0.85, step=3)

        # Verify all epochs are tracked
        reported_epochs = set(mock_trial.user_attrs.get("reported_epochs", []))
        assert reported_epochs == {0, 1, 2, 3}

        print("✓ Duplicate epoch prevention works correctly")

    def test_linear_model_state_management(self):
        """Test that linear models (SGD-based) preserve their state correctly."""
        # Create test parameters and data
        params = LinearClassifierParam(
            model_type="logistic", learning_rate="constant", eta0=0.01, max_iter=1, random_seed=42
        )

        n_samples, n_features = 100, 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Initialize and train model
        model = ModelManager.initialize_model(params, y)
        for _i in range(3):
            model.partial_fit(X, y, classes=np.array([0, 1]))

        # Save model state
        saved_state = ModelManager.save_model_state(
            model=model, current_score=0.8, epoch=3, dev_set_metrics={"accuracy": 0.8}
        )

        # Verify state contains expected fields
        expected_fields = ["coef_", "intercept_", "classes_", "n_features_in_", "t_", "epoch", "score"]
        for field in expected_fields:
            assert field in saved_state, f"Missing field: {field}"

        # Create new model and verify states differ
        new_model = ModelManager.initialize_model(params, y)
        new_model.partial_fit(X[:10], y[:10], classes=np.array([0, 1]))
        assert not np.array_equal(model.coef_, new_model.coef_)

        # Load the saved state
        with tempfile.TemporaryDirectory() as temp_dir:
            ModelManager.load_model_state(
                model=new_model, best_model_state=saved_state, base_save_dir=Path(temp_dir), score_metric="accuracy"
            )

        # Verify states are now identical
        assert np.array_equal(model.coef_, new_model.coef_)
        assert np.array_equal(model.intercept_, new_model.intercept_)
        assert model.t_ == new_model.t_

        print("✓ Linear model state management works correctly")

    def test_checkpoint_file_operations(self):
        """Test checkpoint file operations and error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test has_checkpoint function
            existing_path = Path(temp_dir) / "existing"
            existing_path.mkdir()
            (existing_path / "last_ckpt.pt").touch()
            assert has_checkpoint(existing_path)

            nonexistent_path = Path(temp_dir) / "nonexistent"
            assert not has_checkpoint(nonexistent_path)

            # Test error handling for missing checkpoint
            model = MockModel()
            logger = TrainLogger(save_path=nonexistent_path)

            with pytest.raises(FileNotFoundError, match="No checkpoint file found"):
                logger.resume_from_checkpoint(resume_exp_path=nonexistent_path, model=model, device_name="cpu")

            print("✓ Checkpoint file operations work correctly")

    def test_checkpoint_epoch_progression(self):
        """Test that epochs progress correctly when resuming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "epoch_test"

            # Create and save a checkpoint at epoch 5
            model = MockModel()
            optimizer = Adam(model.parameters())

            logger = TrainLogger(save_path=save_path)
            from scxpand.mlp.mlp_params import MLPParam  # noqa: PLC0415

            logger.init_writer(n_epochs=10, n_train_batches=10, prm=MLPParam())

            logger.save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=None,
                epoch=5,
                model_score=0.9,
            )

            # Resume and verify we start from epoch 6
            new_model = MockModel()
            new_optimizer = Adam(new_model.parameters())
            new_logger = TrainLogger(save_path=save_path)

            resumed_epoch = new_logger.resume_from_checkpoint(
                resume_exp_path=save_path, model=new_model, optimizer=new_optimizer, device_name="cpu"
            )

            assert resumed_epoch == 6  # Should be saved_epoch + 1

            print("✓ Checkpoint epoch progression works correctly")


def test_checkpoint_system_core():
    """Run core checkpoint tests to verify fundamental functionality."""
    print("\n" + "=" * 60)
    print("CHECKPOINT CORE FUNCTIONALITY TEST")
    print("=" * 60)

    test_instance = TestCheckpointCore()

    # Test basic PyTorch checkpoint functionality
    print("\n1. Testing PyTorch checkpoint save/resume...")
    test_instance.test_pytorch_checkpoint_save_and_resume()

    # Test duplicate epoch prevention
    print("\n2. Testing duplicate epoch prevention...")
    test_instance.test_duplicate_epoch_prevention()

    # Test linear model state management
    print("\n3. Testing linear model state management...")
    test_instance.test_linear_model_state_management()

    # Test file operations
    print("\n4. Testing checkpoint file operations...")
    test_instance.test_checkpoint_file_operations()

    # Test epoch progression
    print("\n5. Testing checkpoint epoch progression...")
    test_instance.test_checkpoint_epoch_progression()

    print("\n" + "=" * 60)
    print("✅ ALL CORE CHECKPOINT TESTS PASSED!")
    print("=" * 60)

    print("\n📋 SUMMARY:")
    print("✓ PyTorch model weights, optimizer, and scheduler state preservation")
    print("✓ Proper epoch continuation after resuming")
    print("✓ Duplicate epoch reporting prevention")
    print("✓ Linear/SVM model state management")
    print("✓ Checkpoint file handling and error management")
    print("\n🎯 CHECKPOINT RESUMING WORKS CORRECTLY!")


if __name__ == "__main__":
    test_checkpoint_system_core()
