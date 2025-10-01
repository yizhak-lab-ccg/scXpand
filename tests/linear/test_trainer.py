"""Tests for linear trainer components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from scxpand.data_util.dataset import CellsDataset
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.linear_trainer import LinearTrainer, TrainingSession
from tests.test_utils import safe_context_manager


class TestTrainingSession:
    """Test the TrainingSession class."""

    def test_initialization(self):
        """Test TrainingSession initialization."""
        prm = LinearClassifierParam()
        session = TrainingSession(prm, "harmonic_avg/AUROC")

        assert session.prm == prm
        assert session.score_metric == "harmonic_avg/AUROC"
        assert session.best_score == float("-inf")
        assert session.best_model_state is None
        assert session.patience_counter == 0
        assert np.array_equal(session.classes, np.array([0, 1]))

    def test_check_early_stopping_improvement(self):
        """Test early stopping when there's improvement."""
        prm = LinearClassifierParam()
        session = TrainingSession(prm, "harmonic_avg/AUROC")

        should_stop = session.check_early_stopping(current_score=0.8, epoch=5)

        assert should_stop is False
        assert session.patience_counter == 0

    def test_check_early_stopping_no_improvement(self):
        """Test early stopping when there's no improvement."""
        prm = LinearClassifierParam()
        session = TrainingSession(prm, "harmonic_avg/AUROC")

        # Set initial best score
        session.best_score = 0.8

        should_stop = session.check_early_stopping(current_score=0.7, epoch=5)

        assert should_stop is False
        assert session.patience_counter == 1

    def test_check_early_stopping_triggered(self):
        """Test early stopping when patience is exceeded."""
        prm = LinearClassifierParam(early_stopping_patience=5)
        session = TrainingSession(prm, "harmonic_avg/AUROC")

        # Set initial best score and patience counter
        session.best_score = 0.8
        session.patience_counter = 4  # One less than patience limit

        should_stop = session.check_early_stopping(current_score=0.7, epoch=10)

        assert should_stop is True
        assert session.patience_counter == 5

    def test_update_best_model(self):
        """Test updating best model state."""
        prm = LinearClassifierParam()
        session = TrainingSession(prm, "harmonic_avg/AUROC")

        # Mock model
        model = MagicMock()
        model.coef_ = np.array([[1.0, 2.0]])
        model.intercept_ = np.array([0.5])
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 2
        model.t_ = 100

        # Mock logger
        mock_logger = MagicMock()

        session.update_best_model(
            model=model,
            current_score=0.85,
            epoch=15,
            dev_set_metrics={"AUROC": 0.8},
            logger=mock_logger,
        )

        assert session.best_score == 0.85
        assert session.best_model_state is not None
        assert session.best_model_state["epoch"] == 15
        assert session.best_model_state["score"] == 0.85
        # Verify logger was updated
        mock_logger.update_best_score.assert_called_once_with(0.85, 15, {"AUROC": 0.8})


class TestLinearTrainer:
    """Test the LinearTrainer class."""

    @pytest.fixture
    def mock_train_dataset(self):
        """Create a mock CellsDataset for training."""
        dataset = MagicMock(spec=CellsDataset)
        dataset.y = MagicMock()
        dataset.y.numpy = MagicMock(return_value=np.array([0, 1, 0, 1, 1]))
        dataset.obs_df = pd.DataFrame({"expansion": [0, 1, 0, 1, 1]})
        return dataset

    @pytest.fixture
    def mock_train_dataloader(self):
        """Create a mock DataLoader for training."""
        dataloader = MagicMock(spec=DataLoader)
        dataloader.__len__ = MagicMock(return_value=4)  # 4 batches

        # Mock iteration to return sample batches
        def mock_iter(self):  # noqa: ARG001
            # Return 4 batches of data
            for i in range(4):
                batch_size = 32 if i < 3 else 4  # Last batch smaller
                X_batch = np.random.rand(batch_size, 10)
                y_batch = np.random.randint(0, 2, batch_size)
                yield {
                    "x": MagicMock(numpy=MagicMock(return_value=X_batch)),
                    "y": MagicMock(numpy=MagicMock(return_value=y_batch)),
                }

        dataloader.__iter__ = mock_iter
        return dataloader

    @pytest.fixture
    def mock_eval_dataset(self):
        """Create a mock CellsDataset for evaluation."""
        dataset = MagicMock(spec=CellsDataset)
        dataset.y = MagicMock()
        dataset.y.numpy = MagicMock(return_value=np.random.randint(0, 2, 50))
        dataset.obs_df = pd.DataFrame(
            {
                "patient": [f"p{i // 5}" for i in range(50)],
                "imputed_labels": np.random.choice(["A", "B"], 50),
                "tissue_type": np.random.choice(["X", "Y"], 50),
            }
        )
        return dataset

    @pytest.fixture
    def mock_eval_dataloader(self):
        """Create a mock DataLoader for evaluation."""
        dataloader = MagicMock(spec=DataLoader)
        # Mock iteration to return sample batches
        dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "x": MagicMock(
                            numpy=MagicMock(return_value=np.random.rand(25, 10))
                        ),
                        "y": MagicMock(
                            numpy=MagicMock(return_value=np.random.randint(0, 2, 25))
                        ),
                    },
                    {
                        "x": MagicMock(
                            numpy=MagicMock(return_value=np.random.rand(25, 10))
                        ),
                        "y": MagicMock(
                            numpy=MagicMock(return_value=np.random.randint(0, 2, 25))
                        ),
                    },
                ]
            )
        )
        return dataloader

    def test_train_epoch(self, mock_train_dataloader):
        """Test training for one epoch."""
        prm = LinearClassifierParam()
        base_save_dir = Path("test_dir")
        trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

        mock_train_logger = MagicMock()
        classes = np.array([0, 1])

        # Create a mock model
        model = MagicMock()

        trainer.train_epoch(
            model=model,
            train_dataloader=mock_train_dataloader,
            train_logger=mock_train_logger,
            classes=classes,
            epoch=0,
        )

        # Check that partial_fit was called for each batch
        assert model.partial_fit.call_count == 4  # 4 batches

        # Check that epoch end logging was called
        mock_train_logger.log_epoch_end.assert_called_once_with(epoch=0)

    def test_train_epoch_different_epochs(self, mock_train_dataloader):
        """Test training for different epochs."""
        prm = LinearClassifierParam()
        base_save_dir = Path("test_dir")
        trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

        mock_train_logger = MagicMock()
        classes = np.array([0, 1])

        test_cases = [0, 5, 10]  # Different epochs

        for epoch in test_cases:
            model = MagicMock()
            mock_train_logger.reset_mock()

            trainer.train_epoch(
                model=model,
                train_dataloader=mock_train_dataloader,
                train_logger=mock_train_logger,
                classes=classes,
                epoch=epoch,
            )

            # Check that partial_fit was called for each batch
            assert model.partial_fit.call_count == 4

            # Check that epoch end logging was called with correct epoch
            mock_train_logger.log_epoch_end.assert_called_once_with(epoch=epoch)

    @patch("scxpand.linear.linear_trainer.calculate_metrics")
    @patch("scxpand.linear.linear_trainer.flatten_nested_dict")
    def test_evaluate_model(
        self, mock_flatten, mock_calc_metrics, mock_eval_dataset, mock_eval_dataloader
    ):
        """Test model evaluation."""
        prm = LinearClassifierParam()
        base_save_dir = Path("test_dir")
        trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

        mock_train_logger = MagicMock()

        # Setup mocks
        mock_calc_metrics.return_value = {"AUROC": 0.8, "F1": 0.7}
        mock_flatten.return_value = {"harmonic_avg/AUROC": 0.75}

        # Mock LinearBatchPredictor
        with patch(
            "scxpand.linear.linear_trainer.LinearBatchPredictor"
        ) as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.predict_all.return_value = np.random.rand(50)
            mock_predictor_class.return_value = mock_predictor

            model = MagicMock()

            score, metrics, predictions = trainer.evaluate_model(
                model=model,
                eval_dataset=mock_eval_dataset,
                eval_dataloader=mock_eval_dataloader,
                train_logger=mock_train_logger,
                score_metric="harmonic_avg/AUROC",
                epoch=0,
            )

            assert score == 0.75
            assert "AUROC" in metrics
            assert len(predictions) == 50

            # Verify function calls
            mock_calc_metrics.assert_called_once()
            mock_flatten.assert_called_once()
            mock_predictor.predict_all.assert_called_once_with(model=model)

    @patch("scxpand.linear.linear_trainer.evaluate_predictions_and_save")
    @patch("scxpand.linear.linear_trainer.joblib.dump")
    def test_finalize_training(
        self, mock_dump, mock_eval_save, mock_eval_dataset, mock_eval_dataloader
    ):
        """Test training finalization."""
        prm = LinearClassifierParam()

        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            mock_train_logger = MagicMock()

            # Setup mock data
            model = MagicMock()

            # Setup mocks
            mock_eval_save.return_value = {"AUROC": 0.8, "accuracy": 0.75}

            # Mock LinearBatchPredictor
            with patch(
                "scxpand.linear.linear_trainer.LinearBatchPredictor"
            ) as mock_predictor_class:
                mock_predictor = MagicMock()
                mock_predictor.predict_all.return_value = np.random.rand(50)
                mock_predictor_class.return_value = mock_predictor

                results = trainer.finalize_training(
                    model=model,
                    eval_dataset=mock_eval_dataset,
                    eval_dataloader=mock_eval_dataloader,
                    train_logger=mock_train_logger,
                    trial=None,
                    score_metric="harmonic_avg/AUROC",
                )

                # Verify results structure and function calls
                assert "AUROC" in results
                assert results["AUROC"] == 0.8
                mock_eval_save.assert_called_once()
                mock_dump.assert_called_once()
                mock_predictor.predict_all.assert_called_once_with(model=model)
