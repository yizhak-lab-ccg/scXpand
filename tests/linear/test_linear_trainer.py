"""Tests for linear model trainer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.linear_trainer import LinearTrainer, run_linear_training
from tests.test_utils import create_temp_h5ad_file, safe_context_manager


class TestLinearTrainer:
    """Test the LinearTrainer class."""

    @pytest.fixture
    def mock_adata(self):
        """Create a mock AnnData object."""
        n_obs, n_vars = 100, 20
        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(
            {
                "expansion": np.random.choice([0, 1], n_obs),
                "clone_id_size": np.random.randint(1, 100, n_obs),
                "median_clone_size": np.random.randint(1, 50, n_obs),
                "tissue_type": np.random.choice(["A", "B"], n_obs),
                "imputed_labels": np.random.choice(["X", "Y"], n_obs),
                "patient": [f"p{i // 10}" for i in range(n_obs)],
            }
        )
        return ad.AnnData(X=X, obs=obs)

    @pytest.fixture
    def mock_h5ad_file(self, mock_adata):
        """Create a temporary H5AD file for testing."""
        with safe_context_manager() as ctx:
            file_path = create_temp_h5ad_file(mock_adata, ctx.temp_dir)
            ctx.register_adata(file_path)
            yield file_path

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    def test_prepare_data_and_model(self, mock_prepare, mock_h5ad_file):
        """Test data preparation and model initialization."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam()
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = MagicMock()
            mock_data_bundle.data_path = mock_h5ad_file
            mock_data_bundle.row_inds_train = np.arange(80)
            mock_data_bundle.row_inds_dev = np.arange(80, 100)
            mock_data_bundle.data_format = MagicMock()
            mock_data_bundle.data_format.n_genes = 20
            mock_data_bundle.data_format.aux_categorical_mappings = {}
            mock_data_bundle.data_format.aux_categorical_types = []
            mock_prepare.return_value = mock_data_bundle

            # Mock the CellsDataset and DataLoader creation
            with patch(
                "scxpand.linear.linear_trainer.CellsDataset"
            ) as mock_dataset_class:
                with patch(
                    "scxpand.linear.linear_trainer.create_train_dataloader"
                ) as mock_train_dl:
                    with patch(
                        "scxpand.linear.linear_trainer.create_eval_dataloader"
                    ) as mock_eval_dl:
                        mock_train_dataset = MagicMock()
                        mock_dev_dataset = MagicMock()
                        mock_train_dataloader = MagicMock()
                        mock_dev_dataloader = MagicMock()

                        mock_train_dataset.y.numpy.return_value = np.random.randint(
                            0, 2, 80
                        )
                        mock_train_dataset.__len__.return_value = 80
                        mock_dev_dataset.__len__.return_value = 20

                        # Return different datasets for train and dev
                        mock_dataset_class.side_effect = [
                            mock_train_dataset,
                            mock_dev_dataset,
                        ]
                        mock_train_dl.return_value = mock_train_dataloader
                        mock_eval_dl.return_value = mock_dev_dataloader

                        result = trainer.prepare_data_and_model(
                            dev_ratio=0.2,
                            data_path="test_path",
                        )

                        assert (
                            len(result) == 5
                        )  # Should return model, train_dataset, train_dataloader, dev_dataset, dev_dataloader
                        (
                            model,
                            train_dataset,
                            train_dataloader,
                            dev_dataset,
                            dev_dataloader,
                        ) = result
                        assert train_dataset == mock_train_dataset
                        assert dev_dataset == mock_dev_dataset
                        assert train_dataloader == mock_train_dataloader
                        assert dev_dataloader == mock_dev_dataloader
                        mock_prepare.assert_called_once()

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    @patch("scxpand.linear.linear_trainer.report_to_optuna_and_handle_pruning")
    def test_run_training_basic(self, mock_optuna, mock_prepare, mock_h5ad_file):  # noqa: ARG002
        """Test basic training run."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam(n_epochs=2, eval_interval=1)
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = MagicMock()
            mock_data_bundle.data_path = mock_h5ad_file
            mock_data_bundle.row_inds_train = np.arange(80)
            mock_data_bundle.row_inds_dev = np.arange(80, 100)
            mock_data_bundle.data_format = MagicMock()
            mock_data_bundle.data_format.n_genes = 20
            mock_data_bundle.data_format.aux_categorical_mappings = {}
            mock_data_bundle.data_format.aux_categorical_types = []
            mock_prepare.return_value = mock_data_bundle

            with patch.object(trainer, "prepare_data_and_model") as mock_prep:
                mock_model = MagicMock()
                mock_train_dataset = MagicMock()
                mock_train_dataloader = MagicMock()
                mock_dev_dataset = MagicMock()
                mock_dev_dataloader = MagicMock()

                # Mock the dataloader length for training progress
                mock_train_dataloader.__len__ = MagicMock(return_value=3)

                # Mock the dev dataloader to return batches
                mock_dev_dataloader.__iter__ = MagicMock(
                    return_value=iter(
                        [
                            {
                                "x": MagicMock(
                                    numpy=MagicMock(return_value=np.random.rand(10, 20))
                                ),
                                "y": MagicMock(
                                    numpy=MagicMock(
                                        return_value=np.random.randint(0, 2, 10)
                                    )
                                ),
                            },
                            {
                                "x": MagicMock(
                                    numpy=MagicMock(return_value=np.random.rand(10, 20))
                                ),
                                "y": MagicMock(
                                    numpy=MagicMock(
                                        return_value=np.random.randint(0, 2, 10)
                                    )
                                ),
                            },
                        ]
                    )
                )

                mock_prep.return_value = (
                    mock_model,
                    mock_train_dataset,
                    mock_train_dataloader,
                    mock_dev_dataset,
                    mock_dev_dataloader,
                )

                # Mock the evaluate_model and finalize_training methods directly
                with patch.object(trainer, "evaluate_model") as mock_evaluate:
                    with patch.object(trainer, "finalize_training") as mock_finalize:
                        mock_evaluate.return_value = (
                            0.75,
                            {"AUROC": 0.8},
                            np.random.rand(20),
                        )
                        mock_finalize.return_value = {"dev": {"AUROC": 0.8}}

                        results = trainer.run_training(
                            dev_ratio=0.2,
                            trial=None,
                            score_metric="harmonic_avg/AUROC",
                            data_path="test_path",
                        )

                        assert "dev" in results
                        mock_finalize.assert_called_once()

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    def test_run_training_early_stopping(self, mock_prepare, mock_h5ad_file):
        """Test training with early stopping."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam(
                n_epochs=10, early_stopping_patience=2, eval_interval=1
            )
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = MagicMock()
            mock_data_bundle.data_path = mock_h5ad_file
            mock_data_bundle.row_inds_train = np.arange(80)
            mock_data_bundle.row_inds_dev = np.arange(80, 100)
            mock_data_bundle.data_format = MagicMock()
            mock_data_bundle.data_format.n_genes = 20
            mock_data_bundle.data_format.aux_categorical_mappings = {}
            mock_data_bundle.data_format.aux_categorical_types = []
            mock_prepare.return_value = mock_data_bundle

            with patch.object(trainer, "prepare_data_and_model") as mock_prep:
                mock_model = MagicMock()
                mock_train_dataset = MagicMock()
                mock_train_dataloader = MagicMock()
                mock_dev_dataset = MagicMock()
                mock_dev_dataloader = MagicMock()

                # Mock the dataloader length for training progress
                mock_train_dataloader.__len__ = MagicMock(return_value=3)

                # Mock the dev dataloader to return batches
                mock_dev_dataloader.__iter__ = MagicMock(
                    return_value=iter(
                        [
                            {
                                "x": MagicMock(
                                    numpy=MagicMock(return_value=np.random.rand(10, 20))
                                ),
                                "y": MagicMock(
                                    numpy=MagicMock(
                                        return_value=np.random.randint(0, 2, 10)
                                    )
                                ),
                            },
                            {
                                "x": MagicMock(
                                    numpy=MagicMock(return_value=np.random.rand(10, 20))
                                ),
                                "y": MagicMock(
                                    numpy=MagicMock(
                                        return_value=np.random.randint(0, 2, 10)
                                    )
                                ),
                            },
                        ]
                    )
                )

                mock_prep.return_value = (
                    mock_model,
                    mock_train_dataset,
                    mock_train_dataloader,
                    mock_dev_dataset,
                    mock_dev_dataloader,
                )

                # Mock the evaluate_model and finalize_training methods directly
                with patch.object(trainer, "evaluate_model") as mock_evaluate:
                    with patch.object(trainer, "finalize_training") as mock_finalize:
                        # Return decreasing scores to trigger early stopping
                        scores = [0.8, 0.7, 0.6]  # Decreasing scores
                        mock_evaluate.side_effect = [
                            (score, {"AUROC": score}, np.random.rand(20))
                            for score in scores
                        ]
                        mock_finalize.return_value = {"dev": {"AUROC": 0.8}}

                        results = trainer.run_training(
                            dev_ratio=0.2,
                            trial=None,
                            score_metric="harmonic_avg/AUROC",
                            data_path="test_path",
                        )

                        assert "dev" in results
                        mock_finalize.assert_called_once()


class TestRunLinearTraining:
    """Test the run_linear_training function."""

    @patch("scxpand.linear.linear_trainer.LinearTrainer")
    def test_run_linear_training_function(self, mock_trainer_class):
        """Test the run_linear_training function."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam()

            # Mock the trainer instance
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.run_training.return_value = {"dev": {"AUROC": 0.8}}
            mock_trainer_class.return_value = mock_trainer_instance

            results = run_linear_training(
                base_save_dir=base_save_dir,
                prm=prm,
                dev_ratio=0.2,
                trial=None,
                score_metric="harmonic_avg/AUROC",
                data_path="test_path",
            )

            assert "dev" in results
            mock_trainer_class.assert_called_once_with(
                prm=prm, base_save_dir=base_save_dir
            )
            mock_trainer_instance.run_training.assert_called_once()


class TestLinearTrainerEarlyStopping:
    """Test early stopping functionality in linear trainer."""

    @pytest.fixture
    def mock_adata(self):
        """Create a mock AnnData object for testing."""
        n_cells = 100
        n_genes = 20
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(
            {
                "cell_type": np.random.choice(["A", "B"], n_cells),
                "is_malignant": np.random.choice([0, 1], n_cells),
            }
        )
        return ad.AnnData(X=X, obs=obs)

    @pytest.fixture
    def mock_h5ad_file(self, mock_adata):
        """Create a temporary H5AD file for testing."""
        with safe_context_manager() as ctx:
            file_path = create_temp_h5ad_file(mock_adata, ctx.temp_dir)
            ctx.register_adata(file_path)
            yield file_path

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    def test_early_stopping_with_improving_scores(self, mock_prepare, mock_h5ad_file):
        """Test that early stopping is NOT triggered when scores are improving."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam(
                n_epochs=10, early_stopping_patience=3, eval_interval=1
            )
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = self._create_mock_data_bundle(mock_h5ad_file)
            mock_prepare.return_value = mock_data_bundle

            with patch.object(trainer, "prepare_data_and_model") as mock_prep:
                mock_components = self._create_mock_training_components()
                mock_prep.return_value = mock_components

                with patch.object(trainer, "evaluate_model") as mock_evaluate:
                    with patch.object(trainer, "finalize_training") as mock_finalize:
                        # Return improving scores - should NOT trigger early stopping
                        improving_scores = [
                            0.6,
                            0.7,
                            0.8,
                            0.85,
                            0.9,
                            0.91,
                            0.92,
                            0.93,
                            0.94,
                            0.95,
                        ]  # Continuously improving
                        mock_evaluate.side_effect = [
                            (score, {"AUROC": score}, np.random.rand(20))
                            for score in improving_scores
                        ]
                        mock_finalize.return_value = {"dev": {"AUROC": 0.9}}

                        results = trainer.run_training(
                            dev_ratio=0.2,
                            trial=None,
                            score_metric="harmonic_avg/AUROC",
                            data_path="test_path",
                        )

                        assert "dev" in results
                        # Should complete all 10 evaluations since scores keep improving
                        assert mock_evaluate.call_count == 10
                        mock_finalize.assert_called_once()

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    def test_early_stopping_with_no_improvement(self, mock_prepare, mock_h5ad_file):
        """Test that early stopping IS triggered when scores stop improving."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam(
                n_epochs=10, early_stopping_patience=2, eval_interval=1
            )
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = self._create_mock_data_bundle(mock_h5ad_file)
            mock_prepare.return_value = mock_data_bundle

            with patch.object(trainer, "prepare_data_and_model") as mock_prep:
                mock_components = self._create_mock_training_components()
                mock_prep.return_value = mock_components

                with patch.object(trainer, "evaluate_model") as mock_evaluate:
                    with patch.object(trainer, "finalize_training") as mock_finalize:
                        # Return scores that plateau - should trigger early stopping after patience
                        plateau_scores = [
                            0.8,
                            0.7,
                            0.6,
                            0.65,
                            0.64,
                        ]  # Initial improvement then plateau
                        mock_evaluate.side_effect = [
                            (score, {"AUROC": score}, np.random.rand(20))
                            for score in plateau_scores
                        ]
                        mock_finalize.return_value = {"dev": {"AUROC": 0.8}}

                        results = trainer.run_training(
                            dev_ratio=0.2,
                            trial=None,
                            score_metric="harmonic_avg/AUROC",
                            data_path="test_path",
                        )

                        assert "dev" in results
                        # Should stop early after patience (2) + 1 initial best = 3 evaluations
                        assert mock_evaluate.call_count == 3
                        mock_finalize.assert_called_once()

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    def test_early_stopping_with_mixed_scores(self, mock_prepare, mock_h5ad_file):
        """Test early stopping with scores that improve then worsen."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            prm = LinearClassifierParam(
                n_epochs=15, early_stopping_patience=3, eval_interval=1
            )
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = self._create_mock_data_bundle(mock_h5ad_file)
            mock_prepare.return_value = mock_data_bundle

            with patch.object(trainer, "prepare_data_and_model") as mock_prep:
                mock_components = self._create_mock_training_components()
                mock_prep.return_value = mock_components

                with patch.object(trainer, "evaluate_model") as mock_evaluate:
                    with patch.object(trainer, "finalize_training") as mock_finalize:
                        # Scores improve then worsen - should trigger early stopping
                        mixed_scores = [
                            0.5,
                            0.7,
                            0.9,
                            0.85,
                            0.8,
                            0.75,
                            0.7,
                        ]  # Peak at 0.9, then decline
                        mock_evaluate.side_effect = [
                            (score, {"AUROC": score}, np.random.rand(20))
                            for score in mixed_scores
                        ]
                        mock_finalize.return_value = {"dev": {"AUROC": 0.9}}

                        results = trainer.run_training(
                            dev_ratio=0.2,
                            trial=None,
                            score_metric="harmonic_avg/AUROC",
                            data_path="test_path",
                        )

                        assert "dev" in results
                        # Should stop after 3 (best) + 3 (patience) = 6 evaluations
                        assert mock_evaluate.call_count == 6
                        mock_finalize.assert_called_once()

    @patch("scxpand.linear.linear_trainer.prepare_data_for_training")
    def test_early_stopping_respects_eval_interval(self, mock_prepare, mock_h5ad_file):
        """Test that early stopping respects eval_interval setting."""
        with safe_context_manager() as ctx:
            base_save_dir = Path(ctx.temp_dir)
            # Set eval_interval=2 so evaluation happens every 2 epochs
            prm = LinearClassifierParam(
                n_epochs=10, early_stopping_patience=2, eval_interval=2
            )
            trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)

            # Setup mocks
            mock_data_bundle = self._create_mock_data_bundle(mock_h5ad_file)
            mock_prepare.return_value = mock_data_bundle

            with patch.object(trainer, "prepare_data_and_model") as mock_prep:
                mock_components = self._create_mock_training_components()
                mock_prep.return_value = mock_components

                with patch.object(trainer, "train_epoch") as mock_train_epoch:
                    with patch.object(trainer, "evaluate_model") as mock_evaluate:
                        with patch.object(
                            trainer, "finalize_training"
                        ) as mock_finalize:
                            # Return declining scores
                            declining_scores = [0.8, 0.7, 0.6]
                            mock_evaluate.side_effect = [
                                (score, {"AUROC": score}, np.random.rand(20))
                                for score in declining_scores
                            ]
                            mock_finalize.return_value = {"dev": {"AUROC": 0.8}}

                            results = trainer.run_training(
                                dev_ratio=0.2,
                                trial=None,
                                score_metric="harmonic_avg/AUROC",
                                data_path="test_path",
                            )

                            assert "dev" in results
                            # With eval_interval=2, should evaluate at epochs 2, 4, 6
                            # Early stopping triggers after 2 consecutive non-improvements
                            assert mock_evaluate.call_count == 3
                            # Should train for 6 epochs (3 evaluations * 2 interval)
                            assert mock_train_epoch.call_count == 6

    def _create_mock_data_bundle(self, mock_h5ad_file):
        """Helper to create mock data bundle."""
        mock_data_bundle = MagicMock()
        mock_data_bundle.data_path = mock_h5ad_file
        mock_data_bundle.row_inds_train = np.arange(80)
        mock_data_bundle.row_inds_dev = np.arange(80, 100)
        mock_data_bundle.data_format = MagicMock()
        mock_data_bundle.data_format.n_genes = 20
        mock_data_bundle.data_format.aux_categorical_mappings = {}
        mock_data_bundle.data_format.aux_categorical_types = []
        return mock_data_bundle

    def _create_mock_training_components(self):
        """Helper to create mock training components."""
        mock_model = MagicMock()
        mock_train_dataset = MagicMock()
        mock_train_dataloader = MagicMock()
        mock_dev_dataset = MagicMock()
        mock_dev_dataloader = MagicMock()

        mock_train_dataloader.__len__ = MagicMock(return_value=3)
        mock_dev_dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "x": MagicMock(
                            numpy=MagicMock(return_value=np.random.rand(10, 20))
                        ),
                        "y": MagicMock(
                            numpy=MagicMock(return_value=np.random.randint(0, 2, 10))
                        ),
                    },
                ]
            )
        )

        return (
            mock_model,
            mock_train_dataset,
            mock_train_dataloader,
            mock_dev_dataset,
            mock_dev_dataloader,
        )
