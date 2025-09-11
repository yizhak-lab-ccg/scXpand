"""Tests for core.evaluation module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from scxpand.core.evaluation import evaluate_predictions_and_save
from scxpand.util.classes import ModelType


class TestEvaluatePredictionsAndSave:
    """Test suite for evaluate_predictions_and_save function."""

    @pytest.fixture
    def mock_predictions(self):
        """Mock prediction probabilities."""
        return np.array([0.8, 0.2, 0.9, 0.1, 0.7])

    @pytest.fixture
    def mock_obs_df(self):
        """Mock observation DataFrame with ground truth labels."""
        return pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
                "expansion": ["expanded", "non-expanded", "expanded", "non-expanded", "expanded"],
                "clone_id_size": [5, 2, 8, 1, 6],
                "median_clone_size": [3, 3, 3, 3, 3],
                "tissue_type": ["tumor", "normal", "tumor", "normal", "tumor"],
                "imputed_labels": ["CD8+ T", "CD4+ T", "CD8+ T", "CD4+ T", "CD8+ T"],
            }
        )

    @pytest.fixture
    def mock_ground_truth(self):
        """Mock ground truth labels."""
        return np.array([1, 0, 1, 0, 1])  # Binary labels

    @pytest.fixture
    def mock_evaluation_results(self):
        """Mock evaluation results."""
        return {
            "AUROC": 0.85,
            "accuracy": 0.8,
            "F1": 0.82,
            "precision": 0.83,
            "recall": 0.81,
            "overall": {"AUROC": 0.85, "accuracy": 0.8},
            "harmonic_avg": {"AUROC": 0.84, "accuracy": 0.79},
        }

    def test_evaluate_predictions_and_save_success(
        self, tmp_path, mock_predictions, mock_obs_df, mock_ground_truth, mock_evaluation_results
    ):
        """Test successful evaluation and saving."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv") as mock_save_csv,
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.return_value = mock_evaluation_results

            # Run the function
            result = evaluate_predictions_and_save(
                y_pred_prob=mock_predictions,
                obs_df=mock_obs_df,
                model_type=ModelType.AUTOENCODER,
                save_path=save_path,
                eval_name="test_eval",
                score_metric="harmonic_avg/AUROC",
                trial=None,
            )

            # Verify extract_is_expanded was called correctly
            mock_extract.assert_called_once_with(mock_obs_df)

            # Verify save_predictions_to_csv was called correctly
            mock_save_csv.assert_called_once_with(
                predictions=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.AUTOENCODER, save_path=save_path
            )

            # Verify evaluate_and_save was called correctly
            mock_evaluate.assert_called_once_with(
                y_true=mock_ground_truth,
                y_pred_prob=mock_predictions,
                obs_df=mock_obs_df,
                eval_name="test_eval",
                save_path=save_path,
                plots_dir=save_path / "plots",
                score_metric="harmonic_avg/AUROC",
                trial=None,
            )

            # Verify result
            assert result == mock_evaluation_results

    def test_evaluate_predictions_and_save_with_trial(
        self, tmp_path, mock_predictions, mock_obs_df, mock_ground_truth, mock_evaluation_results
    ):
        """Test evaluation with Optuna trial object."""
        save_path = tmp_path / "results"
        save_path.mkdir()
        mock_trial = Mock()

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv"),
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.return_value = mock_evaluation_results

            # Run the function with trial
            result = evaluate_predictions_and_save(
                y_pred_prob=mock_predictions,
                obs_df=mock_obs_df,
                model_type="mlp",  # Test string input
                save_path=save_path,
                eval_name="validation",
                score_metric="AUROC",
                trial=mock_trial,
            )

            # Verify trial was passed to evaluate_and_save
            mock_evaluate.assert_called_once()
            call_kwargs = mock_evaluate.call_args[1]
            assert call_kwargs["trial"] == mock_trial
            assert call_kwargs["score_metric"] == "AUROC"
            assert call_kwargs["eval_name"] == "validation"

            # Verify result is returned
            assert result == mock_evaluation_results

    def test_evaluate_predictions_and_save_default_parameters(
        self, tmp_path, mock_predictions, mock_obs_df, mock_ground_truth, mock_evaluation_results
    ):
        """Test evaluation with default parameters."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv"),
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.return_value = mock_evaluation_results

            # Run with minimal parameters (using defaults)
            result = evaluate_predictions_and_save(
                y_pred_prob=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.LIGHTGBM, save_path=save_path
            )

            # Verify default parameters were used
            mock_evaluate.assert_called_once()
            call_kwargs = mock_evaluate.call_args[1]
            assert call_kwargs["eval_name"] == "dev"  # default
            assert call_kwargs["score_metric"] == "harmonic_avg/AUROC"  # default
            assert call_kwargs["trial"] is None  # default

            # Verify result is returned
            assert result == mock_evaluation_results

    def test_evaluate_predictions_and_save_array_shapes_mismatch(self, tmp_path, mock_obs_df):
        """Test handling of mismatched array shapes."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        # Create mismatched predictions (wrong length)
        wrong_predictions = np.array([0.8, 0.2])  # Only 2 predictions for 5 obs
        mock_ground_truth = np.array([1, 0, 1, 0, 1])

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv") as mock_save_csv,
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks - evaluate_and_save should handle the shape mismatch
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.side_effect = ValueError("Shape mismatch")

            # Should propagate the error from evaluate_and_save
            with pytest.raises(ValueError, match="Shape mismatch"):
                evaluate_predictions_and_save(
                    y_pred_prob=wrong_predictions, obs_df=mock_obs_df, model_type=ModelType.SVM, save_path=save_path
                )

            # Verify that extraction and saving were attempted
            mock_extract.assert_called_once_with(mock_obs_df)
            mock_save_csv.assert_called_once()

    def test_evaluate_predictions_and_save_logging(
        self, tmp_path, mock_predictions, mock_obs_df, mock_ground_truth, mock_evaluation_results
    ):
        """Test that the function completes successfully (basic integration test)."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv"),
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.return_value = mock_evaluation_results

            # Run evaluation - should complete without error
            result = evaluate_predictions_and_save(
                y_pred_prob=mock_predictions,
                obs_df=mock_obs_df,
                model_type=ModelType.MLP,
                save_path=save_path,
                eval_name="test",
            )

            # Verify function completed and returned results
            assert result == mock_evaluation_results
            mock_extract.assert_called_once()
            mock_evaluate.assert_called_once()

    def test_evaluate_predictions_and_save_missing_auroc(
        self, tmp_path, mock_predictions, mock_obs_df, mock_ground_truth
    ):
        """Test handling when AUROC is missing from results."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        # Results without AUROC key
        results_without_auroc = {"accuracy": 0.8, "F1": 0.75}

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv"),
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.return_value = results_without_auroc

            # Run evaluation - should complete without error even without AUROC
            result = evaluate_predictions_and_save(
                y_pred_prob=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.LOGISTIC, save_path=save_path
            )

            # Verify function completed and returned results
            assert result == results_without_auroc
            assert "AUROC" not in result  # Confirm AUROC is indeed missing

    def test_evaluate_predictions_and_save_path_handling(
        self, tmp_path, mock_predictions, mock_obs_df, mock_ground_truth, mock_evaluation_results
    ):
        """Test that Path objects are handled correctly."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv"),
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = mock_ground_truth
            mock_evaluate.return_value = mock_evaluation_results

            # Run with Path object
            result = evaluate_predictions_and_save(
                y_pred_prob=mock_predictions,
                obs_df=mock_obs_df,
                model_type=ModelType.AUTOENCODER,
                save_path=save_path,  # Path object, not string
            )

            # Verify Path was handled correctly
            mock_evaluate.assert_called_once()
            call_kwargs = mock_evaluate.call_args[1]
            assert call_kwargs["save_path"] == save_path
            assert call_kwargs["plots_dir"] == save_path / "plots"

            # Verify result is returned
            assert result == mock_evaluation_results

    def test_evaluate_predictions_and_save_empty_predictions(self, tmp_path):
        """Test handling of empty predictions array."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        empty_predictions = np.array([])
        empty_obs_df = pd.DataFrame()

        with (
            patch("scxpand.core.evaluation.extract_is_expanded") as mock_extract,
            patch("scxpand.core.evaluation.save_predictions_to_csv") as mock_save_csv,
            patch("scxpand.core.evaluation.evaluate_and_save") as mock_evaluate,
        ):
            # Setup mocks
            mock_extract.return_value = np.array([])
            mock_evaluate.return_value = {}

            # Should handle empty arrays gracefully
            result = evaluate_predictions_and_save(
                y_pred_prob=empty_predictions, obs_df=empty_obs_df, model_type=ModelType.LIGHTGBM, save_path=save_path
            )

            # With empty DataFrame (no expansion column), extract_is_expanded should not be called
            # but save_predictions_to_csv should still be called
            mock_extract.assert_not_called()
            mock_save_csv.assert_called_once()
            mock_evaluate.assert_not_called()  # Should not be called when expansion column is missing
            assert result == {}
