"""Tests for util.io module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scxpand.util.classes import ModelType
from scxpand.util.io import ensure_directory_exists, load_eval_indices, save_predictions_to_csv


class TestLoadEvalIndices:
    """Test suite for load_eval_indices function."""

    def test_load_eval_indices_success(self, tmp_path):
        """Test successful loading of evaluation indices."""
        # Create test file with indices
        indices_file = tmp_path / "eval_indices.txt"
        test_indices = [0, 2, 5, 8, 10]
        indices_file.write_text("\n".join(map(str, test_indices)))

        # Load indices
        result = load_eval_indices(str(indices_file))

        # Verify result
        expected = np.array(test_indices, dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_load_eval_indices_with_path_object(self, tmp_path):
        """Test loading with Path object input."""
        indices_file = tmp_path / "eval_indices.txt"
        test_indices = [1, 3, 7]
        indices_file.write_text("\n".join(map(str, test_indices)))

        # Load with Path object
        result = load_eval_indices(indices_file)  # Path object, not string

        expected = np.array(test_indices, dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_load_eval_indices_file_not_found(self, tmp_path):
        """Test handling of non-existent file."""
        nonexistent_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="Evaluation indices file not found"):
            load_eval_indices(str(nonexistent_file))

    def test_load_eval_indices_empty_file(self, tmp_path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = load_eval_indices(str(empty_file))

        # Should return empty array
        assert len(result) == 0
        assert result.dtype == int

    def test_load_eval_indices_with_whitespace(self, tmp_path):
        """Test handling of file with whitespace and empty lines."""
        indices_file = tmp_path / "indices_with_whitespace.txt"
        content = """0

        2


        5
        """
        indices_file.write_text(content)

        result = load_eval_indices(str(indices_file))

        expected = np.array([0, 2, 5], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_load_eval_indices_with_multiple_formats(self, tmp_path):
        """Test loading indices with different file formats."""
        # Test with different line endings and spacing
        indices_file = tmp_path / "eval_indices.txt"
        indices_file.write_text("  1  \n\n  2  \n  3  \n")

        result = load_eval_indices(str(indices_file))

        expected = np.array([1, 2, 3], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_load_eval_indices_invalid_content(self, tmp_path):
        """Test handling of file with invalid (non-integer) content."""
        indices_file = tmp_path / "invalid.txt"
        indices_file.write_text("0\ninvalid\n2")

        # Should raise ValueError when trying to convert 'invalid' to int
        with pytest.raises(ValueError, match="could not convert string"):
            load_eval_indices(str(indices_file))


class TestSavePredictionsToCsv:
    """Test suite for save_predictions_to_csv function."""

    @pytest.fixture
    def mock_predictions(self):
        """Mock prediction probabilities."""
        return np.array([0.8, 0.2, 0.9, 0.1, 0.7])

    @pytest.fixture
    def mock_obs_df(self):
        """Mock observation DataFrame."""
        return pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
                "tissue_type": ["tumor", "normal", "tumor", "normal", "tumor"],
                "expansion": ["expanded", "non-expanded", "expanded", "non-expanded", "expanded"],
            }
        )

    def test_save_predictions_to_csv_with_enum(self, tmp_path, mock_predictions, mock_obs_df):
        """Test saving predictions with ModelType enum."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        save_predictions_to_csv(
            predictions=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.AUTOENCODER, save_path=save_path
        )

        # Verify file was created
        expected_file = save_path / "autoencoder_predictions.csv"
        assert expected_file.exists()

        # Verify file content
        saved_df = pd.read_csv(expected_file)

        # Check that it contains original columns plus predictions
        assert "cell_id" in saved_df.columns
        assert "tissue_type" in saved_df.columns
        assert "expansion" in saved_df.columns
        assert "autoencoder_prediction" in saved_df.columns

        # Check predictions were saved correctly
        np.testing.assert_array_almost_equal(saved_df["autoencoder_prediction"].values, mock_predictions)

        # Check that original data is preserved
        pd.testing.assert_frame_equal(saved_df[["cell_id", "tissue_type", "expansion"]], mock_obs_df)

    def test_save_predictions_to_csv_with_string(self, tmp_path, mock_predictions, mock_obs_df):
        """Test saving predictions with string model type."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        save_predictions_to_csv(predictions=mock_predictions, obs_df=mock_obs_df, model_type="mlp", save_path=save_path)

        # Verify file was created with correct name
        expected_file = save_path / "mlp_predictions.csv"
        assert expected_file.exists()

        # Verify predictions column name
        saved_df = pd.read_csv(expected_file)
        assert "mlp_prediction" in saved_df.columns

    def test_save_predictions_to_csv_creates_directory(self, tmp_path, mock_predictions, mock_obs_df):
        """Test that save directory is created if it doesn't exist."""
        save_path = tmp_path / "nonexistent" / "nested" / "path"
        # Directory doesn't exist yet

        save_predictions_to_csv(
            predictions=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.LIGHTGBM, save_path=save_path
        )

        # Verify directory was created
        assert save_path.exists()
        assert save_path.is_dir()

        # Verify file was saved
        expected_file = save_path / "lightgbm_predictions.csv"
        assert expected_file.exists()

    def test_save_predictions_to_csv_overwrite_existing(self, tmp_path, mock_predictions, mock_obs_df):
        """Test that existing file is overwritten."""
        save_path = tmp_path / "results"
        save_path.mkdir()
        prediction_file = save_path / "svm_predictions.csv"

        # Create existing file with different content
        existing_df = pd.DataFrame({"old_column": [1, 2, 3]})
        existing_df.to_csv(prediction_file, index=False)

        # Save new predictions
        save_predictions_to_csv(
            predictions=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.SVM, save_path=save_path
        )

        # Verify file was overwritten
        saved_df = pd.read_csv(prediction_file)
        assert "old_column" not in saved_df.columns
        assert "svm_prediction" in saved_df.columns
        assert len(saved_df) == len(mock_predictions)

    def test_save_predictions_to_csv_file_structure(self, tmp_path, mock_predictions, mock_obs_df):
        """Test the structure and content of saved prediction files."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        save_predictions_to_csv(
            predictions=mock_predictions, obs_df=mock_obs_df, model_type=ModelType.LOGISTIC, save_path=save_path
        )

        # Check file was created with correct name
        prediction_file = save_path / "logistic_predictions.csv"
        assert prediction_file.exists()

        # Check file content structure
        saved_df = pd.read_csv(prediction_file)
        assert len(saved_df) == len(mock_predictions)
        assert len(saved_df.columns) == len(mock_obs_df.columns) + 1  # original + prediction column

    def test_save_predictions_to_csv_empty_data(self, tmp_path):
        """Test handling of empty predictions and observations."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        empty_predictions = np.array([])
        empty_obs_df = pd.DataFrame()

        save_predictions_to_csv(
            predictions=empty_predictions, obs_df=empty_obs_df, model_type=ModelType.MLP, save_path=save_path
        )

        # Should create file even with empty data
        expected_file = save_path / "mlp_predictions.csv"
        assert expected_file.exists()

        saved_df = pd.read_csv(expected_file)
        assert len(saved_df) == 0
        assert "mlp_prediction" in saved_df.columns

    def test_save_predictions_to_csv_mismatched_lengths(self, tmp_path, mock_obs_df):
        """Test handling of mismatched prediction and observation lengths."""
        save_path = tmp_path / "results"
        save_path.mkdir()

        # Wrong number of predictions
        wrong_predictions = np.array([0.8, 0.2])  # Only 2 predictions for 5 obs

        # Should raise ValueError due to length mismatch
        with pytest.raises(ValueError, match=r"Predictions length.*doesn't match obs_df length"):
            save_predictions_to_csv(
                predictions=wrong_predictions, obs_df=mock_obs_df, model_type=ModelType.AUTOENCODER, save_path=save_path
            )


class TestEnsureDirectoryExists:
    """Test suite for ensure_directory_exists function."""

    def test_ensure_directory_exists_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_exists_nested_directory(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        ensure_directory_exists(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()
        # Check that all parent directories were created
        assert (tmp_path / "level1").exists()
        assert (tmp_path / "level1" / "level2").exists()

    def test_ensure_directory_exists_existing_directory(self, tmp_path):
        """Test with already existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        assert existing_dir.exists()

        # Should not raise error
        ensure_directory_exists(existing_dir)

        assert existing_dir.exists()
        assert existing_dir.is_dir()

    def test_ensure_directory_exists_with_file_conflict(self, tmp_path):
        """Test behavior when path exists as file."""
        file_path = tmp_path / "existing_file.txt"
        file_path.write_text("content")
        assert file_path.exists()
        assert file_path.is_file()

        # Should raise error when trying to create directory with same name as existing file
        with pytest.raises(FileExistsError):
            ensure_directory_exists(file_path)

    def test_ensure_directory_exists_with_string_path(self, tmp_path):
        """Test with string path instead of Path object."""
        new_dir_str = str(tmp_path / "string_path")

        ensure_directory_exists(Path(new_dir_str))

        assert Path(new_dir_str).exists()
        assert Path(new_dir_str).is_dir()
