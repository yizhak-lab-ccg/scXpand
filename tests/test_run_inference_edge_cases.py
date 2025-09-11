"""Tests for run_inference function with various edge cases and column combinations.

This module tests the run_inference function with different combinations of:
- evaluate_metrics parameter (True/False)
- Available columns in adata.obs (expansion, tissue_type, imputed_labels)
- Different model sources (local, registry, URL)
"""

import tempfile

from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.core.inference import DEFAULT_MODEL_NAME, run_inference
from scxpand.core.inference_results import InferenceResults
from scxpand.data_util.transforms import extract_is_expanded
from scxpand.util.metrics import calculate_metrics


class TestRunInferenceEdgeCases:
    """Test cases for run_inference with various column combinations."""

    @pytest.fixture
    def base_adata(self):
        """Create a base AnnData object with gene expression data."""
        n_cells = 50
        n_genes = 100

        # Random gene expression data
        X = np.random.poisson(5, size=(n_cells, n_genes))

        # Create observation data
        obs_data = {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
        }
        obs_df = pd.DataFrame(obs_data)

        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs_df)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        return adata

    def create_adata_with_columns(self, base_adata, columns_to_add):
        """Add specified columns to the base adata."""
        adata = base_adata.copy()

        if "expansion" in columns_to_add:
            adata.obs["expansion"] = np.random.choice(["expanded", "non-expanded"], len(adata))

        if "tissue_type" in columns_to_add:
            adata.obs["tissue_type"] = np.random.choice(["tumor", "normal"], len(adata))

        if "imputed_labels" in columns_to_add:
            adata.obs["imputed_labels"] = np.random.choice(["T_cell", "B_cell"], len(adata))

        return adata

    @patch("scxpand.core.inference.run_prediction_pipeline")
    def test_run_inference_missing_expansion_column(self, mock_pipeline, base_adata):
        """Test run_inference with missing expansion column."""
        # Create adata without expansion column
        adata = self.create_adata_with_columns(base_adata, ["tissue_type", "imputed_labels"])

        # Mock the prediction pipeline to return empty metrics
        mock_pipeline.return_value = InferenceResults(
            predictions=np.random.random(len(adata)),
            metrics={},  # Empty metrics due to missing expansion column
        )

        # Run inference
        results = run_inference(adata=adata, model_path="fake_model_path", save_path=None)

        # Verify that the pipeline was called
        mock_pipeline.assert_called_once()

        # Verify results structure
        assert isinstance(results, InferenceResults)
        assert len(results.predictions) == len(adata)

        # Metrics should be empty due to missing expansion column
        assert results.metrics == {}

    @patch("scxpand.core.inference.run_prediction_pipeline")
    def test_run_inference_missing_stratification_columns(self, mock_pipeline, base_adata):
        """Test run_inference with missing stratification columns."""
        # Create adata with only expansion column (missing tissue_type, imputed_labels)
        adata = self.create_adata_with_columns(base_adata, ["expansion"])

        # Mock the prediction pipeline to return metrics without stratified results
        mock_pipeline.return_value = InferenceResults(
            predictions=np.random.random(len(adata)),
            metrics={
                "AUROC": 0.85,
                "error_rate": 0.15,
                "harmonic_avg": {"AUROC": 0.85},  # Only overall metrics, no stratified
            },
        )

        # Run inference
        results = run_inference(adata=adata, model_path="fake_model_path", save_path=None)

        # Verify that the pipeline was called
        mock_pipeline.assert_called_once()

        # Verify results structure
        assert isinstance(results, InferenceResults)
        assert len(results.predictions) == len(adata)
        assert results.metrics is not None
        assert "AUROC" in results.metrics
        assert "harmonic_avg" in results.metrics

    @patch("scxpand.core.inference.run_prediction_pipeline")
    def test_run_inference_complete_dataset(self, mock_pipeline, base_adata):
        """Test run_inference with complete dataset (all columns present)."""
        # Create adata with all columns
        adata = self.create_adata_with_columns(base_adata, ["expansion", "tissue_type", "imputed_labels"])

        # Mock the prediction pipeline to return complete metrics
        mock_pipeline.return_value = InferenceResults(
            predictions=np.random.random(len(adata)),
            metrics={
                "AUROC": 0.85,
                "error_rate": 0.15,
                "harmonic_avg": {"AUROC": 0.85},
                "T_cell__tumor": {"AUROC": 0.82},
                "B_cell__normal": {"AUROC": 0.88},
            },
        )

        # Run inference
        results = run_inference(adata=adata, model_path="fake_model_path", save_path=None)

        # Verify that the pipeline was called
        mock_pipeline.assert_called_once()

        # Verify results structure
        assert isinstance(results, InferenceResults)
        assert len(results.predictions) == len(adata)

        # Should have complete metrics including stratified metrics
        assert results.metrics is not None
        assert "AUROC" in results.metrics
        assert "harmonic_avg" in results.metrics
        assert "T_cell__tumor" in results.metrics
        assert "B_cell__normal" in results.metrics

    @patch("scxpand.core.inference.fetch_model_and_run_inference")
    def test_run_inference_registry_model_missing_expansion(self, mock_pretrained, base_adata):
        """Test run_inference with registry model and missing expansion column."""
        # Create adata without expansion column
        adata = self.create_adata_with_columns(base_adata, ["tissue_type", "imputed_labels"])

        # Mock the pretrained inference to return empty metrics
        mock_pretrained.return_value = InferenceResults(
            predictions=np.random.random(len(adata)),
            metrics={},  # Empty metrics due to missing expansion column
        )

        # Run inference with registry model
        results = run_inference(adata=adata, model_name=DEFAULT_MODEL_NAME, save_path=None)

        # Verify that the pretrained inference was called
        mock_pretrained.assert_called_once()

        # Verify results structure
        assert isinstance(results, InferenceResults)
        assert len(results.predictions) == len(adata)
        assert results.metrics == {}

    @patch("scxpand.core.inference.fetch_model_and_run_inference")
    def test_run_inference_url_model(self, mock_pretrained, base_adata):
        """Test run_inference with URL model."""
        # Create adata with all columns
        adata = self.create_adata_with_columns(base_adata, ["expansion", "tissue_type", "imputed_labels"])

        # Mock the pretrained inference to return predictions with metrics
        mock_pretrained.return_value = InferenceResults(
            predictions=np.random.random(len(adata)), metrics={"AUROC": 0.85}
        )

        # Run inference with URL model
        results = run_inference(adata=adata, model_url="https://example.com/model.zip", save_path=None)

        # Verify that the pretrained inference was called
        mock_pretrained.assert_called_once()

        # Verify results structure
        assert isinstance(results, InferenceResults)
        assert len(results.predictions) == len(adata)
        assert results.metrics is not None
        assert "AUROC" in results.metrics

    def test_run_inference_no_data_input(self):
        """Test run_inference with no data input (should raise ValueError)."""
        with pytest.raises(ValueError, match="Either adata or data_path must be provided"):
            run_inference(model_path="fake_model_path")

    @patch("scxpand.core.inference.fetch_model_and_run_inference")
    def test_run_inference_no_model_source(self, mock_fetch, base_adata):
        """Test run_inference with no model source (should use default registry model)."""
        mock_fetch.return_value = InferenceResults(
            predictions=np.array([0.5] * len(base_adata)),
            model_info=None,
            metrics=None,
        )

        result = run_inference(adata=base_adata)

        # Should call fetch_model_and_run_inference with default model
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[1]["model_name"] == DEFAULT_MODEL_NAME
        assert call_args[1]["model_url"] is None
        assert result.predictions is not None

    def test_run_inference_multiple_model_sources(self, base_adata):
        """Test run_inference with multiple model sources (should raise ValueError)."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(
                adata=base_adata,
                model_path="fake_model_path",
                model_name=DEFAULT_MODEL_NAME,
            )

    @patch("scxpand.core.inference.run_prediction_pipeline")
    def test_run_inference_with_save_path(self, mock_pipeline, base_adata):
        """Test run_inference with save_path specified."""
        # Create adata with expansion column
        adata = self.create_adata_with_columns(base_adata, ["expansion"])

        # Mock the prediction pipeline
        mock_pipeline.return_value = InferenceResults(predictions=np.random.random(len(adata)), metrics={"AUROC": 0.85})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_results"

            # Run inference
            _results = run_inference(adata=adata, model_path="fake_model_path", save_path=save_path)

            # Verify that the pipeline was called with save_path
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert call_args[1]["save_path"] == save_path

    @patch("scxpand.core.inference.run_prediction_pipeline")
    def test_run_inference_with_eval_row_inds(self, mock_pipeline, base_adata):
        """Test run_inference with eval_row_inds specified."""
        # Create adata with expansion column
        adata = self.create_adata_with_columns(base_adata, ["expansion"])

        # Define subset of cells to evaluate
        eval_row_inds = np.array([0, 5, 10, 15, 20])

        # Mock the prediction pipeline
        mock_pipeline.return_value = InferenceResults(
            predictions=np.random.random(len(eval_row_inds)), metrics={"AUROC": 0.85}
        )

        # Run inference
        _results = run_inference(adata=adata, model_path="fake_model_path", eval_row_inds=eval_row_inds)

        # Verify that the pipeline was called with eval_row_inds
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        assert np.array_equal(call_args[1]["eval_row_inds"], eval_row_inds)

    @patch("scxpand.core.inference.run_prediction_pipeline")
    def test_run_inference_with_data_path(self, mock_pipeline):
        """Test run_inference with data_path instead of adata."""
        # Mock the prediction pipeline
        mock_pipeline.return_value = InferenceResults(predictions=np.random.random(50), metrics={"AUROC": 0.85})

        # Run inference with data_path
        _results = run_inference(data_path="fake_data.h5ad", model_path="fake_model_path")

        # Verify that the pipeline was called with data_path
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        assert call_args[1]["data_path"] == "fake_data.h5ad"
        assert call_args[1]["adata"] is None


class TestRunInferenceColumnValidation:
    """Test column validation logic in run_inference."""

    def test_extract_is_expanded_missing_expansion_column(self):
        """Test extract_is_expanded function with missing expansion column."""
        # Create DataFrame without expansion column
        obs_df = pd.DataFrame({"cell_id": ["cell_1", "cell_2", "cell_3"], "tissue_type": ["tumor", "normal", "tumor"]})

        # Should raise KeyError with informative message
        with pytest.raises(KeyError, match="'expansion' column not found"):
            extract_is_expanded(obs_df)

    def test_extract_is_expanded_with_expansion_column(self):
        """Test extract_is_expanded function with expansion column present."""
        # Create DataFrame with expansion column
        obs_df = pd.DataFrame(
            {"cell_id": ["cell_1", "cell_2", "cell_3"], "expansion": ["expanded", "non-expanded", "expanded"]}
        )

        # Should work correctly
        labels = extract_is_expanded(obs_df)
        expected = np.array([1, 0, 1])
        assert np.array_equal(labels, expected)

    def test_calculate_metrics_missing_stratification_columns(self):
        """Test calculate_metrics function with missing stratification columns."""
        # Create DataFrame without stratification columns
        obs_df = pd.DataFrame({"cell_id": ["cell_1", "cell_2", "cell_3"]})

        y_true = np.array([1, 0, 1])
        y_pred_prob = np.array([0.8, 0.3, 0.9])

        # Should work but skip stratified metrics
        results = calculate_metrics(y_true, y_pred_prob, obs_df)

        # Should have overall metrics
        assert "AUROC" in results
        assert "harmonic_avg" in results

        # Should not have stratified metrics
        assert not any("__" in key for key in results if key not in ["harmonic_avg", "average"])

    def test_calculate_metrics_with_stratification_columns(self):
        """Test calculate_metrics function with stratification columns present."""
        # Create DataFrame with stratification columns
        obs_df = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3", "cell_4"],
                "tissue_type": ["tumor", "tumor", "normal", "normal"],
                "imputed_labels": ["T_cell", "B_cell", "T_cell", "B_cell"],
            }
        )

        y_true = np.array([1, 0, 1, 0])
        y_pred_prob = np.array([0.8, 0.3, 0.9, 0.2])

        # Should work with stratified metrics
        results = calculate_metrics(y_true, y_pred_prob, obs_df)

        # Should have overall metrics
        assert "AUROC" in results
        assert "harmonic_avg" in results

        # Should have stratified metrics
        stratified_keys = [key for key in results if "__" in key]
        assert len(stratified_keys) > 0


if __name__ == "__main__":
    pytest.main([__file__])
