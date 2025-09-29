"""Tests for LightGBM inference preprocessing compatibility.

This module specifically tests that LightGBM inference uses the same
preprocessing pipeline as training, preventing the regression where
data_format was not used in inference.
"""

import inspect
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.data_util.data_format import DataFormat
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_inference


class TestLightGBMPreprocessingCompatibility:
    """Test LightGBM preprocessing compatibility between training and inference."""

    @pytest.fixture
    def sample_data_format(self):
        """Create a sample DataFormat for testing."""
        n_genes = 20
        return DataFormat(
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
            genes_mu=np.random.randn(n_genes).astype(np.float32),
            genes_sigma=np.random.rand(n_genes).astype(np.float32) + 0.1,
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

    @pytest.fixture
    def sample_adata(self, sample_data_format):
        """Create sample AnnData for testing."""
        n_cells = 15
        n_genes = sample_data_format.n_genes

        # Create realistic gene expression data
        X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 8 + ["non-expanded"] * 7,
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=sample_data_format.gene_names)

        return ad.AnnData(X=X, obs=obs_df, var=var_df)

    def test_inference_uses_data_format_parameter(
        self, sample_adata, sample_data_format
    ):
        """Test that LightGBM inference accepts and uses the data_format parameter."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(15, 2)

        # Test in-memory data processing
        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data"
        ) as mock_preprocess:
            mock_preprocess.return_value = np.random.randn(15, 20)

            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=None,
            )

            # Verify preprocessing was called with data_format
            mock_preprocess.assert_called_once()
            call_args = mock_preprocess.call_args
            assert call_args.kwargs["data_format"] is sample_data_format

            # Verify result shape
            assert isinstance(result, np.ndarray)
            assert len(result) == 15

    def test_file_based_preprocessing(self, tmp_path, sample_adata, sample_data_format):
        """Test that file-based preprocessing uses the utility function."""
        # Save test data to file
        file_path = tmp_path / "test_data.h5ad"
        sample_adata.write_h5ad(file_path)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(15, 2)

        with patch(
            "scxpand.lightgbm.run_lightgbm_._prepare_data_for_lightgbm_inference"
        ) as mock_prepare:
            mock_prepare.return_value = np.random.randn(15, 20)

            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=None,
                data_path=file_path,
                eval_row_inds=None,
            )

            # Verify the utility function was called with correct parameters
            mock_prepare.assert_called_once_with(
                data_format=sample_data_format,
                adata=None,
                data_path=file_path,
                eval_row_inds=None,
            )

            assert isinstance(result, np.ndarray)
            assert len(result) == 15

    def test_preprocessing_with_eval_row_inds(self, sample_adata, sample_data_format):
        """Test preprocessing with specific row indices."""
        mock_model = MagicMock()
        eval_indices = np.array([0, 2, 4, 6, 8])
        mock_model.predict_proba.return_value = np.random.rand(len(eval_indices), 2)

        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data"
        ) as mock_preprocess:
            mock_preprocess.return_value = np.random.randn(len(eval_indices), 20)

            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=eval_indices,
            )

            # Verify preprocessing was called
            mock_preprocess.assert_called_once()
            call_args = mock_preprocess.call_args

            # Check that the correct subset of data was passed
            # The X parameter could be positional or keyword
            if call_args.args:
                X_raw = call_args.args[0]  # First positional argument
            else:
                X_raw = call_args.kwargs["X"]  # Keyword argument
            assert X_raw.shape[0] == len(eval_indices)

            assert isinstance(result, np.ndarray)
            assert len(result) == len(eval_indices)

    def test_preprocessing_pipeline_consistency(self, sample_adata, sample_data_format):
        """Test that the preprocessing pipeline steps are applied in the correct order."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(15, 2)

        # Mock the preprocessing function to capture the input
        def mock_preprocess_func(X, data_format):
            # Verify the data_format is passed correctly
            assert data_format is sample_data_format
            assert data_format.use_log_transform is True
            assert data_format.use_zscore_norm is True
            assert data_format.target_sum == 1e4

            # Return processed data with correct shape
            return np.random.randn(X.shape[0], data_format.n_genes)

        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data",
            side_effect=mock_preprocess_func,
        ):
            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=None,
            )

            assert isinstance(result, np.ndarray)
            assert len(result) == 15

    def test_model_prediction_consistency(self, sample_adata, sample_data_format):
        """Test that model prediction is consistent with training validation."""
        mock_model = MagicMock()

        # Mock predict_proba to return realistic probabilities
        n_samples = 15
        mock_probs = np.column_stack(
            [
                np.random.beta(2, 5, n_samples),  # Probability for class 0
                np.random.beta(5, 2, n_samples),  # Probability for class 1
            ]
        )
        mock_probs = mock_probs / mock_probs.sum(axis=1, keepdims=True)  # Normalize
        mock_model.predict_proba.return_value = mock_probs

        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data"
        ) as mock_preprocess:
            mock_preprocess.return_value = np.random.randn(n_samples, 20)

            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=None,
            )

            # Verify model.predict_proba was called
            mock_model.predict_proba.assert_called_once()

            # Verify result is the positive class probabilities (index 1)
            expected_result = mock_probs[:, 1]
            np.testing.assert_array_equal(result, expected_result)

    def test_svm_model_compatibility(self, sample_adata, sample_data_format):
        """Test that SVM models (using decision_function) work correctly."""
        mock_model = MagicMock()

        # Mock SVM model (no predict_proba, uses decision_function)
        del mock_model.predict_proba  # Remove predict_proba method
        mock_model.loss = "hinge"  # SVM uses hinge loss

        n_samples = 15
        decision_scores = np.random.randn(n_samples)
        mock_model.decision_function.return_value = decision_scores

        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data"
        ) as mock_preprocess:
            mock_preprocess.return_value = np.random.randn(n_samples, 20)

            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=None,
            )

            # Verify decision_function was called
            mock_model.decision_function.assert_called_once()

            # Verify sigmoid transformation was applied
            expected_result = 1 / (1 + np.exp(-decision_scores))
            np.testing.assert_array_almost_equal(result, expected_result)

    def test_error_handling_missing_data(self, sample_data_format):
        """Test error handling when both adata and data_path are None."""
        mock_model = MagicMock()

        # This should raise ValueError for missing data
        with pytest.raises(
            ValueError, match="Either adata or data_path must be provided"
        ):
            run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=None,
                data_path=None,
                eval_row_inds=None,
            )

    def test_data_format_validation(self, sample_adata):
        """Test that data_format parameter is properly validated."""
        mock_model = MagicMock()

        # Test with None data_format (should raise error or handle gracefully)
        with pytest.raises((TypeError, AttributeError)):
            run_lightgbm_inference(
                model=mock_model,
                data_format=None,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=None,
            )

    def test_preprocessing_parameter_types(self, sample_adata, sample_data_format):
        """Test that preprocessing functions receive correct parameter types."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(15, 2)

        def validate_preprocess_params(X, data_format):
            # Validate parameter types
            assert isinstance(X, np.ndarray), f"X should be numpy array, got {type(X)}"
            assert isinstance(data_format, DataFormat), (
                f"data_format should be DataFormat, got {type(data_format)}"
            )

            # Validate X properties
            assert X.ndim == 2, f"X should be 2D, got {X.ndim}D"
            assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"

            return np.random.randn(X.shape[0], data_format.n_genes)

        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data",
            side_effect=validate_preprocess_params,
        ):
            result = run_lightgbm_inference(
                model=mock_model,
                data_format=sample_data_format,
                adata=sample_adata,
                data_path=None,
                eval_row_inds=None,
            )

            assert isinstance(result, np.ndarray)


class TestRegressionPrevention:
    """Specific regression tests for previously fixed issues."""

    def test_data_format_parameter_acceptance(self):
        """Regression test: Ensure run_lightgbm_inference accepts data_format parameter."""
        sig = inspect.signature(run_lightgbm_inference)
        params = sig.parameters

        # The function MUST accept data_format parameter
        assert "data_format" in params, (
            "run_lightgbm_inference must accept data_format parameter"
        )

        # data_format should be a required parameter (not optional with None default)
        data_format_param = params["data_format"]
        assert data_format_param.default == inspect.Parameter.empty, (
            "data_format should be required"
        )

    def test_preprocessing_function_calls(self, tmp_path):
        """Regression test: Ensure preprocessing functions are actually called."""
        # Create minimal test data
        n_cells, n_genes = 5, 10
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        adata = ad.AnnData(X=X)

        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
            genes_mu=np.zeros(n_genes),
            genes_sigma=np.ones(n_genes),
            use_log_transform=True,
            use_zscore_norm=True,
        )

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(n_cells, 2)

        # Test in-memory preprocessing
        with patch(
            "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data"
        ) as mock_preprocess:
            mock_preprocess.return_value = X

            run_lightgbm_inference(
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=None,
            )

            # MUST be called - this was the bug
            mock_preprocess.assert_called_once()

        # Test file-based preprocessing
        file_path = tmp_path / "test.h5ad"
        adata.write_h5ad(file_path)

        with patch(
            "scxpand.lightgbm.run_lightgbm_._prepare_data_for_lightgbm_inference"
        ) as mock_prepare:
            mock_prepare.return_value = X

            run_lightgbm_inference(
                model=mock_model,
                data_format=data_format,
                adata=None,
                data_path=file_path,
                eval_row_inds=None,
            )

            # MUST be called - this was the bug
            mock_prepare.assert_called_once()

    def test_no_type_error_on_data_format(self):
        """Regression test: Ensure no TypeError about unexpected keyword argument 'data_format'."""
        # Create minimal test setup
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        X = np.random.randn(2, 5).astype(np.float32)
        adata = ad.AnnData(X=X)

        data_format = DataFormat(
            n_genes=5,
            gene_names=[f"gene_{i}" for i in range(5)],
            genes_mu=np.zeros(5),
            genes_sigma=np.ones(5),
        )

        # This should NOT raise TypeError about unexpected keyword argument
        try:
            with patch(
                "scxpand.lightgbm.run_lightgbm_.preprocess_expression_data"
            ) as mock_preprocess:
                mock_preprocess.return_value = X

                result = run_lightgbm_inference(
                    model=mock_model,
                    data_format=data_format,
                    adata=adata,
                    data_path=None,
                    eval_row_inds=None,
                )

                assert isinstance(result, np.ndarray)
                assert len(result) == 2

        except TypeError as e:
            if "unexpected keyword argument" in str(e) and "data_format" in str(e):
                pytest.fail(f"Regression: TypeError about data_format parameter: {e}")
            else:
                raise  # Re-raise if it's a different TypeError

    def test_gene_mismatch_handling(self, tmp_path):
        """Test that LightGBM inference handles gene mismatches correctly."""
        # Create a data format with specific genes (simulating trained model)
        model_genes = [f"gene_{i}" for i in range(10)]  # Model expects 10 genes
        data_format = DataFormat(
            n_genes=len(model_genes),
            gene_names=model_genes,
            genes_mu=np.zeros(len(model_genes), dtype=np.float32),
            genes_sigma=np.ones(len(model_genes), dtype=np.float32),
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

        # Create test data with different genes (simulating new data)
        test_genes = [
            f"gene_{i}" for i in range(5, 15)
        ]  # 5 overlapping, 5 new, 5 missing
        n_cells = 20
        X = np.random.exponential(scale=2.0, size=(n_cells, len(test_genes))).astype(
            np.float32
        )

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 10 + ["non-expanded"] * 10,
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=test_genes)

        test_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(n_cells, 2)

        # Test in-memory inference with gene mismatch
        result = run_lightgbm_inference(
            model=mock_model,
            data_format=data_format,
            adata=test_adata,
            data_path=None,
            eval_row_inds=None,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_cells,), (
            f"Expected shape ({n_cells},), got {result.shape}"
        )

        # Test file-based inference with gene mismatch
        file_path = tmp_path / "test_gene_mismatch.h5ad"
        test_adata.write_h5ad(file_path)

        result = run_lightgbm_inference(
            model=mock_model,
            data_format=data_format,
            adata=None,
            data_path=file_path,
            eval_row_inds=None,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_cells,), (
            f"Expected shape ({n_cells},), got {result.shape}"
        )
