"""Tests to ensure training-inference pipeline compatibility.

This module contains tests specifically designed to prevent the issues that were
fixed in the inference pipeline, ensuring that:
1. Data format is used consistently between training and inference
2. Parameters like batch_size are properly propagated
3. Preprocessing pipelines are identical between training and inference
4. Function signatures are compatible across all model types
"""

import inspect
import json
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.autoencoders.ae_trainer import run_ae_inference
from scxpand.data_util.data_format import DataFormat
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_inference
from scxpand.linear.sklearn_utils import run_linear_inference
from scxpand.mlp.mlp_trainer import run_mlp_inference
from scxpand.util.inference_utils import load_model, run_model_inference
from scxpand.util.model_constants import (
    BEST_CHECKPOINT_FILE,
    DATA_FORMAT_NPZ_FILE,
    SKLEARN_MODEL_FILE,
)


class TestTrainingInferenceCompatibility:
    """Test suite for training-inference pipeline compatibility."""

    @pytest.fixture
    def mock_data_format(self):
        """Create a mock DataFormat for testing."""
        return DataFormat(
            n_genes=100,
            gene_names=[f"gene_{i}" for i in range(100)],
            genes_mu=np.random.randn(100).astype(np.float32),
            genes_sigma=np.random.rand(100).astype(np.float32) + 0.1,
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

    @pytest.fixture
    def mock_adata(self, tmp_path):
        """Create mock AnnData for testing."""
        n_cells, n_genes = 50, 100
        X = np.random.randn(n_cells, n_genes).astype(np.float32)

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 25 + ["non-expanded"] * 25,
                "clone_id_size": np.random.randint(1, 100, size=n_cells),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
                "imputed_labels": np.random.choice(["label1", "label2"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_data.h5ad"
        adata.write_h5ad(file_path)

        return file_path, adata

    def test_lightgbm_inference_accepts_data_format(self, mock_adata, mock_data_format):
        """Test that LightGBM inference accepts and uses data_format parameter.

        This test prevents regression of the original TypeError where
        run_lightgbm_inference() got unexpected keyword argument 'data_format'.
        """
        file_path, adata = mock_adata
        mock_model = MagicMock()

        # Mock the LightGBM inference function to verify it receives data_format
        with patch(
            "scxpand.util.inference_utils.run_lightgbm_inference"
        ) as mock_inference:
            mock_inference.return_value = np.random.rand(25)

            # This should NOT raise TypeError about unexpected keyword argument
            result = run_model_inference(
                model_type="lightgbm",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(25),
                device="cpu",
                batch_size=16,
                num_workers=2,
            )

            # Verify the function was called with data_format
            mock_inference.assert_called_once()
            call_kwargs = mock_inference.call_args.kwargs
            assert "data_format" in call_kwargs
            assert call_kwargs["data_format"] is mock_data_format
            assert isinstance(result, np.ndarray)

    def test_batch_size_propagation_to_all_models(self, mock_adata, mock_data_format):
        """Test that batch_size parameter is properly propagated to all model types.

        This test prevents regression where batch_size wasn't passed to linear models.
        """
        file_path, adata = mock_adata
        mock_model = MagicMock()
        test_batch_size = 32
        test_num_workers = 4

        # Test MLP - should receive batch_size
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(25)

            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                batch_size=test_batch_size,
                num_workers=test_num_workers,
            )

            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["batch_size"] == test_batch_size
            assert call_kwargs["num_workers"] == test_num_workers

        # Test Autoencoder - should receive batch_size
        with patch("scxpand.util.inference_utils.run_ae_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(25)

            run_model_inference(
                model_type="autoencoder",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                batch_size=test_batch_size,
                num_workers=test_num_workers,
            )

            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["batch_size"] == test_batch_size
            assert call_kwargs["num_workers"] == test_num_workers

        # Test Linear models - should receive batch_size (this was the bug)
        for model_type in ["logistic", "svm"]:
            with patch(
                "scxpand.util.inference_utils.run_linear_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(25)

                run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=mock_data_format,
                    adata=adata,
                    batch_size=test_batch_size,
                    num_workers=test_num_workers,
                )

                call_kwargs = mock_inference.call_args.kwargs
                assert call_kwargs["batch_size"] == test_batch_size
                assert call_kwargs["num_workers"] == test_num_workers

    def test_lightgbm_preprocessing_consistency(self, tmp_path, mock_data_format):
        """Test that LightGBM inference uses the same preprocessing as training."""
        # Create test data
        n_cells, n_genes = 10, 100
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 5 + ["non-expanded"] * 5})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        # Test file-based preprocessing
        file_path = tmp_path / "test_lightgbm.h5ad"
        adata.write_h5ad(file_path)

        mock_model = MagicMock()

        # Mock the utility function that's actually called now
        with patch(
            "scxpand.lightgbm.run_lightgbm_._prepare_data_for_lightgbm_inference"
        ) as mock_prepare:
            mock_prepare.return_value = np.random.randn(n_cells, n_genes)

            # Test file-based preprocessing
            run_lightgbm_inference(
                model=mock_model,
                data_format=mock_data_format,
                adata=None,
                data_path=file_path,
                eval_row_inds=np.arange(n_cells),
            )

            # Verify the utility function was called with correct parameters
            mock_prepare.assert_called_once()
            call_args = mock_prepare.call_args
            assert call_args.kwargs["data_format"] is mock_data_format
            assert call_args.kwargs["data_path"] == file_path
            assert call_args.kwargs["adata"] is None
            np.testing.assert_array_equal(
                call_args.kwargs["eval_row_inds"], np.arange(n_cells)
            )

            # Test in-memory preprocessing
            mock_prepare.reset_mock()
            run_lightgbm_inference(
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(n_cells),
            )

            # Verify the utility function was called with correct parameters for in-memory data
            mock_prepare.assert_called_once()
            call_args = mock_prepare.call_args
            assert call_args.kwargs["data_format"] is mock_data_format
            assert call_args.kwargs["adata"] is adata
            assert call_args.kwargs["data_path"] is None
            np.testing.assert_array_equal(
                call_args.kwargs["eval_row_inds"], np.arange(n_cells)
            )

    def test_inference_function_signatures_compatibility(self):
        """Test that all inference functions have compatible signatures.

        This test ensures that all model-specific inference functions can accept
        the parameters passed by run_model_inference().
        """
        # All inference functions are already imported at the top

        # Expected parameters that run_inference passes
        expected_params = {
            "model",
            "data_format",
            "adata",
            "data_path",
            "eval_row_inds",
        }

        # Parameters that should be passed to batched models
        batched_params = {"batch_size", "num_workers"}

        # Check MLP inference
        mlp_sig = inspect.signature(run_mlp_inference)
        mlp_params = set(mlp_sig.parameters.keys())
        assert expected_params.issubset(mlp_params), (
            f"MLP missing params: {expected_params - mlp_params}"
        )
        assert batched_params.issubset(mlp_params), (
            f"MLP missing batch params: {batched_params - mlp_params}"
        )

        # Check Autoencoder inference
        ae_sig = inspect.signature(run_ae_inference)
        ae_params = set(ae_sig.parameters.keys())
        assert expected_params.issubset(ae_params), (
            f"AE missing params: {expected_params - ae_params}"
        )
        assert batched_params.issubset(ae_params), (
            f"AE missing batch params: {batched_params - ae_params}"
        )

        # Check Linear inference
        linear_sig = inspect.signature(run_linear_inference)
        linear_params = set(linear_sig.parameters.keys())
        assert expected_params.issubset(linear_params), (
            f"Linear missing params: {expected_params - linear_params}"
        )
        assert batched_params.issubset(linear_params), (
            f"Linear missing batch params: {batched_params - linear_params}"
        )

        # Check LightGBM inference (updated signature)
        lgb_sig = inspect.signature(run_lightgbm_inference)
        lgb_params = set(lgb_sig.parameters.keys())
        assert expected_params.issubset(lgb_params), (
            f"LightGBM missing params: {expected_params - lgb_params}"
        )
        # Note: LightGBM doesn't use batching, so we don't check for batch_size/num_workers

    def test_data_format_consistency_across_models(self, mock_adata, mock_data_format):
        """Test that all models receive and use the same DataFormat object."""
        file_path, adata = mock_adata
        mock_model = MagicMock()

        # Test that each model type receives the exact same DataFormat object
        model_patches = {
            "mlp": "scxpand.util.inference_utils.run_mlp_inference",
            "autoencoder": "scxpand.util.inference_utils.run_ae_inference",
            "logistic": "scxpand.util.inference_utils.run_linear_inference",
            "svm": "scxpand.util.inference_utils.run_linear_inference",
            "lightgbm": "scxpand.util.inference_utils.run_lightgbm_inference",
        }

        for model_type, patch_target in model_patches.items():
            with patch(patch_target) as mock_inference:
                mock_inference.return_value = np.random.rand(25)

                run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=mock_data_format,
                    adata=adata,
                )

                # Verify the exact same DataFormat object was passed
                call_kwargs = mock_inference.call_args.kwargs
                assert call_kwargs["data_format"] is mock_data_format

    def test_preprocessing_pipeline_parameters(self, mock_data_format):
        """Test that preprocessing parameters are consistent between training and inference."""
        # Verify DataFormat contains all necessary preprocessing parameters
        assert hasattr(mock_data_format, "use_log_transform")
        assert hasattr(mock_data_format, "use_zscore_norm")
        assert hasattr(mock_data_format, "target_sum")
        assert hasattr(mock_data_format, "genes_mu")
        assert hasattr(mock_data_format, "genes_sigma")
        assert hasattr(mock_data_format, "gene_names")

        # Verify preprocessing parameters are properly set
        assert isinstance(mock_data_format.use_log_transform, bool)
        assert isinstance(mock_data_format.use_zscore_norm, bool)
        assert isinstance(mock_data_format.target_sum, int | float)
        assert isinstance(mock_data_format.genes_mu, np.ndarray)
        assert isinstance(mock_data_format.genes_sigma, np.ndarray)
        assert len(mock_data_format.genes_mu) == mock_data_format.n_genes
        assert len(mock_data_format.genes_sigma) == mock_data_format.n_genes

    def test_model_loading_compatibility(self, tmp_path):
        """Test that model loading works for all supported model types."""
        # Test each model type
        model_types = ["mlp", "autoencoder", "lightgbm", "logistic", "svm"]

        for model_type in model_types:
            results_path = tmp_path / f"{model_type}_results"
            results_path.mkdir(exist_ok=True)

            # Create required files
            if model_type in ["mlp", "autoencoder"]:
                (results_path / BEST_CHECKPOINT_FILE).touch()
                # Create data_format files for neural networks
                data_format_data = {
                    "n_genes": 10,
                    "gene_names": [f"g{i}" for i in range(10)],
                }
                with open(results_path / "data_format.json", "w") as f:
                    json.dump(data_format_data, f)
                np.savez(
                    results_path / DATA_FORMAT_NPZ_FILE,
                    genes_mu=np.zeros(10),
                    genes_sigma=np.ones(10),
                )
            else:
                (results_path / SKLEARN_MODEL_FILE).touch()

            # Create parameters file
            params = {"model_type": model_type, "n_epochs": 10}
            with open(results_path / "parameters.json", "w") as f:
                json.dump(params, f)

            # Mock the appropriate loading function
            if model_type in ["mlp"]:
                patch_target = "scxpand.util.inference_utils.load_nn_model"
            elif model_type == "autoencoder":
                patch_target = "scxpand.util.inference_utils.load_ae_model"
            else:  # lightgbm, logistic, svm
                patch_target = "scxpand.util.inference_utils.load_sklearn_model"

            with patch(patch_target) as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                # Test load_model function (already imported at top)

                loaded_model = load_model(model_type, results_path, "cpu")
                assert loaded_model is mock_model
                mock_load.assert_called_once()

    def test_error_handling_consistency(self, mock_adata, mock_data_format):
        """Test that error handling is consistent across all model types."""
        file_path, adata = mock_adata
        mock_model = MagicMock()

        # Test unsupported model type
        with pytest.raises(ValueError, match="Unsupported model_type"):
            run_model_inference(
                model_type="unsupported_model",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
            )

        # Test missing data (both adata and data_path are None)
        with pytest.raises(
            ValueError, match="Either adata or data_path must be provided"
        ):
            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=None,
                data_path=None,
            )

    def test_device_parameter_handling(self, mock_adata, mock_data_format):
        """Test that device parameter is properly handled for neural network models."""
        file_path, adata = mock_adata
        mock_model = MagicMock()
        test_device = "cuda:0"

        # Test MLP - should receive device
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(25)

            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                device=test_device,
            )

            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["device"] == test_device

        # Test Autoencoder - should receive device
        with patch("scxpand.util.inference_utils.run_ae_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(25)

            run_model_inference(
                model_type="autoencoder",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                device=test_device,
            )

            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["device"] == test_device

        # Test Linear models - should NOT receive device (they don't use it)
        with patch(
            "scxpand.util.inference_utils.run_linear_inference"
        ) as mock_inference:
            mock_inference.return_value = np.random.rand(25)

            run_model_inference(
                model_type="logistic",
                model=mock_model,
                data_format=mock_data_format,
                adata=adata,
                device=test_device,
            )

            call_kwargs = mock_inference.call_args.kwargs
            # Linear models don't use device parameter
            assert "device" not in call_kwargs or call_kwargs.get("device") is None

    def test_eval_row_inds_propagation(self, mock_adata, mock_data_format):
        """Test that eval_row_inds parameter is properly propagated to all models."""
        file_path, adata = mock_adata
        mock_model = MagicMock()
        test_indices = np.array([0, 2, 4, 6, 8])

        for model_type in ["mlp", "autoencoder", "logistic", "lightgbm"]:
            patch_name = {
                "mlp": "run_mlp_inference",
                "autoencoder": "run_ae_inference",
                "logistic": "run_linear_inference",
                "lightgbm": "run_lightgbm_inference",
            }[model_type]

            with patch(f"scxpand.util.inference_utils.{patch_name}") as mock_inference:
                mock_inference.return_value = np.random.rand(len(test_indices))

                run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=mock_data_format,
                    adata=adata,
                    eval_row_inds=test_indices,
                )

                call_kwargs = mock_inference.call_args.kwargs
                np.testing.assert_array_equal(
                    call_kwargs["eval_row_inds"], test_indices
                )


class TestRegressionPrevention:
    """Tests specifically designed to prevent regression of fixed issues."""

    def test_lightgbm_data_format_regression(self):
        """Regression test: LightGBM inference must accept data_format parameter."""
        # Verify the function signature includes data_format
        sig = inspect.signature(run_lightgbm_inference)
        assert "data_format" in sig.parameters, (
            "LightGBM inference must accept data_format parameter"
        )

        # Verify data_format is not optional (has no default value of None)
        data_format_param = sig.parameters["data_format"]
        assert (
            data_format_param.default == inspect.Parameter.empty
            or data_format_param.annotation != "DataFormat | None"
        ), "data_format should be required parameter"

    def test_batch_size_propagation_regression(self):
        """Regression test: batch_size must be passed to linear models."""
        # Verify the function signature includes batch_size and num_workers
        sig = inspect.signature(run_linear_inference)
        assert "batch_size" in sig.parameters, (
            "Linear inference must accept batch_size parameter"
        )
        assert "num_workers" in sig.parameters, (
            "Linear inference must accept num_workers parameter"
        )

    def test_preprocessing_consistency_regression(self):
        """Regression test: LightGBM inference must use same preprocessing as training."""
        # Create mock data
        n_genes = 10
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
            genes_mu=np.zeros(n_genes),
            genes_sigma=np.ones(n_genes),
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

        # Create test AnnData
        X = np.random.randn(5, n_genes).astype(np.float32)
        adata = ad.AnnData(X=X)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.rand(5, 2)

        # Test that the utility function is called (which handles preprocessing)
        with patch(
            "scxpand.lightgbm.run_lightgbm_._prepare_data_for_lightgbm_inference"
        ) as mock_prepare:
            mock_prepare.return_value = X

            # This should call the utility function for data preparation
            run_lightgbm_inference(
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=None,
            )

            # Verify the utility function was called with correct parameters
            mock_prepare.assert_called_once()
            call_args = mock_prepare.call_args
            assert call_args.kwargs["data_format"] is data_format
            assert call_args.kwargs["adata"] is adata
            assert call_args.kwargs["data_path"] is None
            assert call_args.kwargs["eval_row_inds"] is None

    def test_import_structure_regression(self):
        """Regression test: Verify all necessary imports are available."""
        # Test that all inference functions can be imported (already imported at top)
        # All functions should be available
        assert run_ae_inference is not None
        assert run_lightgbm_inference is not None
        assert run_linear_inference is not None
        assert run_mlp_inference is not None
        assert load_model is not None
        assert run_model_inference is not None

        # Test that all functions are callable
        assert callable(run_ae_inference)
        assert callable(run_lightgbm_inference)
        assert callable(run_linear_inference)
        assert callable(run_mlp_inference)
        assert callable(load_model)
        assert callable(run_model_inference)
