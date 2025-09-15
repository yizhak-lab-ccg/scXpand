"""Tests for parameter propagation in inference pipeline.

This module tests that parameters are correctly passed through the inference
pipeline from the high-level run_inference function to the model-specific
inference functions.
"""

import inspect

from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.autoencoders.ae_trainer import run_ae_inference
from scxpand.core.inference import run_inference
from scxpand.data_util.data_format import DataFormat
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_inference
from scxpand.linear.sklearn_utils import run_linear_inference
from scxpand.mlp.mlp_trainer import run_mlp_inference
from scxpand.util.inference_utils import run_model_inference


class TestParameterPropagation:
    """Test parameter propagation through inference pipeline."""

    @pytest.fixture
    def mock_data_format(self):
        """Create a mock DataFormat for testing."""
        return DataFormat(
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
            genes_mu=np.zeros(50, dtype=np.float32),
            genes_sigma=np.ones(50, dtype=np.float32),
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

    @pytest.fixture
    def mock_adata(self):
        """Create a minimal mock AnnData object."""
        n_cells, n_genes = 20, 50
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 10 + ["non-expanded"] * 10})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        return ad.AnnData(X=X, obs=obs_df, var=var_df)

    def test_mlp_parameter_propagation(self, mock_adata, mock_data_format):
        """Test that all parameters are correctly passed to MLP inference."""
        mock_model = MagicMock()
        test_params = {
            "model_type": "mlp",
            "model": mock_model,
            "data_format": mock_data_format,
            "adata": mock_adata,
            "data_path": None,
            "device": "cuda:0",
            "batch_size": 64,
            "num_workers": 4,
            "eval_row_inds": np.array([0, 1, 2, 3, 4]),
        }

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(5)

            run_model_inference(**test_params)

            # Verify all expected parameters were passed
            mock_inference.assert_called_once()
            call_kwargs = mock_inference.call_args.kwargs

            assert call_kwargs["model"] is mock_model
            assert call_kwargs["data_format"] is mock_data_format
            assert call_kwargs["adata"] is mock_adata
            assert call_kwargs["data_path"] is None
            assert call_kwargs["device"] == "cuda:0"
            assert call_kwargs["batch_size"] == 64
            assert call_kwargs["num_workers"] == 4
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], test_params["eval_row_inds"])

    def test_autoencoder_parameter_propagation(self, mock_adata, mock_data_format):
        """Test that all parameters are correctly passed to Autoencoder inference."""
        mock_model = MagicMock()
        test_params = {
            "model_type": "autoencoder",
            "model": mock_model,
            "data_format": mock_data_format,
            "adata": mock_adata,
            "data_path": None,
            "device": "cpu",
            "batch_size": 128,
            "num_workers": 2,
            "eval_row_inds": np.array([5, 6, 7]),
        }

        with patch("scxpand.util.inference_utils.run_ae_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(3)

            run_model_inference(**test_params)

            # Verify all expected parameters were passed
            mock_inference.assert_called_once()
            call_kwargs = mock_inference.call_args.kwargs

            assert call_kwargs["model"] is mock_model
            assert call_kwargs["data_format"] is mock_data_format
            assert call_kwargs["adata"] is mock_adata
            assert call_kwargs["data_path"] is None
            assert call_kwargs["device"] == "cpu"
            assert call_kwargs["batch_size"] == 128
            assert call_kwargs["num_workers"] == 2
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], test_params["eval_row_inds"])

    def test_linear_parameter_propagation(self, mock_adata, mock_data_format):
        """Test that all parameters are correctly passed to Linear inference."""
        mock_model = MagicMock()

        for model_type in ["logistic", "svm"]:
            test_params = {
                "model_type": model_type,
                "model": mock_model,
                "data_format": mock_data_format,
                "adata": mock_adata,
                "data_path": None,
                "batch_size": 256,
                "num_workers": 8,
                "eval_row_inds": np.array([1, 3, 5, 7, 9]),
            }

            with patch("scxpand.util.inference_utils.run_linear_inference") as mock_inference:
                mock_inference.return_value = np.random.rand(5)

                run_model_inference(**test_params)

                # Verify all expected parameters were passed
                mock_inference.assert_called_once()
                call_kwargs = mock_inference.call_args.kwargs

                assert call_kwargs["model"] is mock_model
                assert call_kwargs["data_format"] is mock_data_format
                assert call_kwargs["adata"] is mock_adata
                assert call_kwargs["data_path"] is None
                assert call_kwargs["batch_size"] == 256
                assert call_kwargs["num_workers"] == 8
                np.testing.assert_array_equal(call_kwargs["eval_row_inds"], test_params["eval_row_inds"])

    def test_lightgbm_parameter_propagation(self, mock_adata, mock_data_format):
        """Test that all parameters are correctly passed to LightGBM inference."""
        mock_model = MagicMock()
        test_params = {
            "model_type": "lightgbm",
            "model": mock_model,
            "data_format": mock_data_format,
            "adata": mock_adata,
            "data_path": None,
            "eval_row_inds": np.array([0, 2, 4, 6]),
        }

        with patch("scxpand.util.inference_utils.run_lightgbm_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(4)

            run_model_inference(**test_params)

            # Verify all expected parameters were passed
            mock_inference.assert_called_once()
            call_kwargs = mock_inference.call_args.kwargs

            assert call_kwargs["model"] is mock_model
            assert call_kwargs["data_format"] is mock_data_format
            assert call_kwargs["adata"] is mock_adata
            assert call_kwargs["data_path"] is None
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], test_params["eval_row_inds"])
            # Note: LightGBM doesn't use batch_size/num_workers, so we don't check for them

    def test_parameter_defaults_handling(self, mock_adata, mock_data_format):
        """Test that default parameter values are handled correctly."""
        mock_model = MagicMock()

        # Test with minimal parameters (using defaults)
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(20)

            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
            )

            call_kwargs = mock_inference.call_args.kwargs

            # Verify default values are used
            assert call_kwargs["data_path"] is None
            assert call_kwargs["device"] is None
            assert call_kwargs["batch_size"] == 1024  # Default from run_inference
            assert call_kwargs["num_workers"] == 0  # Default from run_inference
            assert call_kwargs["eval_row_inds"] is None

    def test_file_path_parameter_propagation(self, tmp_path, mock_data_format):
        """Test parameter propagation when using file path instead of adata."""
        # Create test file
        n_cells, n_genes = 15, 50
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 8 + ["non-expanded"] * 7})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        file_path = tmp_path / "test_data.h5ad"
        adata.write_h5ad(file_path)

        mock_model = MagicMock()
        test_params = {
            "model_type": "mlp",
            "model": mock_model,
            "data_format": mock_data_format,
            "adata": None,
            "data_path": file_path,
            "batch_size": 32,
            "num_workers": 1,
        }

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(15)

            run_model_inference(**test_params)

            call_kwargs = mock_inference.call_args.kwargs

            assert call_kwargs["adata"] is None
            assert call_kwargs["data_path"] == file_path
            assert call_kwargs["batch_size"] == 32
            assert call_kwargs["num_workers"] == 1


class TestFunctionSignatureCompatibility:
    """Test that function signatures are compatible across the inference pipeline."""

    def test_run_inference_signature(self):
        """Test that run_inference has the expected signature."""
        sig = inspect.signature(run_inference)
        params = sig.parameters

        # Model source parameters (one required)
        assert "model_path" in params
        assert "model_name" in params
        assert "model_url" in params

        # Data parameters (one required)
        assert "adata" in params
        assert "data_path" in params

        # Optional parameters with defaults
        assert "save_path" in params
        assert "batch_size" in params
        assert "num_workers" in params
        assert "eval_row_inds" in params

        # Check default values
        assert params["adata"].default is None
        assert params["data_path"].default is None
        assert params["model_path"].default is None
        assert params["model_name"].default is None
        assert params["model_url"].default is None
        assert params["save_path"].default is None
        assert params["device"].default is None
        assert params["batch_size"].default == 1024
        assert params["num_workers"].default == 4
        assert params["eval_row_inds"].default is None

    def test_model_specific_inference_signatures(self):
        """Test that model-specific inference functions have compatible signatures."""
        # All inference functions are already imported at the top

        # Common parameters that should be in all functions
        common_params = {"model", "data_format", "adata", "eval_row_inds"}

        # Parameters for batched models
        batched_params = {"batch_size", "num_workers"}

        # Test MLP
        mlp_sig = inspect.signature(run_mlp_inference)
        mlp_params = set(mlp_sig.parameters.keys())
        assert common_params.issubset(mlp_params)
        assert batched_params.issubset(mlp_params)
        assert "device" in mlp_params

        # Test Autoencoder
        ae_sig = inspect.signature(run_ae_inference)
        ae_params = set(ae_sig.parameters.keys())
        assert common_params.issubset(ae_params)
        assert batched_params.issubset(ae_params)
        assert "device" in ae_params

        # Test Linear
        linear_sig = inspect.signature(run_linear_inference)
        linear_params = set(linear_sig.parameters.keys())
        assert common_params.issubset(linear_params)
        assert batched_params.issubset(linear_params)
        # Linear models don't use device

        # Test LightGBM
        lgb_sig = inspect.signature(run_lightgbm_inference)
        lgb_params = set(lgb_sig.parameters.keys())
        assert common_params.issubset(lgb_params)
        assert "data_path" in lgb_params  # LightGBM supports both adata and data_path
        # LightGBM doesn't use batching or device

    def test_parameter_type_consistency(self):
        """Test that parameter types are consistent across functions."""
        # All inference functions are already imported at the top

        functions = [
            ("mlp", run_mlp_inference),
            ("autoencoder", run_ae_inference),
            ("linear", run_linear_inference),
            ("lightgbm", run_lightgbm_inference),
        ]

        for func_name, func in functions:
            sig = inspect.signature(func)
            params = sig.parameters

            # Check that model parameter exists
            assert "model" in params, f"{func_name} missing model parameter"

            # Check that data_format parameter exists and has correct annotation
            assert "data_format" in params, f"{func_name} missing data_format parameter"

            # Check that eval_row_inds parameter exists
            assert "eval_row_inds" in params, f"{func_name} missing eval_row_inds parameter"

    def test_return_type_consistency(self):
        """Test that all inference functions return numpy arrays."""
        # Create mock adata
        n_cells, n_genes = 20, 50
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 10 + ["non-expanded"] * 10})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        mock_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        mock_model = MagicMock()
        mock_data_format = DataFormat(
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
            genes_mu=np.zeros(50),
            genes_sigma=np.ones(50),
        )

        # Test each model type returns numpy array
        model_types = ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]

        for model_type in model_types:
            patch_name = {
                "mlp": "run_mlp_inference",
                "autoencoder": "run_ae_inference",
                "logistic": "run_linear_inference",
                "svm": "run_linear_inference",
                "lightgbm": "run_lightgbm_inference",
            }[model_type]

            with patch(f"scxpand.util.inference_utils.{patch_name}") as mock_inference:
                # Return a numpy array
                expected_result = np.random.rand(10)
                mock_inference.return_value = expected_result

                result = run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=mock_data_format,
                    adata=mock_adata,
                )

                assert isinstance(result, np.ndarray)
                np.testing.assert_array_equal(result, expected_result)


class TestEdgeCaseParameterHandling:
    """Test edge cases in parameter handling."""

    def test_none_parameters_handling(self):
        """Test handling of None parameters."""
        # Create mock adata
        n_cells, n_genes = 20, 50
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 10 + ["non-expanded"] * 10})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        mock_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        mock_model = MagicMock()
        mock_data_format = DataFormat(
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
            genes_mu=np.zeros(50),
            genes_sigma=np.ones(50),
        )

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(20)

            # Test with None device
            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                device=None,
            )

            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["device"] is None

    def test_empty_eval_row_inds_handling(self):
        """Test handling of empty eval_row_inds."""
        # Create mock adata
        n_cells, n_genes = 20, 50
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 10 + ["non-expanded"] * 10})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        mock_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        mock_model = MagicMock()
        mock_data_format = DataFormat(
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
            genes_mu=np.zeros(50),
            genes_sigma=np.ones(50),
        )

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(0)  # Empty result

            # Test with empty array
            empty_indices = np.array([], dtype=int)
            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                eval_row_inds=empty_indices,
            )

            call_kwargs = mock_inference.call_args.kwargs
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], empty_indices)

    def test_large_batch_size_handling(self):
        """Test handling of very large batch sizes."""
        # Create mock adata
        n_cells, n_genes = 20, 50
        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"] * 10 + ["non-expanded"] * 10})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        mock_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        mock_model = MagicMock()
        mock_data_format = DataFormat(
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
            genes_mu=np.zeros(50),
            genes_sigma=np.ones(50),
        )

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(20)

            # Test with very large batch size
            large_batch_size = 100000
            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                batch_size=large_batch_size,
            )

            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["batch_size"] == large_batch_size
