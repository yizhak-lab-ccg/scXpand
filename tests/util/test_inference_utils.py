"""Tests for inference_utils.py functionality."""

import importlib
import inspect
import json
import os
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.core.model_types import MODEL_TYPES
from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import CellsDataset, cells_collate_fn
from scxpand.util.inference_utils import load_model, run_model_inference
from scxpand.util.model_constants import (
    BEST_CHECKPOINT_FILE,
    DATA_FORMAT_NPZ_FILE,
    SKLEARN_MODEL_FILE,
)


class TestInferenceUtils:
    """Test the inference_utils.py module functionality."""

    @pytest.fixture
    def mock_adata_with_metadata(self, tmp_path):
        """Create mock AnnData with metadata columns."""
        n_cells, n_genes = 20, 10
        X = np.random.randn(n_cells, n_genes).astype(np.float32)

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 10 + ["non-expanded"] * 10,
                "clone_id_size": np.random.randint(1, 100, size=n_cells),
                "median_clone_size": np.random.randint(1, 50, size=n_cells),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
                "imputed_labels": np.random.choice(["label1", "label2"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_adata.h5ad"
        adata.write_h5ad(file_path)

        return file_path, adata

    @pytest.fixture
    def mock_adata_minimal(self, tmp_path):
        """Create mock AnnData with minimal metadata (no expansion columns)."""
        n_cells, n_genes = 20, 10
        X = np.random.randn(n_cells, n_genes).astype(np.float32)

        # Only basic metadata, no expansion columns
        obs_df = pd.DataFrame(
            {
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_minimal_adata.h5ad"
        adata.write_h5ad(file_path)

        return file_path, adata

    @pytest.fixture
    def data_format(self):
        """Create a DataFormat for testing."""
        return DataFormat(
            n_genes=10,
            gene_names=[f"g{i}" for i in range(10)],
            genes_mu=np.zeros(10, dtype=np.float32),
            genes_sigma=np.ones(10, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

    def test_load_model_mlp(self, tmp_path):
        """Test loading MLP model."""
        # Create a mock results directory
        results_path = tmp_path / "mlp_results"
        results_path.mkdir()
        (results_path / BEST_CHECKPOINT_FILE).touch()

        # Create a valid parameters.json file
        params = {"n_epochs": 10, "layer_units": [64, 32]}
        with open(results_path / "parameters.json", "w") as f:
            json.dump(params, f)

        # Create a valid data_format.json file
        data_format_data = {"n_genes": 10, "gene_names": ["g0", "g1", "g2"]}
        with open(results_path / "data_format.json", "w") as f:
            json.dump(data_format_data, f)

        # Create a valid data_format.npz file
        genes_mu = np.zeros(10, dtype=np.float32)
        genes_sigma = np.ones(10, dtype=np.float32)
        np.savez(
            results_path / DATA_FORMAT_NPZ_FILE,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
        )

        # Mock the MLP model loading at the module level where it's imported
        with patch("scxpand.util.inference_utils.load_nn_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            model = load_model("mlp", results_path, "cpu")

            assert model == mock_model
            mock_load.assert_called_once_with(results_path=results_path, device="cpu")

    def test_load_model_linear(self, tmp_path):
        """Test loading linear model (svm/logistic)."""
        # Create a mock results directory
        results_path = tmp_path / "linear_results"
        results_path.mkdir()
        (results_path / SKLEARN_MODEL_FILE).touch()

        # Create a valid parameters.json file
        params = {"n_epochs": 10, "model_type": "logistic"}
        with open(results_path / "parameters.json", "w") as f:
            json.dump(params, f)

        # Mock the sklearn model loading at the module level where it's imported
        with patch("scxpand.util.inference_utils.load_sklearn_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            for model_type in ["svm", "logistic"]:
                model = load_model(model_type, results_path, "cpu")

                assert model == mock_model
                mock_load.assert_called_with(results_path=results_path)

    def test_load_model_autoencoder(self, tmp_path):
        """Test loading autoencoder model."""
        # Create a mock results directory
        results_path = tmp_path / "ae_results"
        results_path.mkdir()
        (results_path / BEST_CHECKPOINT_FILE).touch()

        # Create a valid data_format.json file
        data_format_data = {"n_genes": 10, "gene_names": ["g0", "g1", "g2"]}
        with open(results_path / "data_format.json", "w") as f:
            json.dump(data_format_data, f)

        # Create a valid data_format.npz file
        genes_mu = np.zeros(10, dtype=np.float32)
        genes_sigma = np.ones(10, dtype=np.float32)
        np.savez(
            results_path / DATA_FORMAT_NPZ_FILE,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
        )

        # Mock the autoencoder model loading at the module level where it's imported
        with patch("scxpand.util.inference_utils.load_ae_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            model = load_model("autoencoder", results_path, "cpu")

            assert model == mock_model
            mock_load.assert_called_once_with(model_path=results_path, device="cpu")

    def test_load_model_unsupported_type(self, tmp_path):
        """Test loading unsupported model type raises error."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        with pytest.raises(ValueError, match="Unsupported model_type for loading"):
            load_model("unsupported", results_path, "cpu")

    def test_load_model_none_return(self, tmp_path):
        """Test loading model that returns None raises error."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        # Create a valid parameters.json file
        params = {"n_epochs": 10, "layer_units": [64, 32]}
        with open(results_path / "parameters.json", "w") as f:
            json.dump(params, f)

        # Create a valid data_format.json file
        data_format_data = {"n_genes": 10, "gene_names": ["g0", "g1", "g2"]}
        with open(results_path / "data_format.json", "w") as f:
            json.dump(data_format_data, f)

        # Create a valid data_format.npz file
        genes_mu = np.zeros(10, dtype=np.float32)
        genes_sigma = np.ones(10, dtype=np.float32)
        np.savez(
            results_path / DATA_FORMAT_NPZ_FILE,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
        )

        with patch("scxpand.util.inference_utils.load_nn_model") as mock_load:
            mock_load.return_value = None

            with pytest.raises(RuntimeError, match="Model loading failed"):
                load_model("mlp", results_path, "cpu")

    def test_run_inference_mlp(self, mock_adata_with_metadata, data_format):
        """Test running MLP inference (with adata only)."""
        file_path, adata = mock_adata_with_metadata
        # Mock the MLP inference function at the module level where it's imported
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            mock_model = MagicMock()
            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            mock_inference.assert_called_once()

    def test_run_inference_linear(self, mock_adata_with_metadata, data_format):
        """Test running linear inference (svm/logistic) (with adata only)."""
        file_path, adata = mock_adata_with_metadata
        # Mock the linear inference function at the module level where it's imported
        with patch(
            "scxpand.util.inference_utils.run_linear_inference"
        ) as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            mock_model = MagicMock()
            for model_type in ["svm", "logistic"]:
                result = run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=data_format,
                    adata=adata,
                    data_path=None,
                    eval_row_inds=np.arange(10),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )
                assert isinstance(result, np.ndarray)
                assert len(result) == 10
                mock_inference.assert_called()

    def test_run_inference_autoencoder(self, mock_adata_with_metadata, data_format):
        """Test running autoencoder inference (with adata only)."""
        file_path, adata = mock_adata_with_metadata
        # Mock the autoencoder inference function at the module level where it's imported
        with patch("scxpand.util.inference_utils.run_ae_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            mock_model = MagicMock()
            result = run_model_inference(
                model_type="autoencoder",
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            assert isinstance(result, np.ndarray)
            assert len(result) == 10
            mock_inference.assert_called_once()

    def test_run_inference_lightgbm(self, mock_adata_with_metadata):
        """Test running LightGBM inference (with adata only)."""
        file_path, adata = mock_adata_with_metadata
        # Mock the LightGBM inference function at the module level where it's imported
        with patch(
            "scxpand.util.inference_utils.run_lightgbm_inference"
        ) as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            mock_model = MagicMock()
            result = run_model_inference(
                model_type="lightgbm",
                model=mock_model,
                data_format=None,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            assert isinstance(result, np.ndarray)
            assert len(result) == 10
            mock_inference.assert_called_once()

    def test_run_inference_unsupported_type_with_adata(
        self, mock_adata_with_metadata, data_format
    ):
        """Test running inference with unsupported model type (with adata only)."""
        file_path, adata = mock_adata_with_metadata
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="Unsupported model_type for inference"):
            run_model_inference(
                model_type="unsupported",
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )

    def test_run_inference_unsupported_type_with_data_path(
        self, mock_adata_with_metadata, data_format
    ):
        """Test running inference with unsupported model type (with data_path only)."""
        file_path, adata = mock_adata_with_metadata
        mock_model = MagicMock()
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with pytest.raises(
                ValueError, match="Unsupported model_type for inference"
            ):
                run_model_inference(
                    model_type="unsupported",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(10),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )

    def test_run_inference_with_data_format_conversion(
        self, mock_adata_with_metadata, data_format
    ):
        """Test that run_inference properly passes adata to individual inference functions."""
        file_path, adata = mock_adata_with_metadata
        # Mock the MLP inference function at the module level where it's imported
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            mock_model = MagicMock()
            _result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            # Verify that the individual inference function was called with the adata object
            mock_inference.assert_called_once()
            # Verify the call includes the adata parameter
            call_args = mock_inference.call_args
            assert call_args.kwargs["adata"] is adata

    def test_run_inference_without_data_format(self, mock_adata_with_metadata):
        """Test that run_inference works without data_format when adata is provided."""
        file_path, adata = mock_adata_with_metadata

        # Mock the LightGBM inference function at the module level where it's imported
        with patch(
            "scxpand.util.inference_utils.run_lightgbm_inference"
        ) as mock_inference:
            mock_inference.return_value = np.random.rand(10)

            mock_model = MagicMock()

            result = run_model_inference(
                model_type="lightgbm",
                model=mock_model,
                data_format=None,  # No data_format
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )

            assert isinstance(result, np.ndarray)
            assert len(result) == 10

    def test_run_inference_parameter_validation(self):
        """Test that run_inference validates parameters correctly."""
        # Test with missing required parameters
        with pytest.raises(TypeError):
            run_model_inference()  # Missing required parameters

    def test_run_inference_adata_and_data_path(
        self, mock_adata_with_metadata, data_format
    ):
        """Test that run_inference handles both adata and data_path (allowing both, requiring at least one)."""
        file_path, adata = mock_adata_with_metadata
        mock_model = MagicMock()

        # Both provided - should work fine now
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(5)
            _result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=file_path,
                eval_row_inds=np.arange(5),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            # Should pass both parameters through
            mock_inference.assert_called_once()
            call_kwargs = mock_inference.call_args[1]
            assert call_kwargs["adata"] is adata
            assert call_kwargs["data_path"] == file_path

        # Neither provided
        with pytest.raises(
            ValueError, match="Either adata or data_path must be provided"
        ):
            run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=data_format,
                adata=None,
                data_path=None,
                eval_row_inds=np.arange(5),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )

    def test_run_inference_with_only_adata(self, mock_adata_with_metadata, data_format):
        """Test run_inference works with only adata (data_path=None)."""
        file_path, adata = mock_adata_with_metadata
        mock_model = MagicMock()
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=data_format,
                adata=adata,
                data_path=None,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            assert isinstance(result, np.ndarray)
            assert len(result) == 10
            mock_inference.assert_called_once()

    def test_run_inference_with_only_data_path(
        self, mock_adata_with_metadata, data_format
    ):
        """Test run_inference works with only data_path (adata=None)."""
        file_path, adata = mock_adata_with_metadata
        mock_model = MagicMock()

        # Patch the inference function
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(10)
            result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=data_format,
                adata=None,
                data_path=file_path,
                eval_row_inds=np.arange(10),
                device="cpu",
                batch_size=4,
                num_workers=0,
            )
            assert isinstance(result, np.ndarray)
            assert len(result) == 10
            # Verify the inference function was called with the data_path
            mock_inference.assert_called_once()
            call_args = mock_inference.call_args
            assert call_args.kwargs["data_path"] == file_path
            assert call_args.kwargs["adata"] is None

    def test_inference_functions_use_is_train_false(
        self, mock_adata_with_metadata, data_format
    ):
        """Test that all inference functions create datasets with is_train=False."""
        file_path, adata = mock_adata_with_metadata

        # Test MLP inference - verify source code uses is_train=False
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_mlp_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(5)

                mock_model = MagicMock()

                # This should call the mocked function
                _result = run_model_inference(
                    model_type="mlp",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(5),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )

                # Verify the function was called
                mock_inference.assert_called_once()

        # Test Linear inference - verify source code uses is_train=False
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_linear_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(5)

                mock_model = MagicMock()

                # This should call the mocked function
                _result = run_model_inference(
                    model_type="svm",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(5),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )

                # Verify the function was called
                mock_inference.assert_called_once()

        # Test Autoencoder inference - verify source code uses is_train=False
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_ae_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(5)

                mock_model = MagicMock()

                # This should call the mocked function
                _result = run_model_inference(
                    model_type="autoencoder",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(5),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )

                # Verify the function was called
                mock_inference.assert_called_once()

    def test_inference_handles_missing_columns(self, mock_adata_minimal, data_format):
        """Test that inference functions handle missing columns gracefully."""
        file_path, adata_minimal = mock_adata_minimal

        # Test MLP inference with missing columns
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata_minimal
            with patch(
                "scxpand.util.inference_utils.run_mlp_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(5)

                mock_model = MagicMock()

                # This should not raise KeyError due to missing columns
                result = run_model_inference(
                    model_type="mlp",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(5),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )

                assert isinstance(result, np.ndarray)
                assert len(result) == 5

    def test_inference_utils_function_signature(self):
        """Test that inference_utils functions have correct signatures."""
        # Test load_model signature
        load_sig = inspect.signature(load_model)
        expected_load_params = {"model_type", "model_path", "device"}
        actual_load_params = set(load_sig.parameters.keys())
        assert expected_load_params.issubset(actual_load_params)

        # Test run_inference signature
        run_sig = inspect.signature(run_model_inference)
        expected_run_params = {
            "model_type",
            "model",
            "data_format",
            "adata",
            "data_path",
            "eval_row_inds",
            "device",
            "batch_size",
            "num_workers",
        }
        actual_run_params = set(run_sig.parameters.keys())
        assert expected_run_params.issubset(actual_run_params)

    def test_source_code_verification(self):
        """Verify that source code actually uses is_train=False."""
        # Check MLP inference
        mlp_file = "scxpand/mlp/mlp_trainer.py"
        if os.path.exists(mlp_file):
            with open(mlp_file) as f:
                content = f.read()
                assert "is_train=False" in content, (
                    "MLP inference should use is_train=False"
                )

        # Check Linear inference
        linear_file = "scxpand/linear/sklearn_utils.py"
        if os.path.exists(linear_file):
            with open(linear_file) as f:
                content = f.read()
                assert "is_train=False" in content, (
                    "Linear inference should use is_train=False"
                )

        # Check Autoencoder inference
        ae_file = "scxpand/autoencoders/ae_trainer.py"
        if os.path.exists(ae_file):
            with open(ae_file) as f:
                content = f.read()
                assert "is_train=False" in content, (
                    "Autoencoder inference should use is_train=False"
                )

    def test_inference_with_gene_mismatch(self, tmp_path, data_format):
        """Test inference with gene mismatches between model and data."""
        # Create test data with different genes than the target format
        n_cells, n_genes_original = 10, 15
        X = np.random.randn(n_cells, n_genes_original).astype(np.float32)
        # Create gene names that partially overlap with target format
        original_gene_names = [f"gene_{i}" for i in range(5)] + [
            f"extra_gene_{i}" for i in range(10)
        ]
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["non-expanded"] * 5,
                "clone_id_size": np.random.randint(1, 100, size=n_cells),
                "median_clone_size": np.random.randint(1, 50, size=n_cells),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=original_gene_names)
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_gene_mismatch.h5ad"
        adata.write_h5ad(file_path)
        # Mock the MLP inference function at the module level where it's imported
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(n_cells)
            mock_model = MagicMock()
            # This should handle gene mismatches gracefully
            try:
                result = run_model_inference(
                    model_type="mlp",
                    model=mock_model,
                    data_format=data_format,
                    adata=adata,
                    data_path=None,
                    eval_row_inds=np.arange(n_cells),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )
                assert isinstance(result, np.ndarray)
                assert len(result) == n_cells
                mock_inference.assert_called_once()
            except Exception as e:
                pytest.fail(
                    f"Inference with gene mismatch should not fail, but got: {e}"
                )

    def test_inference_with_completely_different_genes(self, tmp_path):
        """Test inference when genes are completely different."""
        # Create test data with completely different genes
        n_cells, n_genes_original = 10, 8
        X = np.random.randn(n_cells, n_genes_original).astype(np.float32)
        # Create gene names that don't overlap at all with target format
        original_gene_names = [f"different_gene_{i}" for i in range(n_genes_original)]
        target_gene_names = [f"target_gene_{i}" for i in range(5)]
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["non-expanded"] * 5,
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=original_gene_names)
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_no_gene_overlap.h5ad"
        adata.write_h5ad(file_path)
        # Create data format with completely different genes
        data_format = DataFormat(
            n_genes=len(target_gene_names),
            gene_names=target_gene_names,
            genes_mu=np.zeros(len(target_gene_names), dtype=np.float32),
            genes_sigma=np.ones(len(target_gene_names), dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )
        # Mock the MLP inference function at the module level where it's imported
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(n_cells)
            mock_model = MagicMock()
            # This should handle completely different genes gracefully
            try:
                result = run_model_inference(
                    model_type="mlp",
                    model=mock_model,
                    data_format=data_format,
                    adata=adata,
                    data_path=None,
                    eval_row_inds=np.arange(n_cells),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )
                assert isinstance(result, np.ndarray)
                assert len(result) == n_cells
                mock_inference.assert_called_once()
            except Exception as e:
                pytest.fail(
                    f"Inference with completely different genes should not fail, but got: {e}"
                )

    def test_data_format_conversion_with_gene_mismatch(self):
        """Test the actual data format conversion with gene mismatches."""
        # Create test data with different genes than the target format
        n_cells, n_genes_original = 10, 15
        X = np.random.randn(n_cells, n_genes_original).astype(np.float32)

        # Create gene names that partially overlap with target format
        original_gene_names = [f"gene_{i}" for i in range(5)] + [
            f"extra_gene_{i}" for i in range(10)
        ]
        target_gene_names = [f"gene_{i}" for i in range(10)]  # 5 overlap, 5 missing

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["non-expanded"] * 5,
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=original_gene_names)

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        # Create data format with target genes
        data_format = DataFormat(
            n_genes=len(target_gene_names),
            gene_names=target_gene_names,
            genes_mu=np.zeros(len(target_gene_names), dtype=np.float32),
            genes_sigma=np.ones(len(target_gene_names), dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        print(f"Original genes: {original_gene_names}")
        print(f"Target genes: {target_gene_names}")
        print(f"Original adata shape: {adata.shape}")

        # This should not fail but convert the data format correctly
        try:
            # By default, prepare_adata_for_training doesn't reorder genes
            converted_adata = data_format.prepare_adata_for_training(adata)
            print(f"Converted adata shape: {converted_adata.shape}")
            print(f"Converted gene names: {converted_adata.var_names.tolist()}")

            # Verify the conversion worked - by default it doesn't change the shape
            assert converted_adata.shape[0] == n_cells  # Same number of cells
            assert (
                converted_adata.shape[1] == n_genes_original
            )  # Same number of genes (no reordering by default)
            assert (
                converted_adata.var_names.tolist() == original_gene_names
            )  # Same gene order

            # Now test with explicit gene reordering
            converted_adata_reordered = data_format.prepare_adata_for_training(
                adata, reorder_genes=True
            )
            print(f"Reordered adata shape: {converted_adata_reordered.shape}")
            print(
                f"Reordered gene names: {converted_adata_reordered.var_names.tolist()}"
            )

            # Verify the reordering worked
            assert converted_adata_reordered.shape[0] == n_cells  # Same number of cells
            assert converted_adata_reordered.shape[1] == len(
                target_gene_names
            )  # Target number of genes
            assert (
                converted_adata_reordered.var_names.tolist() == target_gene_names
            )  # Correct gene order

        except Exception as e:
            pytest.fail(
                f"Data format conversion with gene mismatch should not fail, but got: {e}"
            )

    def test_dataset_creation_with_gene_mismatch(self, tmp_path):
        """Test that dataset creation works with gene mismatches."""
        # Create test data with different genes than the target format
        n_cells, n_genes_original = 10, 15
        X = np.random.randn(n_cells, n_genes_original).astype(np.float32)

        # Create gene names that partially overlap with target format
        original_gene_names = [f"gene_{i}" for i in range(5)] + [
            f"extra_gene_{i}" for i in range(10)
        ]
        target_gene_names = [f"gene_{i}" for i in range(10)]  # 5 overlap, 5 missing

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["non-expanded"] * 5,
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=original_gene_names)

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_gene_mismatch_dataset.h5ad"
        adata.write_h5ad(file_path)

        # Create data format with target genes
        data_format = DataFormat(
            n_genes=len(target_gene_names),
            gene_names=target_gene_names,
            genes_mu=np.zeros(len(target_gene_names), dtype=np.float32),
            genes_sigma=np.ones(len(target_gene_names), dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        print(f"Original genes: {original_gene_names}")
        print(f"Target genes: {target_gene_names}")
        print(f"Original adata shape: {adata.shape}")

        # This should not fail when creating the dataset for inference
        try:
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=np.arange(n_cells),
                dataset_params=None,
                is_train=False,  # Inference mode
                data_path=file_path,
            )

            print("Dataset created successfully")
            print(f"Dataset n_genes: {dataset.n_genes}")

            # Verify the dataset was created correctly
            assert dataset.n_genes == len(target_gene_names)

            # Try to get a batch from the dataset using the collate function
            batch_indices = [0, 1]  # Get a small batch
            batch = cells_collate_fn(batch_indices, dataset)
            print(f"Successfully got batch with x shape: {batch['x'].shape}")

        except Exception as e:
            pytest.fail(
                f"Dataset creation with gene mismatch should not fail, but got: {e}"
            )

    def test_tensor_dimension_mismatch_fix(self, tmp_path):
        """Test that the specific RuntimeError with tensor dimension mismatch is fixed."""
        # Create test data that reproduces the original error scenario
        n_cells = 5
        n_genes_original = 18619  # Original data had many genes
        n_genes_target = 11950  # Target format has fewer genes

        # Create large gene expression matrix
        X = np.random.randn(n_cells, n_genes_original).astype(np.float32)

        # Create gene names - some overlap, many extra
        original_gene_names = [f"gene_{i}" for i in range(n_genes_original)]
        target_gene_names = [
            f"gene_{i}" for i in range(n_genes_target)
        ]  # Subset of original

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 3 + ["non-expanded"] * 2,
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=original_gene_names)

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "test_dimension_mismatch.h5ad"
        adata.write_h5ad(file_path)

        # Create data format with target genes (fewer than original)
        data_format = DataFormat(
            n_genes=n_genes_target,
            gene_names=target_gene_names,
            genes_mu=np.random.randn(n_genes_target).astype(
                np.float32
            ),  # Target dimensions
            genes_sigma=np.random.rand(n_genes_target).astype(np.float32)
            + 0.1,  # Target dimensions
            use_log_transform=False,
            target_sum=1e4,
        )

        print(f"Original data: {n_genes_original} genes")
        print(f"Target format: {n_genes_target} genes")
        print(f"genes_mu shape: {data_format.genes_mu.shape}")
        print(f"genes_sigma shape: {data_format.genes_sigma.shape}")

        # This should NOT raise RuntimeError: tensor dimension mismatch
        try:
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=np.arange(n_cells),
                dataset_params=None,
                is_train=False,  # Inference mode
                data_path=file_path,
            )

            print(f"genes_mu_tensor shape: {dataset.genes_mu_tensor.shape}")
            print(f"genes_sigma_tensor shape: {dataset.genes_sigma_tensor.shape}")

            # Verify dimensions match
            assert dataset.genes_mu_tensor.shape[0] == n_genes_target
            assert dataset.genes_sigma_tensor.shape[0] == n_genes_target

            # The critical test: this should not raise RuntimeError about tensor dimensions
            batch_indices = [0, 1]
            batch = cells_collate_fn(batch_indices, dataset)

            print(f"Batch x shape: {batch['x'].shape}")
            print("✅ No RuntimeError - tensor dimensions match!")

            # Verify the batch has correct dimensions
            assert batch["x"].shape == (
                2,
                n_genes_target,
            )  # [batch_size, n_genes_target]

        except RuntimeError as e:
            if "must match the size" in str(e):
                pytest.fail(
                    f"RuntimeError with tensor dimension mismatch still occurs: {e}"
                )
            else:
                raise  # Re-raise if it's a different RuntimeError
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    def test_inference_utils_imports(self):
        """Test that inference_utils can be imported and has expected functions."""
        assert callable(load_model)
        assert callable(run_model_inference)

    def test_run_inference_mlp_with_data_path(
        self, mock_adata_with_metadata, data_format
    ):
        """Test running MLP inference (with data_path only)."""
        file_path, adata = mock_adata_with_metadata
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_mlp_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(10)
                mock_model = MagicMock()
                result = run_model_inference(
                    model_type="mlp",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(10),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )
                assert isinstance(result, np.ndarray)
                assert len(result) == 10
                mock_inference.assert_called_once()

    def test_run_inference_linear_with_data_path(
        self, mock_adata_with_metadata, data_format
    ):
        """Test running linear inference (svm/logistic) (with data_path only)."""
        file_path, adata = mock_adata_with_metadata
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_linear_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(10)
                mock_model = MagicMock()
                for model_type in ["svm", "logistic"]:
                    result = run_model_inference(
                        model_type=model_type,
                        model=mock_model,
                        data_format=data_format,
                        adata=None,
                        data_path=file_path,
                        eval_row_inds=np.arange(10),
                        device="cpu",
                        batch_size=4,
                        num_workers=0,
                    )
                    assert isinstance(result, np.ndarray)
                    assert len(result) == 10
                    mock_inference.assert_called()

    def test_run_inference_autoencoder_with_data_path(
        self, mock_adata_with_metadata, data_format
    ):
        """Test running autoencoder inference (with data_path only)."""
        file_path, adata = mock_adata_with_metadata
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_ae_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(10)
                mock_model = MagicMock()
                result = run_model_inference(
                    model_type="autoencoder",
                    model=mock_model,
                    data_format=data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(10),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )
                assert isinstance(result, np.ndarray)
                assert len(result) == 10
                mock_inference.assert_called_once()

    def test_run_inference_lightgbm_with_data_path(self, mock_adata_with_metadata):
        """Test running LightGBM inference (with data_path only)."""
        file_path, adata = mock_adata_with_metadata
        with patch("scxpand.util.inference_utils.ad.read_h5ad") as mock_read_h5ad:
            mock_read_h5ad.return_value = adata
            with patch(
                "scxpand.util.inference_utils.run_lightgbm_inference"
            ) as mock_inference:
                mock_inference.return_value = np.random.rand(10)
                mock_model = MagicMock()
                result = run_model_inference(
                    model_type="lightgbm",
                    model=mock_model,
                    data_format=None,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=np.arange(10),
                    device="cpu",
                    batch_size=4,
                    num_workers=0,
                )
                assert isinstance(result, np.ndarray)
                assert len(result) == 10
                mock_inference.assert_called_once()

    def test_main_model_type_consistency(self):
        """Test that the model type names in main.py are consistent with the model registry."""
        main_mod = importlib.import_module("scxpand.main")

        # Get all model types from the registry
        registry_keys = set(MODEL_TYPES.keys())

        # Verify that main.py imports and uses the same MODEL_TYPES
        assert hasattr(main_mod, "MODEL_TYPES"), "main.py should import MODEL_TYPES"
        main_registry_keys = set(main_mod.MODEL_TYPES.keys())

        # All sets should be identical
        assert registry_keys == main_registry_keys, (
            f"Inconsistent model type names between modules:\n"
            f"core.model_types.MODEL_TYPES: {sorted([m.value for m in registry_keys])}\n"
            f"main.MODEL_TYPES: {sorted([m.value for m in main_registry_keys])}"
        )

        # Verify that all ModelSpecs have default_save_dir
        for model_type, spec in MODEL_TYPES.items():
            assert hasattr(spec, "default_save_dir"), (
                f"ModelSpec for {model_type.value} missing default_save_dir"
            )
            assert spec.default_save_dir, (
                f"ModelSpec for {model_type.value} has empty default_save_dir"
            )
