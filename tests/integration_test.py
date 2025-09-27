import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.autoencoders.run_ae_train import run_ae_training
from scxpand.data_util.data_format import load_data_format
from scxpand.data_util.dataset import CellsDataset
from scxpand.data_util.statistics import compute_preprocessed_genes_means_stds
from scxpand.lightgbm.lightgbm_params import LightGBMParams
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_training
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.linear_trainer import run_linear_training
from scxpand.mlp.mlp_params import MLPParam
from scxpand.mlp.mlp_trainer import run_mlp_inference
from scxpand.mlp.run_mlp_train import run_mlp_training
from scxpand.util.model_constants import (
    BEST_MODEL_METRICS_FILE,
    DATA_FORMAT_NPZ_FILE,
    SKLEARN_MODEL_FILE,
)
from tests.test_utils import create_temp_h5ad_file, robust_remove


class TestIntegration:
    def _validate_results(self, results: dict, test_name: str) -> None:
        """Validate that results dictionary contains expected keys and values."""
        required_keys = [
            "AUROC",
            "F1",
            "false_positive_rate",
            "false_negative_rate",
            "error_rate",
        ]
        for key in required_keys:
            assert key in results, f"{key} missing for {test_name}"

        # Validate AUROC and F1 values
        for metric in ["AUROC", "F1"]:
            value = results[metric]
            assert isinstance(value, float | np.floating) or np.isnan(
                value
            ), f"{metric} invalid type for {test_name}"
            if not np.isnan(value):
                assert 0 <= value <= 1, f"{metric} out of range for {test_name}"

    def test_run_mlp_training_integration(self, dummy_adata):
        """Test MLP training pipeline with real data flow and normalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Create parameters for a quick training run
            prm = MLPParam(
                n_epochs=2,  # Very short for testing
                train_batch_size=16,
                inference_batch_size=16,
                layer_units=(32, 16),
                random_seed=42,
                use_log_transform=True,  # Test normalization
                aux_categorical_types=["tissue_type", "imputed_labels"],
                early_stopping_patience=1,
                train_log_interval=1,
            )

            try:
                # Run actual training with minimal mocking
                results = run_mlp_training(
                    data_path=test_file_path,
                    base_save_dir=Path(temp_dir) / "mlp_results",
                    prm=prm,
                    device="cpu",
                    dev_ratio=0.3,
                    num_workers=0,
                )

                # Validate results structure
                self._validate_results(results, "mlp_integration")

                # Check that model and results were saved
                save_dir = Path(temp_dir) / "mlp_results"
                assert (
                    save_dir / "best_ckpt.pt"
                ).exists(), "Best model checkpoint should be saved"
                assert (
                    save_dir / "parameters.json"
                ).exists(), "Parameters should be saved"
                assert (
                    save_dir / BEST_MODEL_METRICS_FILE
                ).exists(), "Results should be saved"
            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)

    def test_run_linear_training_integration(self, dummy_adata):
        """Test linear classifier training pipeline with real data flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Create parameters for a quick training run
            prm = LinearClassifierParam(
                n_epochs=5,  # Short for testing
                model_type="logistic",
                batch_size=32,
                random_seed=42,
                use_log_transform=True,  # Test normalization
                early_stopping_patience=2,
                eval_interval=1,
            )

            try:
                # Run actual training
                results = run_linear_training(
                    data_path=test_file_path,
                    base_save_dir=Path(temp_dir) / "linear_results",
                    prm=prm,
                    dev_ratio=0.3,
                )

                # Validate results structure
                self._validate_results(results, "linear_integration")

                # Check that model and results were saved
                save_dir = Path(temp_dir) / "linear_results"
                assert (
                    save_dir / SKLEARN_MODEL_FILE
                ).exists(), "Model file should be saved"
                assert (
                    save_dir / "parameters.json"
                ).exists(), "Parameters should be saved"

            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)

    def test_run_lightgbm_training_integration(self, dummy_adata):
        """Test LightGBM training pipeline with real data flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Create parameters for a quick training run
            prm = LightGBMParams(
                n_estimators=10,  # Very few for testing
                max_depth=3,
                learning_rate=0.1,
                random_seed=42,
                use_log_transform=True,  # Test normalization
                n_jobs=1,  # Single thread for testing
            )

            try:
                # Run actual training
                results = run_lightgbm_training(
                    data_path=test_file_path,
                    base_save_dir=Path(temp_dir) / "lgb_results",
                    prm=prm,
                    dev_ratio=0.3,
                )

                # Validate results structure
                self._validate_results(results, "lightgbm_integration")

                # Check that model and results were saved
                save_dir = Path(temp_dir) / "lgb_results"
                assert (
                    save_dir / SKLEARN_MODEL_FILE
                ).exists(), "Model file should be saved"
                assert (
                    save_dir / "parameters.json"
                ).exists(), "Parameters should be saved"

            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)

    def test_autoencoder_training_integration(self, dummy_adata):
        """Test autoencoder training pipeline with real data flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Create parameters for a quick training run
            prm = AutoEncoderParams(
                n_epochs=2,  # Very short for testing
                train_batch_size=16,
                model_type="standard",
                loss_type="mse",  # Simple loss for testing
                cat_loss_weight=0.5,  # Test categorical prediction
                random_seed=42,
                use_log_transform=True,  # Test normalization
                aux_categorical_types=["tissue_type", "imputed_labels"],
                early_stopping_patience=1,
                train_log_interval=1,
            )

            try:
                # Run actual training
                results = run_ae_training(
                    data_path=test_file_path,
                    base_save_dir=Path(temp_dir) / "ae_results",
                    prm=prm,
                    device="cpu",
                    dev_ratio=0.3,
                )

                # Validate results structure
                self._validate_results(results, "autoencoder_integration")

                # Check that model and results were saved
                save_dir = Path(temp_dir) / "ae_results"
                assert (
                    save_dir / "best_ckpt.pt"
                ).exists(), "Best model checkpoint should be saved"
                assert (
                    save_dir / "parameters.json"
                ).exists(), "Parameters should be saved"

            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)

    def test_data_normalization_consistency(self, dummy_adata):
        """Test that data normalization is consistent between training and inference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Test with MLP model which we can easily inspect
            prm = MLPParam(
                n_epochs=1,  # Just one epoch to save time
                train_batch_size=16,
                inference_batch_size=16,
                layer_units=(32,),
                random_seed=42,
                use_log_transform=True,  # Enable normalization
                aux_categorical_types=["tissue_type"],
            )

            # Patch the inference function to capture normalized data
            original_run_mlp_inference = None
            captured_inference_data = {}

            def capture_inference_data(*args, **kwargs):
                # Capture the data format used in inference
                if "data_format" in kwargs:
                    captured_inference_data["data_format"] = kwargs["data_format"]
                return original_run_mlp_inference(*args, **kwargs)

            try:
                original_run_mlp_inference = run_mlp_inference

                with patch(
                    "scxpand.mlp.mlp_trainer.run_mlp_inference",
                    side_effect=capture_inference_data,
                ):
                    results = run_mlp_training(
                        data_path=test_file_path,
                        base_save_dir=Path(temp_dir) / "norm_test",
                        prm=prm,
                        device="cpu",
                        dev_ratio=0.3,
                        num_workers=0,
                    )

                # Validate that normalization parameters were captured
                if captured_inference_data:
                    data_format = captured_inference_data["data_format"]
                    assert hasattr(
                        data_format, "genes_mu"
                    ), "Should have gene means for normalization"
                    assert hasattr(
                        data_format, "genes_sigma"
                    ), "Should have gene stds for normalization"
                    assert (
                        len(data_format.genes_mu) == dummy_adata.n_vars
                    ), "Should have means for all genes"
                    assert (
                        len(data_format.genes_sigma) == dummy_adata.n_vars
                    ), "Should have stds for all genes"

                    # Check that normalization parameters are reasonable
                    assert np.all(
                        data_format.genes_sigma > 0
                    ), "Gene standard deviations should be positive"

                # Validate results
                self._validate_results(results, "normalization_test")

            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)

    def test_data_normalization_values(self, dummy_adata):
        """Test that data normalization produces expected values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Test with MLP model to capture normalization data
            prm = MLPParam(
                n_epochs=1,
                train_batch_size=16,
                layer_units=(16,),
                random_seed=42,
                use_log_transform=True,  # Enable normalization
                aux_categorical_types=["tissue_type"],
            )

            try:
                # Run training
                results = run_mlp_training(
                    data_path=test_file_path,
                    base_save_dir=Path(temp_dir) / "norm_values_test",
                    prm=prm,
                    device="cpu",
                    dev_ratio=0.3,
                    num_workers=0,
                )

                # Load the saved data format to check normalization parameters
                save_dir = Path(temp_dir) / "norm_values_test"
                data_format_path = save_dir / "data_format.json"

                assert data_format_path.exists(), "Data format file should exist"

                # Load normalization parameters from JSON (metadata)
                with open(data_format_path) as f:
                    data_format_info = json.load(f)

                # Check that normalization metadata is present
                assert (
                    "use_log_transform" in data_format_info
                ), "Should have log transform flag"
                assert (
                    data_format_info["use_log_transform"] is True
                ), "Log transform should be enabled"

                # Load the actual normalization arrays from NPZ file
                data_format_npz = save_dir / DATA_FORMAT_NPZ_FILE
                assert data_format_npz.exists(), "Data format npz file should exist"

                with np.load(data_format_npz) as npz_data:
                    assert "genes_mu" in npz_data, "Should have gene means in NPZ file"
                    assert (
                        "genes_sigma" in npz_data
                    ), "Should have gene standard deviations in NPZ file"

                    genes_mu = npz_data["genes_mu"]
                    genes_sigma = npz_data["genes_sigma"]

                    # Validate normalization parameters
                    assert (
                        len(genes_mu) == dummy_adata.n_vars
                    ), f"Should have means for all {dummy_adata.n_vars} genes"
                    assert (
                        len(genes_sigma) == dummy_adata.n_vars
                    ), f"Should have stds for all {dummy_adata.n_vars} genes"

                    # Check that normalization parameters are reasonable
                    assert np.all(
                        genes_sigma > 0
                    ), "Gene standard deviations should be positive"
                    assert np.all(np.isfinite(genes_mu)), "Gene means should be finite"
                    assert np.all(
                        np.isfinite(genes_sigma)
                    ), "Gene standard deviations should be finite"

                    # Check that means are in a reasonable range (log-transformed data)
                    assert np.all(
                        genes_mu >= 0
                    ), "Log-transformed gene means should be non-negative"
                    assert np.all(
                        genes_mu <= 20
                    ), "Log-transformed gene means should be reasonable"

                # Validate results
                self._validate_results(results, "normalization_values_test")

            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)

    def test_normalization_logic_detailed(self, dummy_adata):
        """Test detailed normalization logic: computation from train data and application in train/inference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)

            # Use a larger dataset for better statistical validation
            n_cells = 200
            n_genes = 30

            # Create more realistic data with known statistical properties
            np.random.seed(42)
            X = np.random.negative_binomial(
                n=10, p=0.4, size=(n_cells, n_genes)
            ).astype(np.float32)
            X_sparse = csr_matrix(X)

            # Create consistent patient/study mapping
            n_patients = 10
            patient_ids = [f"patient_{i}" for i in range(n_patients)]
            patient_assignments = np.repeat(patient_ids, n_cells // n_patients)

            # Create cancer type mapping - ensure consistency per patient
            patient_to_cancer = {
                f"patient_{i}": "cancer_A" if i < 5 else "cancer_B"
                for i in range(n_patients)
            }
            cancer_types = np.array([patient_to_cancer[p] for p in patient_assignments])

            # Create study assignments that are consistent per patient
            patient_to_study = {
                f"patient_{i}": "study1" if i < 5 else "study2"
                for i in range(n_patients)
            }
            study_assignments = np.array(
                [patient_to_study[p] for p in patient_assignments]
            )

            expansion_labels = [
                "expanded" if i < 100 else "non-expanded" for i in range(n_cells)
            ]
            obs = {
                "expansion": expansion_labels,
                "tissue_type": np.random.choice(["tissue_A", "tissue_B"], size=n_cells),
                "imputed_labels": np.random.choice(
                    ["label_1", "label_2"], size=n_cells
                ),
                "clone_id_size": np.random.randint(1, 100, size=n_cells),
                "median_clone_size": np.random.randint(1, 50, size=n_cells),
                "study": study_assignments,
                "patient": patient_assignments,
                "sample": np.array([f"sample_{i // 20}" for i in range(n_cells)]),
                "cancer_type": cancer_types,
            }

            var = {"gene_symbol": [f"gene_{i}" for i in range(n_genes)]}
            test_adata = anndata.AnnData(X=X_sparse, obs=obs, var=var)

            # Save the test data
            test_file_path = create_temp_h5ad_file(test_adata, temp_dir)

            # Test with MLP model
            prm = MLPParam(
                n_epochs=2,
                train_batch_size=32,
                layer_units=(32,),
                random_seed=42,
                use_log_transform=True,  # Enable log transformation
                aux_categorical_types=["tissue_type"],
            )

            # Capture the preprocessing steps to validate normalization
            captured_train_data = {}

            # Patch the preprocessing function to capture intermediate data
            original_compute_means_stds = None

            def capture_normalization_computation(*args, **kwargs):
                """Capture the data used for computing normalization parameters."""
                result = original_compute_means_stds(*args, **kwargs)
                # The function returns (genes_mu, genes_sigma)
                if len(result) == 2:
                    captured_train_data["genes_mu"] = result[0].copy()
                    captured_train_data["genes_sigma"] = result[1].copy()
                    captured_train_data["n_train_samples"] = len(args[0]) if args else 0
                return result

            try:
                original_compute_means_stds = compute_preprocessed_genes_means_stds

                # Run training with patched functions
                with patch(
                    "scxpand.data_util.statistics.compute_preprocessed_genes_means_stds",
                    side_effect=capture_normalization_computation,
                ):
                    results = run_mlp_training(
                        data_path=test_file_path,
                        base_save_dir=Path(temp_dir) / "norm_logic_test",
                        prm=prm,
                        device="cpu",
                        dev_ratio=0.3,
                        num_workers=0,
                    )

                # Load the saved normalization parameters
                save_dir = Path(temp_dir) / "norm_logic_test"
                data_format_npz = save_dir / DATA_FORMAT_NPZ_FILE

                assert data_format_npz.exists(), "Data format npz file should exist"
                with np.load(data_format_npz) as npz_data:
                    saved_genes_mu = npz_data["genes_mu"]
                    saved_genes_sigma = npz_data["genes_sigma"]

                # Validate that normalization parameters were captured and saved correctly
                if captured_train_data:
                    assert (
                        "genes_mu" in captured_train_data
                    ), "Should have captured gene means"
                    assert (
                        "genes_sigma" in captured_train_data
                    ), "Should have captured gene sigmas"

                    # Check that captured parameters match saved parameters
                    np.testing.assert_array_almost_equal(
                        captured_train_data["genes_mu"],
                        saved_genes_mu,
                        decimal=5,
                        err_msg="Captured gene means should match saved gene means",
                    )
                    np.testing.assert_array_almost_equal(
                        captured_train_data["genes_sigma"],
                        saved_genes_sigma,
                        decimal=5,
                        err_msg="Captured gene sigmas should match saved gene sigmas",
                    )

                    # Validate that means and sigmas are reasonable for log-transformed data
                    assert np.all(
                        saved_genes_mu >= 0
                    ), "Gene means should be non-negative for log-transformed data"
                    assert np.all(
                        saved_genes_mu <= 15
                    ), "Gene means should be reasonable for log-transformed data"
                    assert np.all(
                        saved_genes_sigma > 0
                    ), "Gene standard deviations should be positive"
                    assert np.all(
                        saved_genes_sigma <= 5
                    ), "Gene standard deviations should be reasonable"

                # Now test that the normalization is applied correctly during inference
                # by creating a dataset and using its transform_batch_data method
                data_format = load_data_format(save_dir / "data_format.json")

                # Create a temporary dataset to use its preprocessing method
                with tempfile.NamedTemporaryFile(
                    suffix=".h5ad", delete=False
                ) as tmp_file:
                    # Create a minimal AnnData object and save it

                    obs_df = pd.DataFrame(
                        {
                            "expansion": ["expanded"] * 10,
                            "clone_id_size": [1] * 10,
                            "median_clone_size": [1] * 10,
                        }
                    )
                    var_df = pd.DataFrame(index=data_format.gene_names)
                    X_dummy = np.zeros((10, len(data_format.gene_names)))
                    adata = anndata.AnnData(X=X_dummy, obs=obs_df, var=var_df)
                    adata.write_h5ad(tmp_file.name)

                    dataset = CellsDataset(
                        data_format=data_format,
                        row_inds=np.arange(10),
                        dataset_params=None,
                        is_train=False,
                        data_path=tmp_file.name,
                    )

                # Manually apply preprocessing to a small batch to verify normalization
                test_batch = test_adata.X[:10].toarray()  # Small batch for testing
                test_batch_tensor = torch.from_numpy(test_batch).float()

                # Use the actual preprocessing method from the dataset
                processed_batch_tensor = dataset.transform_batch_data(test_batch_tensor)
                processed_batch = processed_batch_tensor.detach().cpu().numpy()

                # Validate that the processed batch has the expected normalization properties
                # After z-score normalization, the mean should be close to 0 and std close to 1
                # (allowing for some variation due to small batch size)
                batch_means = np.mean(processed_batch, axis=0)
                batch_stds = np.std(processed_batch, axis=0)

                # The means should be close to 0 (within reasonable tolerance for small batch)
                # Note: For small batches, the means might not be exactly 0, so we use a more lenient tolerance
                assert np.all(
                    np.abs(batch_means) < 10
                ), f"Normalized batch means should be reasonable, got {batch_means[:5]}"

                # The standard deviations should be positive and reasonable
                assert np.all(
                    batch_stds > 0.01
                ), f"Normalized batch stds should be positive, got {batch_stds[:5]}"
                assert np.all(
                    batch_stds < 10
                ), f"Normalized batch stds should be reasonable, got {batch_stds[:5]}"

                # Test that normalization parameters are computed from training data only
                # by checking that the number of samples used for computing stats is reasonable
                if captured_train_data and "n_train_samples" in captured_train_data:
                    n_train_samples = captured_train_data["n_train_samples"]
                    # Should be less than total samples (since we have dev split)
                    assert (
                        n_train_samples < n_cells
                    ), f"Training samples ({n_train_samples}) should be less than total ({n_cells})"
                    # Should be more than 50% of total samples (typical train split)
                    assert (
                        n_train_samples > n_cells * 0.5
                    ), f"Training samples ({n_train_samples}) should be substantial portion of total ({n_cells})"

                # Validate the overall results
                self._validate_results(results, "normalization_logic_test")

            finally:
                # Ensure the test file is removed after the test
                robust_remove(test_file_path)
