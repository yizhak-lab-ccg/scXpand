"""Integration tests for normalization functions with LightGBM and dataset components."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import CellsDataset, cells_collate_fn
from scxpand.data_util.prepare_data_for_train import prepare_data_for_training
from scxpand.data_util.transforms import (
    DEFAULT_EPS,
    apply_row_normalization,
    apply_zscore_normalization,
    load_and_preprocess_data_numpy,
    preprocess_expression_data,
)
from scxpand.util.classes import DataAugmentParams


class TestLightGBMIntegration:
    """Test integration of normalization functions with LightGBM pipeline."""

    @pytest.fixture
    def mock_adata_with_labels(self):
        """Create mock AnnData with proper columns for LightGBM testing."""
        n_cells, n_genes = 100, 20

        # Create sparse matrix with realistic gene expression values
        np.random.seed(42)
        X = csr_matrix(np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32))

        # Create observation DataFrame with consistent patient-level attributes
        # First create patient-level data to ensure consistency
        patients = [
            ("study1", "patient1", "cancer1"),
            ("study1", "patient2", "cancer2"),
            ("study1", "patient3", "cancer1"),
            ("study2", "patient1", "cancer1"),
            ("study2", "patient2", "cancer2"),
            ("study2", "patient3", "cancer2"),
        ]

        # Assign cells to patients
        patient_assignments = np.random.choice(len(patients), size=n_cells)

        obs = pd.DataFrame(
            {
                "expansion": np.random.choice([True, False], size=n_cells),
                "study": [patients[i][0] for i in patient_assignments],
                "patient": [patients[i][1] for i in patient_assignments],
                "cancer_type": [patients[i][2] for i in patient_assignments],
                "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells),
                "imputed_labels": np.random.choice(["label1", "label2"], size=n_cells),
                "sample": np.random.choice(
                    ["sample1", "sample2", "sample3", "sample4"], size=n_cells
                ),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata

    def test_lightgbm_preprocessing_consistency(self, mock_adata_with_labels):
        """Test that LightGBM preprocessing produces consistent results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save adata to file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            mock_adata_with_labels.write_h5ad(data_file)

            # Prepare data for training
            data_bundle = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=("tissue_type", "imputed_labels"),
                use_log_transform=True,
                save_dir=tmp_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=False,
            )

            adata = data_bundle.adata
            data_format = data_bundle.data_format
            row_inds_train = data_bundle.row_inds_train[:10]  # Use subset for speed

            # Test preprocessing consistency
            X_processed = load_and_preprocess_data_numpy(
                data_path=data_file, row_indices=row_inds_train, data_format=data_format
            )

            # Verify preprocessing was applied correctly
            assert X_processed.shape == (len(row_inds_train), data_format.n_genes)
            assert X_processed.dtype == np.float32
            assert not np.any(np.isnan(X_processed))
            assert not np.any(np.isinf(X_processed))

            # Verify z-score normalization was applied (values should be centered around 0)
            # After z-score normalization, values can be negative (that's expected)
            assert abs(X_processed.mean()) < 1.0  # Should be roughly centered around 0

            # Close the file handle
            adata.file.close()

    def test_lightgbm_preprocessing_vs_manual(self, mock_adata_with_labels):
        """Test that new preprocessing matches manual preprocessing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save adata to file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            mock_adata_with_labels.write_h5ad(data_file)

            # Prepare data for training
            data_bundle = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=(),
                use_log_transform=True,
                save_dir=tmp_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=False,
            )

            adata = data_bundle.adata
            data_format = data_bundle.data_format
            row_inds = data_bundle.row_inds_train[:5]  # Small subset

            # Method 1: Use new function
            X_new = load_and_preprocess_data_numpy(
                data_path=data_file, data_format=data_format, row_indices=row_inds
            )

            # Method 2: Manual preprocessing (old way)
            X_raw = adata.X[row_inds]
            if hasattr(X_raw, "toarray"):
                X_raw = X_raw.toarray()
            else:
                X_raw = np.array(X_raw, dtype=np.float32)

            # Apply manual preprocessing steps
            # In the new design, scaling factors are computed on-the-fly
            # We don't need to check for precomputed scaling factors

            # Apply row normalization manually
            row_sums = X_raw.sum(axis=1, keepdims=True)
            scaling_factors = np.where(
                row_sums > 0, data_format.target_sum / row_sums, 1.0
            )
            X_manual = X_raw * scaling_factors

            if data_format.use_log_transform:
                X_manual = np.log1p(X_manual)

            X_manual = (X_manual - data_format.genes_mu) / (
                data_format.genes_sigma + DEFAULT_EPS
            )  # Use same eps as the transforms

            # Should be identical
            assert np.allclose(X_new, X_manual, rtol=1e-6)

            # Close the file handle
            adata.file.close()


class TestDatasetIntegration:
    """Test integration of normalization functions with dataset components."""

    @pytest.fixture
    def mock_dataset_setup(self):
        """Create mock dataset setup for testing."""
        n_cells, n_genes = 50, 10

        # Create sparse matrix with realistic gene expression values
        np.random.seed(42)
        X = csr_matrix(np.random.poisson(3, size=(n_cells, n_genes)).astype(np.float32))

        # Create observation DataFrame with required columns for dataset
        obs = pd.DataFrame(
            {
                "expansion": np.random.choice([True, False], size=n_cells),
                "clone_id_size": np.random.randint(1, 100, size=n_cells),
                "median_clone_size": np.random.randint(1, 50, size=n_cells),
                "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells),
                "imputed_labels": np.random.choice(["label1", "label2"], size=n_cells),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs, var=var)

        # Create data format
        data_format = DataFormat(
            gene_names=[f"gene_{i}" for i in range(n_genes)],
            n_genes=n_genes,
            genes_mu=np.random.rand(n_genes).astype(np.float32),
            genes_sigma=np.random.rand(n_genes).astype(np.float32) + 0.1,
            target_sum=1e4,
            use_log_transform=True,
            aux_categorical_types=("tissue_type", "imputed_labels"),
        )

        return adata, data_format

    def test_dataset_preprocessing_consistency(self, mock_dataset_setup):
        """Test that dataset preprocessing produces consistent results."""
        adata, data_format = mock_dataset_setup

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save adata to file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            adata.write_h5ad(data_file)

            # Create dataset parameters
            dataset_params = DataAugmentParams(
                soft_loss_beta=1.0,
                mask_rate=0.0,
                noise_std=0.0,
            )

            # Create dataset
            row_inds = np.arange(10)  # Use subset
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=row_inds,
                dataset_params=dataset_params,
                is_train=True,
                data_path=data_file,
            )

            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )

            # Get a batch
            batch = next(iter(dataloader))

            # Verify batch structure
            assert "x" in batch
            assert batch["x"].shape[0] == 5  # batch size
            assert batch["x"].shape[1] == data_format.n_genes
            assert batch["x"].dtype == torch.float32
            assert not torch.any(torch.isnan(batch["x"]))
            assert not torch.any(torch.isinf(batch["x"]))

    def test_dataset_vs_direct_preprocessing(self, mock_dataset_setup):
        """Test that dataset preprocessing matches direct function calls."""
        adata, data_format = mock_dataset_setup

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save adata to file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            adata.write_h5ad(data_file)

            # Method 1: Use dataset preprocessing
            dataset_params = DataAugmentParams(
                soft_loss_beta=1.0,
                mask_rate=0.0,  # No augmentation for comparison
                noise_std=0.0,
            )

            row_inds = np.array([0, 1, 2])
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=row_inds,
                dataset_params=dataset_params,
                is_train=False,  # No augmentation
                data_path=data_file,
            )

            # Get batch through dataset
            dataloader = DataLoader(
                dataset,
                batch_size=3,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )
            batch = next(iter(dataloader))
            X_dataset = batch["x"]

            # Method 2: Direct preprocessing
            # Load raw data
            adata_loaded = ad.read_h5ad(data_file, backed="r")
            X_raw = adata_loaded.X[row_inds]
            if hasattr(X_raw, "toarray"):
                X_raw = X_raw.toarray()
            else:
                X_raw = np.array(X_raw, dtype=np.float32)

            # Apply direct preprocessing
            X_direct_torch = torch.from_numpy(X_raw).float()
            X_direct_processed = preprocess_expression_data(
                X_direct_torch.clone(), data_format=data_format
            )

            # Should be very close (allowing for minor floating point differences)
            assert torch.allclose(X_dataset, X_direct_processed, rtol=1e-5, atol=1e-6)

            adata_loaded.file.close()

    def test_dataset_gene_transformation_with_preprocessing(self, mock_dataset_setup):
        """Test dataset gene transformation combined with preprocessing."""
        adata, data_format = mock_dataset_setup

        # Create data format with different gene order to trigger transformation
        reordered_genes = list(reversed(data_format.gene_names))
        data_format_reordered = DataFormat(
            gene_names=reordered_genes,
            n_genes=data_format.n_genes,
            genes_mu=data_format.genes_mu[
                ::-1
            ].copy(),  # Reverse and copy to avoid negative stride
            genes_sigma=data_format.genes_sigma[::-1].copy(),
            target_sum=data_format.target_sum,
            use_log_transform=data_format.use_log_transform,
            aux_categorical_types=data_format.aux_categorical_types,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save adata to file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            adata.write_h5ad(data_file)

            # Create dataset with gene transformation
            dataset_params = DataAugmentParams(
                soft_loss_beta=1.0,
                mask_rate=0.0,
                noise_std=0.0,
            )

            row_inds = np.array([0, 1])
            dataset = CellsDataset(
                data_format=data_format_reordered,
                row_inds=row_inds,
                dataset_params=dataset_params,
                is_train=False,
                data_path=data_file,
            )

            # Verify that gene transformation is needed
            assert dataset.needs_gene_transformation

            # Get batch
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )
            batch = next(iter(dataloader))
            X_transformed = batch["x"]

            # Verify proper shape and no NaN/Inf values
            assert X_transformed.shape == (2, data_format_reordered.n_genes)
            assert not torch.any(torch.isnan(X_transformed))
            assert not torch.any(torch.isinf(X_transformed))


class TestDirectNormalizationUsage:
    """Test direct usage of normalization functions without complex integrations."""

    def test_direct_numpy_preprocessing(self):
        """Test direct usage of numpy preprocessing functions."""
        # Create simple test data
        X = np.random.rand(10, 5).astype(np.float32) * 100

        data_format = DataFormat(
            gene_names=[f"gene_{i}" for i in range(5)],
            n_genes=5,
            genes_mu=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            target_sum=1e4,
            use_log_transform=True,
            aux_categorical_types=(),
        )

        # Apply preprocessing
        X_processed = preprocess_expression_data(X.copy(), data_format)

        # Verify results
        assert X_processed.shape == X.shape
        assert X_processed.dtype == np.float32
        assert not np.allclose(X_processed, X)  # Should be different
        assert not np.any(np.isnan(X_processed))
        assert not np.any(np.isinf(X_processed))

    def test_direct_torch_preprocessing(self):
        """Test direct usage of torch preprocessing functions."""
        # Create simple test data
        X = torch.rand(10, 5).float() * 100

        data_format = DataFormat(
            gene_names=[f"gene_{i}" for i in range(5)],
            n_genes=5,
            genes_mu=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            target_sum=1e4,
            use_log_transform=True,
            aux_categorical_types=(),
        )

        # Apply preprocessing
        X_processed = preprocess_expression_data(X.clone(), data_format)

        # Verify results
        assert X_processed.shape == X.shape
        assert X_processed.dtype == torch.float32
        assert not torch.allclose(X_processed, X)  # Should be different
        assert not torch.any(torch.isnan(X_processed))
        assert not torch.any(torch.isinf(X_processed))


class TestBackwardCompatibility:
    """Test that new functions maintain backward compatibility."""

    def test_preprocessing_module_compatibility(self):
        """Test that existing preprocessing module functions still work."""
        # Test that apply_zscore_normalization works as expected
        X = np.random.rand(10, 5).astype(np.float32)
        X_original = X.copy()
        genes_mu = np.random.rand(5).astype(np.float32)
        genes_sigma = np.random.rand(5).astype(np.float32) + 0.1

        # This should work without error
        result = apply_zscore_normalization(X, genes_mu, genes_sigma)

        # Verify it was applied in-place
        assert not np.allclose(X, X_original)  # Should be different
        assert np.allclose(X, result)  # Result should match the modified X

    def test_dataset_backward_compatibility(self):
        """Test that dataset still works with existing interfaces."""
        # This test ensures that the dataset changes don't break existing code

        # Should be able to import without errors
        assert CellsDataset is not None
        assert cells_collate_fn is not None
        assert DataAugmentParams is not None


class TestEpsilonIntegration:
    """Test epsilon consistency in integration scenarios."""

    def test_notebook_scenario_epsilon_consistency(self):
        """Test the exact scenario from the notebook with epsilon consistency."""
        # Reproduce the notebook bug scenario with epsilon fix
        np.random.seed(12345)

        n_cells = 500
        n_genes = 50

        # Create data similar to real scRNA-seq with sparse gene
        X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

        # Create a very sparse gene (like ENSG00000153563 in the notebook)
        sparse_gene_idx = 25
        X[:, sparse_gene_idx] = 0
        expressing_cells = np.random.choice(n_cells, size=15, replace=False)
        X[expressing_cells, sparse_gene_idx] = np.random.choice([1, 2, 3, 4], size=15)

        # Create AnnData with metadata similar to notebook
        obs = pd.DataFrame(
            {
                "imputed_labels": np.random.choice(
                    ["is_CD4", "is_CD8", "is_NK"], n_cells
                ),
                "tissue_type": np.random.choice(["Tumor", "Normal"], n_cells),
                "patient_id": np.random.choice(["P1", "P2", "P3"], n_cells),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=[f"GENE_{i:04d}" for i in range(n_genes)])

        adata = ad.AnnData(X=csr_matrix(X), obs=obs, var=var)

        # Filter to CD4 Tumor cells (like in notebook)
        cd4_tumor_mask = (obs["imputed_labels"] == "is_CD4") & (
            obs["tissue_type"] == "Tumor"
        )
        filtered_indices = np.where(cd4_tumor_mask)[0]

        # Create data format with scaling factors
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=list(adata.var_names),
            target_sum=10000.0,
            use_log_transform=False,  # Match notebook
        )

        # Create scaling factors for training data
        train_indices = np.arange(
            min(400, len(filtered_indices))
        )  # Subset for training

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "notebook_test.h5ad"
            adata.write_h5ad(temp_path)

            data_format.create_data_format(
                data_path=temp_path,
                adata=adata,
                row_inds_train=train_indices,
            )

            # Test the problematic sparse gene
            sparse_gene_name = adata.var_names[sparse_gene_idx]

            # Method 1: Batch processing (CellsDataset approach)

            eval_indices = filtered_indices[:50]  # Subset for testing
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=eval_indices,
                dataset_params=DataAugmentParams(
                    soft_loss_beta=1.0, mask_rate=0.0, noise_std=0.0
                ),
                is_train=False,
                data_path=temp_path,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=50,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )

            batch = next(iter(dataloader))
            expr_batch = batch["x"].numpy()
            gene_expr_batch = expr_batch[:, sparse_gene_idx]

            # Method 2: Use load_and_preprocess_data_numpy for consistency
            # This should produce identical results to CellsDataset since it uses the same approach
            expr_full = load_and_preprocess_data_numpy(
                data_path=temp_path,
                data_format=data_format,
                row_indices=eval_indices,
                gene_subset=[sparse_gene_name],
            ).flatten()

            # With shared epsilon, results should be essentially identical (fixing the notebook bug)
            # Allow for small floating-point precision differences
            np.testing.assert_allclose(
                gene_expr_batch,
                expr_full,
                rtol=1e-6,  # Relaxed tolerance for floating-point precision
                atol=1e-6,  # Relaxed tolerance for floating-point precision
                err_msg="Notebook scenario: batch vs full should be essentially identical with shared epsilon",
            )

            # Statistics should match essentially exactly (allowing for floating-point precision)
            assert abs(gene_expr_batch.mean() - expr_full.mean()) < 1e-6
            assert abs(gene_expr_batch.std() - expr_full.std()) < 1e-6

            # Both should handle the sparse gene correctly
            nonzero_batch = np.count_nonzero(gene_expr_batch)
            nonzero_full = np.count_nonzero(expr_full)
            assert nonzero_batch == nonzero_full

    def test_dataset_vs_manual_epsilon_consistency(self):
        """Test CellsDataset vs manual preprocessing with explicit epsilon control."""
        # Create test data
        np.random.seed(999)
        n_cells = 200
        n_genes = 30

        X = np.random.exponential(scale=1.5, size=(n_cells, n_genes)).astype(np.float32)

        obs = pd.DataFrame(
            {
                "expansion": np.random.choice([True, False], n_cells),
                "cell_type": np.random.choice(["A", "B", "C"], n_cells),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=csr_matrix(X), obs=obs, var=var)

        # Create data format
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=list(adata.var_names),
            target_sum=10000.0,
            use_log_transform=True,
            aux_categorical_types=("cell_type",),
        )

        # Create with scaling factors
        train_indices = np.arange(150)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "epsilon_integration_test.h5ad"
            adata.write_h5ad(temp_path)

            data_format.create_data_format(
                data_path=temp_path,
                adata=adata,
                row_inds_train=train_indices,
            )

            # Test different epsilon values
            eps_values = [
                1e-10,
                1e-3,
                1e-1,
            ]  # High precision, medium, large (ensure measurable differences)

            test_indices = np.arange(50)

            for eps_val in eps_values:
                # Method 1: Dataset approach (we can't easily control epsilon here,
                # so we'll test that it uses 1e-10 internally)
                if (
                    eps_val == 1e-10
                ):  # Only test the epsilon that CellsDataset actually uses
                    dataset = CellsDataset(
                        data_format=data_format,
                        row_inds=test_indices,
                        dataset_params=DataAugmentParams(
                            soft_loss_beta=1.0, mask_rate=0.0, noise_std=0.0
                        ),
                        is_train=False,
                        data_path=temp_path,
                    )

                    # Create collate function with proper closure
                    def collate_fn_for_dataset(batch_indices, ds=dataset):
                        return cells_collate_fn(batch_indices, ds)

                    dataloader = DataLoader(
                        dataset,
                        batch_size=50,
                        shuffle=False,
                        collate_fn=collate_fn_for_dataset,
                    )

                    batch = next(iter(dataloader))
                    expr_dataset = batch["x"].numpy()

                # Method 2: Manual approach with controlled epsilon
                X_raw_subset = adata.X[test_indices, :].toarray()

                # In the new design, scaling factors are computed on-the-fly
                # We don't need to get precomputed scaling factors

                expr_manual = preprocess_expression_data(
                    X=torch.from_numpy(X_raw_subset).float(),
                    data_format=data_format,
                    eps=eps_val,
                ).numpy()

                # When eps matches CellsDataset's internal epsilon (1e-10), results should be identical
                if eps_val == 1e-10:
                    np.testing.assert_allclose(
                        expr_dataset,
                        expr_manual,
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"Dataset vs manual should be identical with eps={eps_val}",
                    )

                # For different epsilon values, verify they produce different results
                if eps_val != 1e-10:
                    # This manual result should differ from the dataset result (which uses 1e-10)
                    max_diff = np.abs(expr_dataset - expr_manual).max()
                    assert (
                        max_diff > 1e-8
                    ), f"Different epsilon {eps_val} should produce measurable differences from 1e-10"

    def test_shared_epsilon_constant_usage(self):
        """Test using a shared epsilon constant across the codebase."""
        # Define shared epsilon constant (like in the notebook fix)
        SHARED_EPS = 1e-10

        # Create simple test data
        np.random.seed(42)
        n_cells = 100
        n_genes = 20
        X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
            target_sum=10000.0,
            use_log_transform=False,
            genes_mu=np.random.normal(0, 1, n_genes).astype(np.float32),
            genes_sigma=np.random.uniform(0.5, 2.0, n_genes).astype(np.float32),
        )

        # Test that using SHARED_EPS consistently gives reproducible results
        results = []

        for _i in range(3):  # Run multiple times
            X_copy = X.copy()
            result = preprocess_expression_data(
                X=X_copy, data_format=data_format, eps=SHARED_EPS
            )
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0],
                results[i],
                rtol=1e-12,
                atol=1e-14,
                err_msg="Shared epsilon should give identical results across runs",
            )

        # Test that different epsilon gives different results
        result_different_eps = preprocess_expression_data(
            X=X.copy(),
            data_format=data_format,
            eps=1e-6,  # Different from SHARED_EPS
        )

        max_diff = np.abs(results[0] - result_different_eps).max()
        assert (
            max_diff > 1e-12
        ), "Different epsilon should produce measurably different results"


class TestPerformance:
    """Test performance characteristics of new functions."""

    def test_in_place_vs_copy_memory_usage(self):
        """Test that in-place operations actually modify original arrays."""
        # Large array to test memory behavior
        X_np = np.random.rand(1000, 100).astype(np.float32)
        X_torch = torch.rand(1000, 100).float()

        # Store original ids
        original_np_id = id(X_np)
        original_torch_id = id(X_torch)

        # Test in-place operations

        result_np = apply_row_normalization(X_np, target_sum=1000)
        result_torch = apply_row_normalization(X_torch, target_sum=1000)

        # Should be same objects
        assert id(result_np) == original_np_id
        assert id(result_torch) == original_torch_id

    def test_preprocessing_pipeline_efficiency(self):
        """Test that complete preprocessing pipeline is efficient."""
        # Create realistic sized data
        X = np.random.rand(1000, 500).astype(np.float32) * 100

        data_format = DataFormat(
            gene_names=[f"gene_{i}" for i in range(500)],
            n_genes=500,
            genes_mu=np.random.rand(500).astype(np.float32),
            genes_sigma=np.random.rand(500).astype(np.float32) + 0.1,
            target_sum=1e4,
            use_log_transform=True,
            aux_categorical_types=(),
        )

        # Should complete without error in reasonable time
        result = preprocess_expression_data(X.copy(), data_format)

        assert result.shape == X.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
