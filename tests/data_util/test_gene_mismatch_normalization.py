"""Tests for normalization with gene mismatches during inference.

This tests the complete pipeline when inference data has different genes than training data.
"""

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
from scxpand.data_util.transforms import (
    DEFAULT_EPS,
    DEFAULT_SIGMA_CLIP_FACTOR,
    apply_row_normalization,
    load_and_preprocess_data_numpy,
    preprocess_expression_data,
)
from scxpand.util.classes import DataAugmentParams


class TestGeneMismatchNormalization:
    """Test normalization when inference data has different genes than training data."""

    @pytest.fixture
    def training_data_format(self):
        """Create a training data format with specific genes."""
        return DataFormat(
            gene_names=["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"],
            n_genes=5,
            genes_mu=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            target_sum=1e4,
            use_log_transform=True,
            aux_categorical_types=(),
        )

    @pytest.fixture
    def training_adata(self):
        """Create training AnnData with specific genes."""
        n_cells, n_genes = 100, 5
        X = csr_matrix(np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32))

        obs = pd.DataFrame(
            {
                "expansion": np.random.choice([True, False], size=n_cells),
                "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"])

        return ad.AnnData(X=X, obs=obs, var=var)

    def create_inference_adata(self, gene_scenario: str) -> ad.AnnData:
        """Create inference AnnData with different gene scenarios."""
        n_cells = 50

        if gene_scenario == "perfect_match":
            # Same genes as training
            gene_names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
            n_genes = 5
        elif gene_scenario == "missing_genes":
            # Missing some training genes
            gene_names = ["GENE_A", "GENE_C", "GENE_E"]  # Missing B and D
            n_genes = 3
        elif gene_scenario == "extra_genes":
            # Has all training genes plus extra ones
            gene_names = [
                "GENE_A",
                "GENE_B",
                "GENE_C",
                "GENE_D",
                "GENE_E",
                "GENE_F",
                "GENE_G",
            ]
            n_genes = 7
        elif gene_scenario == "reordered_genes":
            # Same genes but different order
            gene_names = ["GENE_E", "GENE_C", "GENE_A", "GENE_D", "GENE_B"]
            n_genes = 5
        elif gene_scenario == "mixed_scenario":
            # Missing some, has extra, different order
            gene_names = [
                "GENE_F",
                "GENE_C",
                "GENE_A",
                "GENE_G",
            ]  # Missing B,D,E + extra F,G + reordered
            n_genes = 4
        else:
            raise ValueError(f"Unknown scenario: {gene_scenario}")

        X = csr_matrix(np.random.poisson(3, size=(n_cells, n_genes)).astype(np.float32))

        obs = pd.DataFrame(
            {
                "expansion": np.random.choice([True, False], size=n_cells),
                "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=gene_names)

        return ad.AnnData(X=X, obs=obs, var=var)

    def test_dataset_gene_transformation_scenarios(self, training_data_format):
        """Test that CellsDataset correctly handles different gene scenarios."""
        scenarios = [
            "perfect_match",
            "missing_genes",
            "extra_genes",
            "reordered_genes",
            "mixed_scenario",
        ]

        for scenario in scenarios:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create inference data with this scenario
                inference_adata = self.create_inference_adata(scenario)

                # Save to file
                inference_file = Path(tmp_dir) / f"inference_{scenario}.h5ad"
                inference_adata.write_h5ad(inference_file)

                # Create dataset (this should handle gene transformation internally)
                dataset = CellsDataset(
                    data_format=training_data_format,
                    row_inds=np.arange(10),  # Use subset for speed
                    dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                    is_train=False,  # Inference mode
                    data_path=inference_file,
                )

                # Verify gene transformation was detected correctly
                if scenario == "perfect_match":
                    assert not dataset.needs_gene_transformation, (
                        f"Should not need transformation for {scenario}"
                    )
                else:
                    assert dataset.needs_gene_transformation, (
                        f"Should need transformation for {scenario}"
                    )

                # Test batch loading
                def make_collate_fn(ds):
                    return lambda batch_indices: cells_collate_fn(batch_indices, ds)

                dataloader = DataLoader(
                    dataset,
                    batch_size=5,
                    shuffle=False,
                    collate_fn=make_collate_fn(dataset),
                )

                batch = next(iter(dataloader))

                # Verify output shape is correct (should match training format)
                assert batch["x"].shape[1] == training_data_format.n_genes, (
                    f"Output genes {batch['x'].shape[1]} != training genes {training_data_format.n_genes} for {scenario}"
                )

                # Verify no NaN/Inf values after normalization
                assert not torch.any(torch.isnan(batch["x"])), (
                    f"NaN values found in {scenario}"
                )
                assert not torch.any(torch.isinf(batch["x"])), (
                    f"Inf values found in {scenario}"
                )

                # Verify the data is properly normalized (z-score should have reasonable range)
                assert batch["x"].abs().max() < 50, (
                    f"Z-score values too large in {scenario}: {batch['x'].abs().max()}"
                )

    def test_missing_genes_filled_with_zeros(self, training_data_format):
        """Test that missing genes are correctly filled with zeros before normalization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create data missing genes B and D
            inference_adata = self.create_inference_adata("missing_genes")
            inference_file = Path(tmp_dir) / "inference_missing.h5ad"
            inference_adata.write_h5ad(inference_file)

            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(5),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=inference_file,
            )

            # Check that gene indices mapping is correct
            expected_indices = [0, -1, 1, -1, 2]  # A, missing, C, missing, E
            assert np.array_equal(dataset.gene_indices, expected_indices), (
                f"Gene indices wrong: {dataset.gene_indices} != {expected_indices}"
            )

            # Test that transformation creates zeros for missing genes
            raw_data = torch.rand(
                3, 3, dtype=torch.float32
            )  # [batch_size, n_raw_genes]
            transformed = dataset.transform_batch_data(raw_data, in_place=False)

            # Check that missing gene positions (1 and 3) are processed correctly
            # After normalization, missing genes should have specific z-score values
            # Since missing genes start as 0, after row norm they stay 0, after log they stay 0,
            # after z-score they become -mu/sigma
            expected_missing_gene_b = -training_data_format.genes_mu[1] / (
                training_data_format.genes_sigma[1]
                + 1e-7  # High precision for gene mismatch testing
            )
            expected_missing_gene_d = -training_data_format.genes_mu[3] / (
                training_data_format.genes_sigma[3]
                + 1e-7  # High precision for gene mismatch testing
            )

            # All cells should have the same value for missing genes
            assert torch.allclose(
                transformed[:, 1], torch.tensor(expected_missing_gene_b), atol=1e-5
            ), (
                f"Missing GENE_B not handled correctly: {transformed[:, 1]} != {expected_missing_gene_b}"
            )
            assert torch.allclose(
                transformed[:, 3], torch.tensor(expected_missing_gene_d), atol=1e-5
            ), (
                f"Missing GENE_D not handled correctly: {transformed[:, 3]} != {expected_missing_gene_d}"
            )

    def test_extra_genes_ignored(self, training_data_format):
        """Test that extra genes in inference data are correctly ignored."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create data with extra genes
            inference_adata = self.create_inference_adata("extra_genes")
            inference_file = Path(tmp_dir) / "inference_extra.h5ad"
            inference_adata.write_h5ad(inference_file)

            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(5),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=inference_file,
            )

            # Verify gene mapping ignores extra genes
            expected_indices = [
                0,
                1,
                2,
                3,
                4,
            ]  # Perfect match for first 5 genes, ignore F and G
            assert np.array_equal(dataset.gene_indices, expected_indices), (
                f"Gene indices wrong: {dataset.gene_indices} != {expected_indices}"
            )

            # Test batch processing
            def collate_fn(batch_indices):
                return cells_collate_fn(batch_indices, dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=3,
                shuffle=False,
                collate_fn=collate_fn,
            )

            batch = next(iter(dataloader))

            # Output should only have training genes (extra genes ignored)
            assert batch["x"].shape[1] == training_data_format.n_genes

    def test_gene_reordering_preserves_values(self, training_data_format):
        """Test that gene reordering preserves the correct values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create data with reordered genes: E, C, A, D, B
            reordered_adata = self.create_inference_adata("reordered_genes")
            reordered_file = Path(tmp_dir) / "inference_reordered.h5ad"
            reordered_adata.write_h5ad(reordered_file)

            # Also create reference data with correct order
            reference_adata = self.create_inference_adata("perfect_match")
            reference_file = Path(tmp_dir) / "inference_reference.h5ad"
            reference_adata.write_h5ad(reference_file)

            # Use same raw data for both to test reordering
            np.random.seed(42)
            n_test_cells = reference_adata.n_obs
            raw_data = np.random.poisson(5, size=(n_test_cells, 5)).astype(np.float32)

            # Set the same data in both AnnData objects but in different gene orders
            reference_adata.X = csr_matrix(raw_data)  # A, B, C, D, E order

            # For reordered: genes are E, C, A, D, B, so data should be in columns [4,2,0,3,1]
            reordered_data = raw_data[:, [4, 2, 0, 3, 1]]  # Reorder to E, C, A, D, B
            reordered_adata.X = csr_matrix(reordered_data)

            # Save the modified data
            reference_adata.write_h5ad(reference_file)
            reordered_adata.write_h5ad(reordered_file)

            # Create datasets for both
            reference_dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(5),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=reference_file,
            )

            reordered_dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(5),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=reordered_file,
            )

            # Process same batch through both datasets
            batch_indices = [0, 1, 2]

            reference_batch = cells_collate_fn(batch_indices, reference_dataset)
            reordered_batch = cells_collate_fn(batch_indices, reordered_dataset)

            # Results should be identical after reordering
            assert torch.allclose(
                reference_batch["x"], reordered_batch["x"], atol=1e-5
            ), "Reordered genes should produce identical results"

    def test_inference_vs_training_consistency(self, training_data_format):
        """Test that inference normalization is consistent with training normalization approaches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test data with perfect gene match
            test_adata = self.create_inference_adata("perfect_match")
            test_file = Path(tmp_dir) / "test_data.h5ad"
            test_adata.write_h5ad(test_file)

            row_indices = np.array([0, 1, 2, 3, 4])

            # Method 1: Using dataset (inference approach)
            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=row_indices,
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=test_file,
            )

            def collate_fn(batch_indices):
                return cells_collate_fn(batch_indices, dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                collate_fn=collate_fn,
            )

            batch = next(iter(dataloader))
            X_dataset = batch["x"]

            # Method 2: Using direct normalization (training/LightGBM approach)
            X_direct = load_and_preprocess_data_numpy(
                data_path=test_file,
                row_indices=row_indices,
                data_format=training_data_format,
            )

            # Convert to tensor for comparison
            X_direct_tensor = torch.from_numpy(X_direct).float()

            # Should be identical (within floating point precision)
            assert torch.allclose(X_dataset, X_direct_tensor, rtol=1e-5, atol=1e-6), (
                "Dataset and direct normalization should produce identical results"
            )

    def test_normalization_with_edge_cases(self, training_data_format):
        """Test normalization handles edge cases correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create data with edge cases
            inference_adata = self.create_inference_adata("missing_genes")
            n_cells = inference_adata.n_obs

            # Modify to include edge cases
            X_edge = np.zeros((n_cells, 3), dtype=np.float32)  # Start with zeros

            # Add some edge cases
            X_edge[0, :] = [0, 0, 0]  # All zeros row
            if n_cells > 1:
                X_edge[1, :] = [1, 0, 0]  # Single non-zero
            if n_cells > 2:
                X_edge[2, :] = [1e6, 1e6, 1e6]  # Very large values
            if n_cells > 3:
                X_edge[3, :] = [0.001, 0.001, 0.001]  # Very small values
            if n_cells > 4:
                X_edge[4:, :] = np.random.poisson(
                    5, size=(n_cells - 4, 3)
                )  # Normal values

            inference_adata.X = csr_matrix(X_edge)

            inference_file = Path(tmp_dir) / "inference_edge.h5ad"
            inference_adata.write_h5ad(inference_file)

            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(min(10, n_cells)),  # Use up to 10 cells
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=inference_file,
            )

            batch_size = min(10, n_cells)

            def collate_fn(batch_indices):
                return cells_collate_fn(batch_indices, dataset)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

            batch = next(iter(dataloader))
            X_processed = batch["x"]

            # Verify no NaN or Inf values
            assert not torch.any(torch.isnan(X_processed)), (
                "Edge cases should not produce NaN"
            )
            assert not torch.any(torch.isinf(X_processed)), (
                "Edge cases should not produce Inf"
            )

            # Verify all-zero row is handled correctly
            # After row normalization, all-zero row stays zero
            # After log transform, zero stays zero
            # After z-score, zero becomes -mu/sigma for each gene
            expected_zero_row = torch.tensor(
                [
                    -training_data_format.genes_mu[i]
                    / (
                        training_data_format.genes_sigma[i] + 1e-7
                    )  # High precision for zero row testing
                    for i in range(training_data_format.n_genes)
                ],
                dtype=torch.float32,
            )

            assert torch.allclose(X_processed[0], expected_zero_row, atol=1e-5), (
                f"All-zero row not handled correctly: {X_processed[0]} != {expected_zero_row}"
            )


class TestGeneMismatchIntegrationWithModels:
    """Test gene mismatch scenarios with actual model inference patterns."""

    def test_inference_utils_gene_reordering(self):
        """Test that inference_utils.run_model_inference handles gene mismatches correctly."""
        # This would test the complete inference pipeline, but requires setting up models
        # For now, we'll test the data preparation part

        # Create test data format
        data_format = DataFormat(
            gene_names=["GENE_A", "GENE_B", "GENE_C"],
            n_genes=3,
            genes_mu=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7], dtype=np.float32),
            target_sum=1e4,
            use_log_transform=True,
            aux_categorical_types=(),
        )

        # Create inference data with different gene order
        n_cells = 10
        X = csr_matrix(np.random.poisson(3, size=(n_cells, 3)).astype(np.float32))
        obs = pd.DataFrame({"expansion": np.random.choice([True, False], size=n_cells)})
        var = pd.DataFrame(index=["GENE_C", "GENE_A", "GENE_B"])  # Different order

        inference_adata = ad.AnnData(X=X, obs=obs, var=var)

        # Test that prepare_adata_for_training with reorder_genes=True works
        prepared_adata = data_format.prepare_adata_for_training(
            inference_adata, reorder_genes=True
        )

        # Genes should now be in correct order
        assert list(prepared_adata.var_names) == data_format.gene_names, (
            f"Genes not reordered correctly: {list(prepared_adata.var_names)} != {data_format.gene_names}"
        )

        # Test that prepare_adata_for_training with reorder_genes=False doesn't change order
        unchanged_adata = data_format.prepare_adata_for_training(
            inference_adata, reorder_genes=False
        )
        assert list(unchanged_adata.var_names) == [
            "GENE_C",
            "GENE_A",
            "GENE_B",
        ], "Genes should not be reordered when reorder_genes=False"

        # Note: Full inference testing would require mock models, which is beyond scope here


class TestNormalizationParameterConsistency:
    """Test that normalization parameters are used consistently across different code paths."""

    def test_data_format_parameter_consistency(self):
        """Test that DataFormat parameters are used consistently."""
        data_format = DataFormat(
            gene_names=["GENE_A", "GENE_B", "GENE_C"],
            n_genes=3,
            genes_mu=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7], dtype=np.float32),
            target_sum=5000.0,  # Non-default value
            use_log_transform=False,  # Non-default value
            aux_categorical_types=(),
        )

        # Test data
        X_numpy = np.random.rand(5, 3).astype(np.float32) * 100
        X_torch = torch.from_numpy(X_numpy.copy()).float()

        # Test that both numpy and torch preprocessing use same parameters
        X_numpy_processed = preprocess_expression_data(
            X=X_numpy, data_format=data_format
        )
        X_torch_processed = preprocess_expression_data(
            X=X_torch, data_format=data_format
        )

        # Should be identical
        assert np.allclose(X_numpy_processed, X_torch_processed.numpy(), rtol=1e-6), (
            "Numpy and torch preprocessing should use identical parameters"
        )

        # Test that custom parameters are actually used
        # After row normalization, sum should be target_sum (5000, not default 10000)
        X_test = np.array([[100, 200, 300]], dtype=np.float32)

        # Apply just row normalization

        X_row_norm = apply_row_normalization(
            X=X_test, target_sum=data_format.target_sum
        )

        assert np.allclose(X_row_norm.sum(axis=1), data_format.target_sum), (
            f"Row normalization should use custom target_sum: {X_row_norm.sum()} != {data_format.target_sum}"
        )

        # Test that log transform is correctly disabled
        X_after_row_norm = X_row_norm.copy()
        X_after_row_norm_original = (
            X_after_row_norm.copy()
        )  # Keep original for expected calculation

        # Full preprocessing with log_transform=False
        X_full = preprocess_expression_data(X=X_after_row_norm, data_format=data_format)

        # If log transform was incorrectly applied, values would be different
        # Since log_transform=False, the values after row norm should only be z-score normalized
        expected_raw = (X_after_row_norm_original - data_format.genes_mu) / (
            data_format.genes_sigma + DEFAULT_EPS
        )  # Use the same eps as the transforms
        expected = np.clip(
            expected_raw, -DEFAULT_SIGMA_CLIP_FACTOR, DEFAULT_SIGMA_CLIP_FACTOR
        )

        assert np.allclose(X_full, expected, rtol=1e-6), (
            "Log transform should be disabled when use_log_transform=False"
        )
