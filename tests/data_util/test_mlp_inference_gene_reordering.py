"""Tests for MLP inference with gene reordering scenarios.

This module tests that MLP inference works correctly when the inference data
has different genes than the training data, including missing genes, extra genes,
and reordered genes.
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
from scxpand.data_util.transforms import load_and_preprocess_data_numpy
from scxpand.util.classes import DataAugmentParams


class TestMLPInferenceGeneReordering:
    """Test MLP inference scenarios with different gene configurations."""

    @pytest.fixture
    def training_data_format(self):
        """Create a training data format representing what the model was trained on."""
        return DataFormat(
            gene_names=["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"],
            n_genes=5,
            genes_mu=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            target_sum=1e4,
            use_log_transform=True,
            eps=1e-6,
            aux_categorical_types=(),
        )

    def create_inference_adata(self, scenario: str, n_cells: int = 20) -> ad.AnnData:
        """Create inference AnnData with different gene scenarios."""
        if scenario == "perfect_match":
            # Same genes as training in same order
            gene_names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
            n_genes = 5
        elif scenario == "missing_genes":
            # Missing some training genes (B and D missing)
            gene_names = ["GENE_A", "GENE_C", "GENE_E"]
            n_genes = 3
        elif scenario == "extra_genes":
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
        elif scenario == "reordered_genes":
            # Same genes but different order
            gene_names = ["GENE_E", "GENE_C", "GENE_A", "GENE_D", "GENE_B"]
            n_genes = 5
        elif scenario == "mixed_scenario":
            # Missing some (B,D,E), has extra (F,G), different order
            gene_names = ["GENE_F", "GENE_C", "GENE_A", "GENE_G"]
            n_genes = 4
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Create realistic expression data
        X = csr_matrix(np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32))

        obs = pd.DataFrame(
            {
                "expansion": np.random.choice(
                    ["expanded", "not_expanded"], size=n_cells
                ),
                "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells),
            }
        )
        obs.index = obs.index.astype(str)

        var = pd.DataFrame(index=gene_names)

        return ad.AnnData(X=X, obs=obs, var=var)

    def test_inference_normalization_with_perfect_match(self, training_data_format):
        """Test that inference normalization works with perfectly matching genes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create inference data with perfect gene match
            inference_adata = self.create_inference_adata("perfect_match")
            inference_file = Path(tmp_dir) / "inference_perfect.h5ad"
            inference_adata.write_h5ad(inference_file)

            # Method 1: Using CellsDataset (MLP inference approach)
            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(10),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,  # Inference mode
                data_path=inference_file,
            )

            # Should not need gene transformation
            assert not dataset.needs_gene_transformation

            # Get batch data

            dataloader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )
            batch = next(iter(dataloader))
            X_dataset = batch["x"]

            # Method 2: Using direct normalization approach
            X_direct = load_and_preprocess_data_numpy(
                data_path=inference_file,
                row_indices=np.arange(5),
                data_format=training_data_format,
            )
            X_direct_tensor = torch.from_numpy(X_direct).float()

            # Results should be identical
            assert torch.allclose(
                X_dataset, X_direct_tensor, rtol=1e-5, atol=1e-6
            ), "Perfect match scenario should produce identical results"

            inference_adata.file.close()

    def test_inference_normalization_with_missing_genes(self, training_data_format):
        """Test that inference normalization correctly handles missing genes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create inference data missing genes B and D
            inference_adata = self.create_inference_adata("missing_genes")
            inference_file = Path(tmp_dir) / "inference_missing.h5ad"
            inference_adata.write_h5ad(inference_file)

            # Using CellsDataset (MLP inference approach)
            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(10),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=inference_file,
            )

            # Should need gene transformation
            assert dataset.needs_gene_transformation

            # Check gene mapping is correct
            # Inference genes: [A, C, E] at indices [0, 1, 2]
            # Training genes: [A, B, C, D, E]
            # Expected mapping: [0, -1, 1, -1, 2] (A->0, B->missing, C->1, D->missing, E->2)
            expected_gene_indices = [0, -1, 1, -1, 2]
            assert np.array_equal(dataset.gene_indices, expected_gene_indices)

            # Get processed batch

            dataloader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )
            batch = next(iter(dataloader))
            X_processed = batch["x"]

            # Verify output shape is correct (should match training format)
            assert X_processed.shape[1] == training_data_format.n_genes

            # Missing genes should have specific z-score values (since they start as 0)
            # After preprocessing: 0 -> 0 (row norm) -> 0 (log) -> -mu/sigma (z-score)
            expected_missing_gene_b = -training_data_format.genes_mu[1] / (
                training_data_format.genes_sigma[1]
                + 1e-9  # Ultra high precision for MLP inference testing
            )
            expected_missing_gene_d = -training_data_format.genes_mu[3] / (
                training_data_format.genes_sigma[3]
                + 1e-9  # Ultra high precision for MLP inference testing
            )

            # All cells should have the same value for missing genes
            assert torch.allclose(
                X_processed[:, 1], torch.tensor(expected_missing_gene_b), atol=1e-5
            ), f"Missing GENE_B not handled correctly: {X_processed[:, 1]} != {expected_missing_gene_b}"
            assert torch.allclose(
                X_processed[:, 3], torch.tensor(expected_missing_gene_d), atol=1e-5
            ), f"Missing GENE_D not handled correctly: {X_processed[:, 3]} != {expected_missing_gene_d}"

    def test_inference_normalization_with_extra_genes(self, training_data_format):
        """Test that inference normalization correctly ignores extra genes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create inference data with extra genes F and G
            inference_adata = self.create_inference_adata("extra_genes")
            inference_file = Path(tmp_dir) / "inference_extra.h5ad"
            inference_adata.write_h5ad(inference_file)

            # Using CellsDataset
            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(10),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=inference_file,
            )

            # Should need gene transformation
            assert dataset.needs_gene_transformation

            # Gene mapping should ignore extra genes
            # Inference genes: [A, B, C, D, E, F, G] at indices [0, 1, 2, 3, 4, 5, 6]
            # Training genes: [A, B, C, D, E]
            # Expected mapping: [0, 1, 2, 3, 4] (perfect match for first 5, ignore F and G)
            expected_gene_indices = [0, 1, 2, 3, 4]
            assert np.array_equal(dataset.gene_indices, expected_gene_indices)

            # Get processed batch

            dataloader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )
            batch = next(iter(dataloader))
            X_processed = batch["x"]

            # Output should only have training genes (extra genes ignored)
            assert X_processed.shape[1] == training_data_format.n_genes

    def test_inference_normalization_with_reordered_genes(self, training_data_format):
        """Test that inference normalization correctly handles reordered genes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create inference data with reordered genes: E, C, A, D, B
            reordered_adata = self.create_inference_adata("reordered_genes")
            reordered_file = Path(tmp_dir) / "inference_reordered.h5ad"

            # Also create reference data with correct order for comparison
            reference_adata = self.create_inference_adata("perfect_match")
            reference_file = Path(tmp_dir) / "inference_reference.h5ad"

            # Use same raw data for both to test reordering preserves values
            np.random.seed(42)
            n_test_cells = 10
            raw_data = np.random.poisson(5, size=(n_test_cells, 5)).astype(np.float32)

            # Trim AnnData objects first
            reordered_adata = reordered_adata[:n_test_cells]
            reference_adata = reference_adata[:n_test_cells]

            # Set the same data in both AnnData objects but in different gene orders
            # Create a new AnnData object to avoid SparseEfficiencyWarning
            reference_adata_new = ad.AnnData(
                X=csr_matrix(raw_data), obs=reference_adata.obs, var=reference_adata.var
            )

            # For reordered: genes are E, C, A, D, B, so data should be reordered
            # raw_data columns: A, B, C, D, E (indices 0, 1, 2, 3, 4)
            # reordered genes: E, C, A, D, B (should use columns 4, 2, 0, 3, 1)
            reordered_data = raw_data[:, [4, 2, 0, 3, 1]]
            # Create a new AnnData object to avoid SparseEfficiencyWarning
            reordered_adata_new = ad.AnnData(
                X=csr_matrix(reordered_data),
                obs=reordered_adata.obs,
                var=reordered_adata.var,
            )

            # Save data
            reordered_adata_new.write_h5ad(reordered_file)
            reference_adata_new.write_h5ad(reference_file)

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

            batch_indices = [0, 1, 2, 3, 4]
            reference_batch = cells_collate_fn(batch_indices, reference_dataset)
            reordered_batch = cells_collate_fn(batch_indices, reordered_dataset)

            # Results should be identical after reordering
            assert torch.allclose(
                reference_batch["x"], reordered_batch["x"], atol=1e-5
            ), "Reordered genes should produce identical results to correctly ordered genes"

    def test_inference_normalization_with_mixed_scenario(self, training_data_format):
        """Test complex scenario with missing, extra, and reordered genes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create inference data with mixed scenario: F, C, A, G (missing B,D,E, extra F,G, reordered)
            mixed_adata = self.create_inference_adata("mixed_scenario")
            mixed_file = Path(tmp_dir) / "inference_mixed.h5ad"
            mixed_adata.write_h5ad(mixed_file)

            # Using CellsDataset
            dataset = CellsDataset(
                data_format=training_data_format,
                row_inds=np.arange(10),
                dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=mixed_file,
            )

            # Should need gene transformation
            assert dataset.needs_gene_transformation

            # Gene mapping analysis:
            # Inference genes: [F, C, A, G] at indices [0, 1, 2, 3]
            # Training genes: [A, B, C, D, E]
            # Expected mapping: [2, -1, 1, -1, -1] (A->2, B->missing, C->1, D->missing, E->missing)
            expected_gene_indices = [2, -1, 1, -1, -1]
            assert np.array_equal(dataset.gene_indices, expected_gene_indices)

            # Get processed batch

            dataloader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(
                    batch_indices, dataset
                ),
            )
            batch = next(iter(dataloader))
            X_processed = batch["x"]

            # Verify output shape is correct
            assert X_processed.shape[1] == training_data_format.n_genes

            # Verify missing genes have correct z-score values
            # Genes B (index 1), D (index 3), E (index 4) should be missing
            for missing_idx in [1, 3, 4]:
                expected_missing_value = -training_data_format.genes_mu[missing_idx] / (
                    training_data_format.genes_sigma[missing_idx]
                    + 1e-9  # Ultra high precision for MLP inference consistency
                )
                assert torch.allclose(
                    X_processed[:, missing_idx],
                    torch.tensor(expected_missing_value),
                    atol=1e-5,
                ), f"Missing gene at index {missing_idx} not handled correctly"

    def test_inference_pipeline_consistency(self, training_data_format):
        """Test that MLP inference pipeline produces consistent results regardless of gene order."""
        scenarios = ["perfect_match", "missing_genes", "extra_genes", "reordered_genes"]

        for scenario in scenarios:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create inference data
                inference_adata = self.create_inference_adata(scenario, n_cells=15)
                inference_file = Path(tmp_dir) / f"inference_{scenario}.h5ad"
                inference_adata.write_h5ad(inference_file)

                # Test using CellsDataset (what MLP inference actually uses)
                dataset = CellsDataset(
                    data_format=training_data_format,
                    row_inds=np.arange(10),
                    dataset_params=DataAugmentParams(mask_rate=0.0, noise_std=0.0),
                    is_train=False,
                    data_path=inference_file,
                )

                def make_collate_fn(ds):
                    return lambda batch_indices: cells_collate_fn(batch_indices, ds)

                dataloader = DataLoader(
                    dataset,
                    batch_size=5,
                    shuffle=False,
                    collate_fn=make_collate_fn(dataset),
                )

                for batch in dataloader:
                    X_processed = batch["x"]

                    # Basic sanity checks
                    assert (
                        X_processed.shape[1] == training_data_format.n_genes
                    ), f"Output shape wrong for {scenario}: {X_processed.shape[1]} != {training_data_format.n_genes}"
                    assert not torch.any(
                        torch.isnan(X_processed)
                    ), f"NaN values in {scenario}"
                    assert not torch.any(
                        torch.isinf(X_processed)
                    ), f"Inf values in {scenario}"

                    # Reasonable value range after z-score normalization
                    assert (
                        X_processed.abs().max() < 50
                    ), f"Extreme values in {scenario}: {X_processed.abs().max()}"

                    break  # Just test first batch

    def test_data_format_compatibility_for_shap_analysis(self, training_data_format):
        """Test that data_format from training can be used for post-training analysis."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Simulate what happens in run_mlp_train.py
            # Create training-like data
            training_adata = self.create_inference_adata("perfect_match", n_cells=100)
            training_file = Path(tmp_dir) / "training_data.h5ad"
            training_adata.write_h5ad(training_file)

            # Simulate getting data_format from training (like in run_mlp_train.py)
            # This would be equivalent to data_bundle.data_format
            data_format_from_training = training_data_format

            # Simulate post-training analysis (like getting data for SHAP)
            row_inds_train = np.arange(50)  # Some training indices

            # Load AnnData (like what happens in actual usage)
            adata = ad.read_h5ad(training_file, backed="r")

            # This is what the user wants to do for SHAP analysis
            X_train = load_and_preprocess_data_numpy(
                data_path=training_file,
                row_indices=row_inds_train,
                data_format=data_format_from_training,
            )

            # Verify this works correctly
            assert X_train.shape == (
                len(row_inds_train),
                data_format_from_training.n_genes,
            )
            assert X_train.dtype == np.float32
            assert not np.any(np.isnan(X_train))
            assert not np.any(np.isinf(X_train))

            # Verify normalization was applied correctly
            # After z-score normalization, data should have reasonable range
            assert np.abs(X_train).max() < 50, "Data should be normalized"

            adata.file.close()

            print("✅ data_format from training is valid for post-training analysis")
            print(f"   Shape: {X_train.shape}")
            print(f"   Range: [{X_train.min():.3f}, {X_train.max():.3f}]")
            print(f"   Mean: {X_train.mean():.3f}, Std: {X_train.std():.3f}")
