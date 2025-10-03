import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import (
    CellsDataset,
    cells_collate_fn,
    compute_categorical_targets_from_batch_obs,
)
from scxpand.mlp.mlp_params import DataAugmentParams


class TestCellsDatasetDataFormatIntegration:
    @pytest.fixture(scope="class")
    def minimal_adata(self) -> ad.AnnData:
        # Create a minimal AnnData object with 3 cells and 2 genes
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        # IMPORTANT: Expansion values are explicitly set as [0, 1, 0]
        # First cell (index 0): not expanded (0)
        # Second cell (index 1): expanded (1)
        # Third cell (index 2): not expanded (0)
        obs = pd.DataFrame(
            {
                "expansion": [
                    "not_expanded",
                    "expanded",
                    "not_expanded",
                ],  # String values that will be converted
                "clone_id_size": [10, 20, 30],
                "median_clone_size": [10, 10, 10],
                "tissue_type": ["A", "B", "A"],
                "imputed_labels": ["X", "Y", "X"],
            }
        )
        var = pd.DataFrame(index=["gene1", "gene2"])
        return ad.AnnData(X=X, obs=obs, var=var)

    @pytest.fixture(scope="class")
    def data_format(self, minimal_adata) -> DataFormat:
        # Create a DataFormat instance with categorical features
        data_format = DataFormat(
            n_genes=2,
            aux_categorical_types=["tissue_type", "imputed_labels"],
            use_log_transform=False,
            target_sum=1.0,
            eps=1e-6,
            on_disk_mode=False,
        )
        # Simulate fitting on the minimal adata
        data_format.genes_mu = np.mean(minimal_adata.X, axis=0)
        data_format.genes_sigma = np.std(minimal_adata.X, axis=0) + 1e-6
        data_format.gene_names = ["gene1", "gene2"]
        data_format.aux_categorical_mappings = {
            "tissue_type": {"A": 0, "B": 1},
            "imputed_labels": {"X": 0, "Y": 1},
        }
        return data_format

    @pytest.fixture(scope="class")
    def dataset_params(self) -> DataAugmentParams:
        # Minimal DatasetParams stub
        class DummyParam:
            soft_loss_beta = 1.0
            mask_rate = 0.0
            noise_std = 0.0

        return DummyParam()

    def test_cellsdataset_and_dataformat_integration(
        self, minimal_adata, data_format, dataset_params, tmp_path
    ):
        # Save the data to a temporary file
        temp_file = tmp_path / "temp_adata.h5ad"
        minimal_adata.write_h5ad(temp_file)

        # Create CellsDataset
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=None,
            dataset_params=dataset_params,
            is_train=True,
            data_path=temp_file,
            include_row_normalized_gene_counts=True,
        )
        # Check dataset length
        assert len(dataset) == 3
        # Check gene count
        assert dataset.n_genes == 2
        # Check y shape
        assert dataset.y.shape[0] == 3

        # Check y values are correctly converted from string values to binary
        # "not_expanded" -> 0, "expanded" -> 1
        assert dataset.y[0].item() == 0  # First cell is "not_expanded"
        assert dataset.y[1].item() == 1  # Second cell is "expanded"
        assert dataset.y[2].item() == 0  # Third cell is "not_expanded"

        # Check soft labels
        assert dataset.y_soft is not None
        assert dataset.y_soft.shape[0] == 3
        # Check obs_df columns
        for col in [
            "expansion",
            "clone_id_size",
            "median_clone_size",
            "tissue_type",
            "imputed_labels",
        ]:
            assert col in dataset.obs_df.columns

    def test_batch_collation(
        self, minimal_adata, data_format, dataset_params, tmp_path
    ):
        """Test that batch collation works correctly with cells_collate_fn."""
        # Save the data to a temporary file
        temp_file = tmp_path / "temp_adata.h5ad"
        minimal_adata.write_h5ad(temp_file)

        dataset = CellsDataset(
            data_format=data_format,
            row_inds=None,
            dataset_params=dataset_params,
            is_train=True,
            data_path=temp_file,
            include_row_normalized_gene_counts=True,
        )

        # Test with a simple batch (all indices)
        batch_indices = [0, 1, 2]
        batch = cells_collate_fn(batch_indices, dataset)

        # Verify batch structure
        assert isinstance(batch, dict)
        assert "x" in batch
        assert "y" in batch
        assert "y_soft" in batch
        assert "x_row_normalized_gene_counts" in batch
        assert "categorical_targets" in batch

        # Verify batch shapes - checking each dimension separately for better error messages
        assert batch["x"].shape[0] == 3, (
            f"Expected 3 samples, got {batch['x'].shape[0]}"
        )
        # Feature dimension should be just the genes (2)
        assert batch["x"].shape[1] == 2, (
            f"Feature dimension {batch['x'].shape[1]} should be 2 (genes only)"
        )

        assert batch["y"].shape == (3,)
        assert batch["y_soft"].shape == (3,)
        assert batch["x_row_normalized_gene_counts"].shape == (
            3,
            2,
        )  # 3 samples, 2 genes

        # Check categorical targets are now a dictionary
        assert isinstance(batch["categorical_targets"], dict)
        assert "tissue_type" in batch["categorical_targets"]
        assert "imputed_labels" in batch["categorical_targets"]
        assert batch["categorical_targets"]["tissue_type"].shape == (3,)  # 3 samples
        assert batch["categorical_targets"]["imputed_labels"].shape == (3,)  # 3 samples

        # Check categorical targets are correct
        # For tissue_type: A=0, B=1 and for imputed_labels: X=0, Y=1
        # Sample 0: A,X -> tissue_type=0, imputed_labels=0
        # Sample 1: B,Y -> tissue_type=1, imputed_labels=1
        # Sample 2: A,X -> tissue_type=0, imputed_labels=0
        expected_tissue_type = torch.tensor([0, 1, 0])
        expected_imputed_labels = torch.tensor([0, 1, 0])
        assert torch.all(
            batch["categorical_targets"]["tissue_type"] == expected_tissue_type
        )
        assert torch.all(
            batch["categorical_targets"]["imputed_labels"] == expected_imputed_labels
        )

    def test_batch_loading_mode(self, minimal_adata, dataset_params, tmp_path):
        """Test dataset with batch loading (default behavior)."""
        # Save the data to a temporary file
        temp_file = tmp_path / "temp_adata.h5ad"
        minimal_adata.write_h5ad(temp_file)

        # Create data format for batch loading
        data_format_batch = DataFormat(
            n_genes=2,
            aux_categorical_types=["tissue_type", "imputed_labels"],
            use_log_transform=True,
            target_sum=1.0,
            eps=1e-6,
        )
        data_format_batch.genes_mu = np.mean(minimal_adata.X, axis=0)
        data_format_batch.genes_sigma = np.std(minimal_adata.X, axis=0) + 1e-6
        data_format_batch.gene_names = ["gene1", "gene2"]
        data_format_batch.aux_categorical_mappings = {
            "tissue_type": {"A": 0, "B": 1},
            "imputed_labels": {"X": 0, "Y": 1},
        }

        # Create dataset with batch loading
        dataset_batch = CellsDataset(
            data_format=data_format_batch,
            row_inds=None,
            dataset_params=dataset_params,
            is_train=True,
            data_path=temp_file,
            include_row_normalized_gene_counts=True,
        )

        # Create batch using the batch loading dataset
        batch_indices = [0, 1, 2]
        batch = cells_collate_fn(batch_indices, dataset_batch)

        # Verify basic structure
        assert isinstance(batch, dict)
        assert "x" in batch
        assert "y" in batch

        # The data should be processed on-the-fly, so verify shape
        # Check only sample dimension (first dimension)
        assert batch["x"].shape[0] == 3, (
            f"Expected 3 samples, got {batch['x'].shape[0]}"
        )
        # Feature dimension should be just the genes (2)
        assert batch["x"].shape[1] == 2, (
            f"Feature dimension {batch['x'].shape[1]} should be 2 (genes only)"
        )

        # Verify that categorical targets are present
        assert "categorical_targets" in batch
        assert "tissue_type" in batch["categorical_targets"]
        assert "imputed_labels" in batch["categorical_targets"]

    def test_specific_row_indices(
        self, minimal_adata, data_format, dataset_params, tmp_path
    ):
        """Test dataset with specific row indices."""
        # Print the original expansion values to understand what we're working with
        print(
            "\nOriginal data expansion values:", minimal_adata.obs["expansion"].values
        )

        # Use only first two rows
        row_inds = np.array([0, 1])

        # Save the data to a temporary file
        temp_file = tmp_path / "temp_adata.h5ad"
        minimal_adata.write_h5ad(temp_file)

        # Create dataset with specific row indices
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=row_inds,
            dataset_params=dataset_params,
            is_train=True,
            data_path=temp_file,
        )

        # Should have length 2, not 3
        assert len(dataset) == 2

        # Print dataset.y - this should show us how the string values are converted to binary
        print("Dataset.y values:", dataset.y)

        # The CellsDataset will convert the expansion values:
        # "not_expanded" -> 0
        # "expanded" -> 1
        # So for row_inds=[0, 1], we expect dataset.y to be [0, 1]

        # Verify conversion happened correctly
        assert dataset.y[0].item() == 0  # "not_expanded"
        assert dataset.y[1].item() == 1  # "expanded"

        # Get batch for the 2 indices
        batch_indices = [0, 1]
        batch = cells_collate_fn(batch_indices, dataset)

        # Print the batch values to debug
        print("Batch y values:", batch["y"])

        # Now test the batch values
        # batch["y"][0] should correspond to the value at row_inds[0] which is 0
        assert batch["y"][0].item() == 0

        # batch["y"][1] should correspond to the value at row_inds[1] which is 1
        assert batch["y"][1].item() == 1

    def test_data_augmentation(self, minimal_adata, data_format, tmp_path):
        """Test data augmentation during training."""

        # Create dataset params with augmentation
        class AugmentedParams:
            soft_loss_beta = 1.0
            mask_rate = 0.5  # 50% masking rate
            noise_std = 0.1  # Add noise with std=0.1

        # Save the data to a temporary file
        temp_file = tmp_path / "temp_adata.h5ad"
        minimal_adata.write_h5ad(temp_file)

        # Create dataset with augmentation enabled
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=None,
            dataset_params=AugmentedParams(),
            is_train=True,  # Must be True for augmentation
            data_path=temp_file,
        )

        # Create two batches with the same indices but different random seeds
        batch_indices = [0, 1, 2]

        # Set different seeds to get different augmentations
        torch.manual_seed(42)
        batch1 = cells_collate_fn(batch_indices, dataset)

        torch.manual_seed(43)
        batch2 = cells_collate_fn(batch_indices, dataset)

        # Batches should be different due to random augmentation
        assert not torch.allclose(batch1["x"], batch2["x"])

        # Create batch with no augmentation for comparison
        no_aug_dataset = CellsDataset(
            data_format=data_format,
            row_inds=None,
            dataset_params=AugmentedParams(),
            is_train=False,  # Turn off augmentation by setting is_train=False
            data_path=temp_file,
        )

        torch.manual_seed(42)
        no_aug_batch = cells_collate_fn(batch_indices, no_aug_dataset)

        # No augmentation batch should be different from augmented batch
        assert not torch.allclose(batch1["x"], no_aug_batch["x"])

    def test_categorical_targets_extraction(
        self, minimal_adata, data_format, dataset_params, tmp_path
    ):
        """Test that categorical targets are correctly extracted."""
        # Save the data to a temporary file
        temp_file = tmp_path / "temp_adata.h5ad"
        minimal_adata.write_h5ad(temp_file)

        # Create dataset with categorical targets
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=None,
            dataset_params=dataset_params,
            is_train=True,
            data_path=temp_file,
        )

        # Create batch observation data for testing
        batch_obs = {
            "tissue_type": np.array(["A", "B", "A"]),
            "imputed_labels": np.array(["X", "Y", "X"]),
        }

        # Compute targets directly
        targets = compute_categorical_targets_from_batch_obs(
            dataset=dataset,
            batch_obs=batch_obs,
        )

        # Expected targets based on the mappings:
        # tissue_type: A=0, B=1
        # imputed_labels: X=0, Y=1
        # Now targets is a dictionary
        assert isinstance(targets, dict)
        assert "tissue_type" in targets
        assert "imputed_labels" in targets

        expected_tissue_type = torch.tensor([0, 1, 0])  # A, B, A
        expected_imputed_labels = torch.tensor([0, 1, 0])  # X, Y, X

        assert torch.all(targets["tissue_type"] == expected_tissue_type)
        assert torch.all(targets["imputed_labels"] == expected_imputed_labels)

        # Test with unknown values
        batch_obs_unknown = {
            "tissue_type": np.array(["A", "Unknown", "A"]),
            "imputed_labels": np.array(["X", "Y", "Unknown"]),
        }

        targets_unknown = compute_categorical_targets_from_batch_obs(
            dataset=dataset,
            batch_obs=batch_obs_unknown,
        )

        # For unknown values, the index should remain 0
        expected_tissue_type_unknown = torch.tensor(
            [0, 0, 0]
        )  # A, Unknown (defaults to 0), A
        expected_imputed_labels_unknown = torch.tensor(
            [0, 1, 0]
        )  # X, Y, Unknown (defaults to 0)

        assert torch.all(targets_unknown["tissue_type"] == expected_tissue_type_unknown)
        assert torch.all(
            targets_unknown["imputed_labels"] == expected_imputed_labels_unknown
        )
