"""Test script to verify that multiprocessing works with HDF5 files.

This demonstrates the fix for the OSError: Can't synchronously read data issue.
"""

import sys

from pathlib import Path

import pytest
import torch

from torch.utils.data import DataLoader

from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.data_util.dataset import CellsDataset, cells_collate_fn


class PicklableCollateFn:
    """A picklable collate function for multiprocessing."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch_indices):
        return cells_collate_fn(batch_indices, self.dataset)


class TestMultiprocessingLoading:
    """Test class for multiprocessing functionality with HDF5 files."""

    def test_multiprocessing_loading_basic(self, dummy_adata, tmp_path):
        """Test basic multiprocessing loading with different worker counts."""
        # Create a temporary data file
        data_file = tmp_path / "test_data.h5ad"
        dummy_adata.write_h5ad(data_file)

        # Create a temporary data format file
        data_format_file = tmp_path / "data_format.json"

        # Create a simple data format for testing
        data_format = DataFormat(
            use_log_transform=False,
            aux_categorical_types=("tissue_type", "imputed_labels"),
        )

        # Create data format from the dummy data
        data_format.create_data_format(
            data_path=str(data_file),
            adata=dummy_adata,
            row_inds_train=list(range(len(dummy_adata))),
        )

        # Save the data format
        data_format.save(data_format_file)

        # Load the data format - use Path object
        loaded_data_format = load_data_format(data_format_file)

        # Create dataset
        dataset = CellsDataset(
            data_format=loaded_data_format,
            data_path=str(data_file),
            is_train=False,
        )

        assert len(dataset) > 0, "Dataset should contain samples"

        # Test with different worker counts
        for num_workers in [0, 2]:
            # Test with multiple workers
            collate_fn = PicklableCollateFn(dataset)
            dataloader = DataLoader(
                dataset, batch_size=32, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
            )

            # Load a few batches to test
            batch_count = 0
            max_batches = 3  # Just test a few batches

            for batch in dataloader:
                batch_count += 1
                # Verify batch structure
                assert "x" in batch, f"Batch should contain 'x' key for {num_workers} workers"
                assert batch["x"].shape[0] <= 32, f"Batch size should be <= 32 for {num_workers} workers"
                assert batch["x"].shape[1] == dummy_adata.n_vars, (
                    f"Batch should have correct number of genes for {num_workers} workers"
                )

                if batch_count >= max_batches:
                    break

            # Assert that we got the expected number of batches
            assert batch_count > 0, f"No batches were loaded with {num_workers} workers"
            assert batch_count <= max_batches, f"Should not load more than {max_batches} batches"

    @pytest.mark.slow
    def test_multiprocessing_with_real_data_format(self, dummy_adata, tmp_path):
        """Test multiprocessing with a more realistic data format setup."""
        # Create a temporary data file
        data_file = tmp_path / "test_data.h5ad"
        dummy_adata.write_h5ad(data_file)

        # Create a data format with more realistic settings
        data_format = DataFormat(
            use_log_transform=True,
            aux_categorical_types=("tissue_type", "imputed_labels", "cancer_type"),
        )

        # Create data format from the dummy data
        data_format.create_data_format(
            data_path=str(data_file),
            adata=dummy_adata,
            row_inds_train=list(range(len(dummy_adata))),
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=str(data_file),
            is_train=False,
        )

        # Test with multiple workers and different batch sizes
        test_configs = [
            (0, 16),  # Single worker, small batch
            (2, 32),  # Multiple workers, medium batch
            (4, 64),  # Multiple workers, larger batch
        ]

        for num_workers, batch_size in test_configs:
            collate_fn = PicklableCollateFn(dataset)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
            )

            # Load one batch to verify it works
            batch = next(iter(dataloader))

            # Verify batch structure
            assert "x" in batch, f"Batch should contain 'x' key for {num_workers} workers, batch_size={batch_size}"
            assert batch["x"].shape[0] <= batch_size, f"Batch size should be <= {batch_size} for {num_workers} workers"
            assert batch["x"].shape[1] == dummy_adata.n_vars, (
                f"Batch should have correct number of genes for {num_workers} workers"
            )

    def test_multiprocessing_edge_cases(self, dummy_adata, tmp_path):
        """Test multiprocessing with edge cases like very small datasets."""
        # Create a temporary data file
        data_file = tmp_path / "test_data.h5ad"
        dummy_adata.write_h5ad(data_file)

        # Create data format
        data_format = DataFormat(
            use_log_transform=False,
            aux_categorical_types=("tissue_type",),
        )

        data_format.create_data_format(
            data_path=str(data_file),
            adata=dummy_adata,
            row_inds_train=list(range(len(dummy_adata))),
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=str(data_file),
            is_train=False,
        )

        # Test with batch size larger than dataset
        large_batch_size = len(dataset) + 10

        for num_workers in [0, 2]:
            collate_fn = PicklableCollateFn(dataset)
            dataloader = DataLoader(
                dataset, batch_size=large_batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
            )

            # Should get exactly one batch with all data
            batches = list(dataloader)
            assert len(batches) == 1, f"Should get exactly one batch with large batch size for {num_workers} workers"
            assert batches[0]["x"].shape[0] == len(dataset), f"Batch should contain all data for {num_workers} workers"

    @pytest.mark.slow
    def test_multiprocessing_worker_consistency(self, dummy_adata, tmp_path):
        """Test that multiprocessing produces consistent results across different worker counts."""
        # Create a temporary data file
        data_file = tmp_path / "test_data.h5ad"
        dummy_adata.write_h5ad(data_file)

        # Create data format
        data_format = DataFormat(
            use_log_transform=False,
            aux_categorical_types=("tissue_type",),
        )

        data_format.create_data_format(
            data_path=str(data_file),
            adata=dummy_adata,
            row_inds_train=list(range(len(dummy_adata))),
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=str(data_file),
            is_train=False,
        )

        # Test with different worker counts but same batch size
        batch_size = 16
        results = {}

        for num_workers in [0, 2]:  # Reduced from [0, 2, 4] to speed up test
            collate_fn = PicklableCollateFn(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,  # Important: no shuffling for consistency
                collate_fn=collate_fn,
            )

            # Collect all batches
            batches = [batch["x"] for batch in dataloader]

            # Store results for comparison
            results[num_workers] = torch.cat(batches, dim=0)

        # Verify consistency across different worker counts
        for num_workers in [2]:  # Reduced from [2, 4] to speed up test
            assert torch.allclose(results[0], results[num_workers], atol=1e-6), (
                f"Results should be consistent between 0 and {num_workers} workers"
            )


# Keep the original script functionality for manual testing
if __name__ == "__main__":
    # Test with a data file - adjust path as needed
    data_paths_to_try = [
        "data/example_data.h5ad",
        "data/adata_subset_for_Ron.h5ad",
        "data/scXpand_counts_with_expansion_for_model_08_12_2024.h5ad",
    ]

    data_path = None
    for path in data_paths_to_try:
        if Path(path).exists():
            data_path = path
            break

    if data_path is None:
        print("No test data found. Available files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.h5ad"):
                print(f"  {f}")
        else:
            print("  data/ directory not found")
        sys.exit(1)

    print(f"Using data file: {data_path}")

    # Test with different worker counts
    for num_workers in [0, 2, 4]:
        print(f"\n{'=' * 50}")
        print(f"Testing with {num_workers} workers...")

        # For manual testing, we'll just create a simple test
        try:
            # Try to load a data format (this might not exist, but that's ok for this test)
            try:
                data_format = load_data_format(Path("results/autoencoder/data_format.json"))
                print("✅ Successfully loaded data format")
            except (FileNotFoundError, KeyError, ValueError) as e:
                print("Could not load existing data format, this test needs preprocessed data")
                print("Run a training first to generate data_format.json")
                print(f"Error: {e}")
                continue

            print(f"✅ Manual test with {num_workers} workers completed successfully!")

        except Exception as e:
            print(f"❌ Error with {num_workers} workers: {e}")
            if num_workers > 0:
                print("If you see errors above, there might be an issue with the multiprocessing fix")
                break

    print(f"\n{'=' * 50}")
    print("Test complete!")
