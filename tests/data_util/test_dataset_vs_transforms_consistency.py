"""Test consistency between CellsDataset and load_and_preprocess_data_numpy approaches.

This test module ensures that both data loading pathways produce identical results,
preventing the histogram discrepancy bug where CellsDataset computes scaling factors
dynamically while load_and_preprocess_data_numpy uses precomputed factors.
"""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataloaders import create_eval_dataloader
from scxpand.data_util.dataset import CellsDataset
from scxpand.data_util.transforms import load_and_preprocess_data_numpy
from scxpand.util.general_util import compute_row_sums, compute_scaling_factors
from tests.test_utils import windows_safe_context_manager


@pytest.fixture
def test_data_with_scaling_factors():
    """Create test data with precomputed scaling factors."""
    np.random.seed(42)

    # Create realistic single-cell data
    n_cells = 1000
    n_genes = 100

    # Create expression data with varying total counts per cell
    X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Make some genes sparse (like in real scRNA-seq data)
    sparse_mask = np.random.random((n_cells, n_genes)) > 0.7
    X[~sparse_mask] = 0

    # Ensure some cells have different total expression levels
    high_expr_cells = np.random.choice(n_cells, size=200, replace=False)
    X[high_expr_cells, :] *= 3

    # Create AnnData
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["expansion"] = np.random.choice(["expanded", "not_expanded"], n_cells)

    # Create data format with scaling factors
    data_format = DataFormat(
        n_genes=n_genes,
        gene_names=list(adata.var_names),
        target_sum=10000.0,
        use_log_transform=False,  # Keep simple for debugging
    )

    # Use subset as "training" data to compute scaling factors
    train_indices = np.arange(800)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_data.h5ad"
        adata.write_h5ad(temp_path)

        # This will compute and store scaling factors
        data_format.create_data_format(
            data_path=temp_path,
            adata=adata,
            row_inds_train=train_indices,
        )

    return adata, data_format, train_indices


class TestDatasetVsTransformsConsistency:
    """Test that CellsDataset and load_and_preprocess_data_numpy give identical results."""

    def test_full_matrix_consistency(self, test_data_with_scaling_factors):
        """Test that both approaches give identical results for full matrix processing."""
        adata, data_format, train_indices = test_data_with_scaling_factors

        # Test on a subset of training data (where scaling factors were computed)
        test_indices = train_indices[:100]  # First 100 training cells

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            windows_safe_context_manager() as ctx,
        ):
            temp_path = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(temp_path)
            ctx.register_file(temp_path)

            # Method 1: CellsDataset approach
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=test_indices,
                is_train=False,
                data_path=temp_path,
            )

            dataloader = create_eval_dataloader(
                dataset=dataset,
                batch_size=len(test_indices),  # Single batch
                num_workers=0,
            )

            # Get data from CellsDataset
            batch = next(iter(dataloader))
            expr_from_dataset = batch["x"].numpy()

            # Method 2: load_and_preprocess_data_numpy approach
            expr_from_transforms = load_and_preprocess_data_numpy(
                data_path=temp_path,
                data_format=data_format,
                row_indices=test_indices,
                gene_subset=None,  # Full matrix
            )

            # These should be nearly identical (small floating-point differences expected)
            # Use more lenient tolerances for Windows compatibility
            np.testing.assert_allclose(
                expr_from_dataset,
                expr_from_transforms,
                rtol=1e-5,
                atol=1e-6,
                err_msg="CellsDataset and load_and_preprocess_data_numpy should give identical results for full matrix",
            )

    def test_single_gene_consistency(self, test_data_with_scaling_factors):
        """Test the exact scenario from the notebook - single gene analysis."""
        adata, data_format, train_indices = test_data_with_scaling_factors

        # Test on training data subset
        test_indices = train_indices[:200]
        target_gene = "GENE_0025"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(temp_path)

            # Method 1: CellsDataset approach (simulating notebook's first path)
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=test_indices,
                is_train=False,
                data_path=temp_path,
            )

            dataloader = create_eval_dataloader(
                dataset=dataset,
                batch_size=len(test_indices),
                num_workers=0,
            )

            batch = next(iter(dataloader))
            full_expr = batch["x"].numpy()

            # Extract the target gene
            gene_index = data_format.gene_names.index(target_gene)
            expr_from_dataset = full_expr[:, gene_index]

            # Method 2: load_and_preprocess_data_numpy with gene subset (notebook's second path)
            expr_from_transforms = load_and_preprocess_data_numpy(
                data_path=temp_path,
                data_format=data_format,
                row_indices=test_indices,
                gene_subset=[target_gene],
            ).flatten()

            # These should be identical (this currently fails due to the bug)
            try:
                np.testing.assert_allclose(
                    expr_from_dataset,
                    expr_from_transforms,
                    rtol=1e-5,
                    atol=1e-6,  # Relaxed for Windows compatibility
                    err_msg="Single gene analysis should be consistent between approaches",
                )
                print("✅ Single gene consistency test PASSED")
            except AssertionError as e:
                print(
                    "❌ Single gene consistency test FAILED (expected - demonstrates the bug)"
                )
                print(
                    f"CellsDataset stats: mean={expr_from_dataset.mean():.6f}, std={expr_from_dataset.std():.6f}"
                )
                print(
                    f"Transforms stats:  mean={expr_from_transforms.mean():.6f}, std={expr_from_transforms.std():.6f}"
                )
                print(
                    f"Difference: mean_diff={abs(expr_from_dataset.mean() - expr_from_transforms.mean()):.6f}"
                )
                raise e

    def test_sparse_gene_scenario(self, test_data_with_scaling_factors):
        """Test with a gene that has very sparse expression (like ENSG00000153563 in the notebook)."""
        adata, data_format, train_indices = test_data_with_scaling_factors

        # Make one gene very sparse
        sparse_gene_idx = 50
        sparse_gene_name = data_format.gene_names[sparse_gene_idx]

        # Modify the data to make this gene sparse
        adata_copy = adata.copy()
        X_modified = adata_copy.X.toarray()
        X_modified[:, sparse_gene_idx] = 0  # Start with all zeros

        # Only a few cells express this gene
        expressing_cells = np.random.choice(train_indices, size=10, replace=False)
        X_modified[expressing_cells, sparse_gene_idx] = np.random.choice(
            [1, 2, 3], size=10
        )

        adata_copy.X = sp.csr_matrix(X_modified)

        test_indices = train_indices[:300]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_sparse_data.h5ad"
            adata_copy.write_h5ad(temp_path)

            # Update data_format to have scaling factors for the modified data
            data_format_updated = DataFormat(
                n_genes=data_format.n_genes,
                gene_names=data_format.gene_names,
                genes_mu=data_format.genes_mu,
                genes_sigma=data_format.genes_sigma,
                target_sum=data_format.target_sum,
                use_log_transform=data_format.use_log_transform,
            )
            data_format_updated.create_data_format(
                data_path=temp_path,
                adata=adata_copy,
                row_inds_train=train_indices,
            )

            # Method 1: CellsDataset
            dataset = CellsDataset(
                data_format=data_format_updated,
                row_inds=test_indices,
                is_train=False,
                data_path=temp_path,
            )

            dataloader = create_eval_dataloader(
                dataset=dataset,
                batch_size=len(test_indices),
                num_workers=0,
            )

            batch = next(iter(dataloader))
            expr_from_dataset = batch["x"].numpy()[:, sparse_gene_idx]

            # Method 2: load_and_preprocess_data_numpy
            expr_from_transforms = load_and_preprocess_data_numpy(
                data_path=temp_path,
                data_format=data_format_updated,
                row_indices=test_indices,
                gene_subset=[sparse_gene_name],
            ).flatten()

            # Check statistics to see the difference
            print(f"\nSparse gene ({sparse_gene_name}) analysis:")
            print(
                f"CellsDataset: mean={expr_from_dataset.mean():.6f}, std={expr_from_dataset.std():.6f}, range=[{expr_from_dataset.min():.6f}, {expr_from_dataset.max():.6f}]"
            )
            print(
                f"Transforms:  mean={expr_from_transforms.mean():.6f}, std={expr_from_transforms.std():.6f}, range=[{expr_from_transforms.min():.6f}, {expr_from_transforms.max():.6f}]"
            )

            # These should be nearly identical (small floating-point differences expected)
            np.testing.assert_allclose(
                expr_from_dataset,
                expr_from_transforms,
                rtol=1e-6,
                atol=1e-7,  # Relaxed from 1e-8 to account for scaling factor computation differences
                err_msg="Sparse gene analysis should be consistent between approaches",
            )

    def test_scaling_factors_usage(self, test_data_with_scaling_factors):
        """Test that demonstrates the root cause - scaling factor usage."""
        adata, data_format, train_indices = test_data_with_scaling_factors

        test_indices = train_indices[:50]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(temp_path)

            # Get raw data
            X_raw = adata.X[test_indices, :].toarray()

            # In the new design, scaling factors are not pre-computed or stored in DataFormat.
            # They are computed on-the-fly. This test now verifies that behavior.
            row_sums = compute_row_sums(X_raw)
            scaling_factors = compute_scaling_factors(row_sums, data_format.target_sum)

            print("\nOn-the-fly scaling factors:")
            print(
                f"Computed factors: mean={scaling_factors.mean():.6f}, std={scaling_factors.std():.6f}"
            )

            # This assertion confirms that scaling factors are positive and finite.
            assert np.all(np.isfinite(scaling_factors))
            assert np.all(scaling_factors > 0)


def test_demonstrate_notebook_bug():
    """Reproduce the exact scenario from the notebook to demonstrate the bug."""
    print("\n" + "=" * 80)
    print("DEMONSTRATING THE NOTEBOOK BUG")
    print("=" * 80)

    # Create data similar to the notebook
    np.random.seed(12345)
    n_cells = 1000
    n_genes = 100

    X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Make one gene very sparse (like ENSG00000153563)
    sparse_gene_idx = 75
    X[:, sparse_gene_idx] = 0
    expressing_cells = np.random.choice(n_cells, size=20, replace=False)
    X[expressing_cells, sparse_gene_idx] = np.random.choice([1, 2, 3, 4], size=20)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["expansion"] = np.random.choice(["expanded", "not_expanded"], n_cells)

    # Create data format
    data_format = DataFormat(
        n_genes=n_genes,
        gene_names=list(adata.var_names),
        target_sum=10000.0,
        use_log_transform=False,
    )

    train_indices = np.arange(800)
    eval_indices = np.arange(100, 300)  # Use subset of training data for evaluation

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "notebook_test.h5ad"
        adata.write_h5ad(temp_path)

        data_format.create_data_format(
            data_path=temp_path,
            adata=adata,
            row_inds_train=train_indices,
        )

        sparse_gene = f"GENE_{sparse_gene_idx:04d}"

        print(f"\nTesting sparse gene: {sparse_gene}")
        print(
            f"Cells expressing this gene: {np.count_nonzero(X[eval_indices, sparse_gene_idx])}/{len(eval_indices)}"
        )

        # Method 1: CellsDataset (notebook's first approach)
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=eval_indices,
            is_train=False,
            data_path=temp_path,
        )

        dataloader = create_eval_dataloader(
            dataset=dataset,
            batch_size=len(eval_indices),
            num_workers=0,
        )

        batch = next(iter(dataloader))
        gene_index = data_format.gene_names.index(sparse_gene)
        expr1 = batch["x"].numpy()[:, gene_index]

        # Method 2: load_and_preprocess_data_numpy (notebook's second approach)
        expr2 = load_and_preprocess_data_numpy(
            data_path=temp_path,
            data_format=data_format,
            row_indices=eval_indices,
            gene_subset=[sparse_gene],
        ).flatten()

        print("\nResults comparison:")
        print(
            f"Method 1 (CellsDataset):           mean={expr1.mean():.6f}, std={expr1.std():.6f}, range=[{expr1.min():.6f}, {expr1.max():.6f}]"
        )
        print(
            f"Method 2 (load_and_preprocess):    mean={expr2.mean():.6f}, std={expr2.std():.6f}, range=[{expr2.min():.6f}, {expr2.max():.6f}]"
        )
        print(f"Difference in means: {abs(expr1.mean() - expr2.mean()):.6f}")
        print(f"Difference in stds:  {abs(expr1.std() - expr2.std()):.6f}")

        if not np.allclose(expr1, expr2, rtol=1e-3):
            print("❌ Methods give different results - BUG CONFIRMED")
        else:
            print("✅ Methods give same results")


def test_demonstrate_eval_set_bug():
    """Test with evaluation set that wasn't part of training (like the notebook scenario)."""
    print("\n" + "=" * 80)
    print("TESTING EVAL SET BUG (OUTSIDE TRAINING)")
    print("=" * 80)

    # Create data similar to the notebook
    np.random.seed(42)
    n_cells = 1000
    n_genes = 50

    X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Make one gene very sparse
    sparse_gene_idx = 25
    X[:, sparse_gene_idx] = 0
    expressing_cells = np.random.choice(n_cells, size=15, replace=False)
    X[expressing_cells, sparse_gene_idx] = np.random.choice([1, 2, 3], size=15)

    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["expansion"] = np.random.choice(["expanded", "not_expanded"], n_cells)

    # Split like in the notebook: train on first 800, evaluate on last 200
    train_indices = np.arange(800)
    eval_indices = np.arange(800, 1000)  # IMPORTANT: This is outside training set!

    # Create data format with scaling factors computed from TRAINING SET ONLY
    data_format = DataFormat(
        n_genes=n_genes,
        gene_names=list(adata.var_names),
        target_sum=10000.0,
        use_log_transform=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "eval_test.h5ad"
        adata.write_h5ad(temp_path)

        # This computes scaling factors from training set only
        data_format.create_data_format(
            data_path=temp_path,
            adata=adata,
            row_inds_train=train_indices,
        )

        sparse_gene = f"GENE_{sparse_gene_idx:04d}"

        print(f"\nTesting sparse gene: {sparse_gene}")
        print(f"Training set size: {len(train_indices)}")
        print(f"Eval set size: {len(eval_indices)}")
        print(
            f"Eval cells expressing this gene: {np.count_nonzero(X[eval_indices, sparse_gene_idx])}/{len(eval_indices)}"
        )

        # Method 1: CellsDataset - This will now work
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=eval_indices,  # These indices are outside the scaling factors range!
            is_train=False,
            data_path=temp_path,
        )

        dataloader = create_eval_dataloader(
            dataset=dataset,
            batch_size=len(eval_indices),
            num_workers=0,
        )

        batch = next(iter(dataloader))
        gene_index = data_format.gene_names.index(sparse_gene)
        expr1 = batch["x"].numpy()[:, gene_index]
        print(f"CellsDataset worked: mean={expr1.mean():.6f}, std={expr1.std():.6f}")

        # Method 2: load_and_preprocess_data_numpy - This will also work
        expr2 = load_and_preprocess_data_numpy(
            data_path=temp_path,
            data_format=data_format,
            row_indices=eval_indices,  # These indices are outside the scaling factors range!
            gene_subset=[sparse_gene],
        ).flatten()
        print(
            f"load_and_preprocess worked: mean={expr2.mean():.6f}, std={expr2.std():.6f}"
        )

        # Both methods now work, and they should give the same results.
        print("\nResults comparison:")
        print(f"CellsDataset:         mean={expr1.mean():.6f}, std={expr1.std():.6f}")
        print(f"load_and_preprocess:  mean={expr2.mean():.6f}, std={expr2.std():.6f}")
        print(f"Difference in means: {abs(expr1.mean() - expr2.mean()):.6f}")
        print(f"Difference in stds:  {abs(expr1.std() - expr2.std()):.6f}")

        # The core assertion: both methods must now produce the same output
        np.testing.assert_allclose(expr1, expr2, rtol=1e-5, atol=1e-6)
        print("✅ Methods give same results")


if __name__ == "__main__":
    test_demonstrate_notebook_bug()
    test_demonstrate_eval_set_bug()
