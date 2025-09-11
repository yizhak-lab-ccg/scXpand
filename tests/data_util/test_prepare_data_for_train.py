import tempfile

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from anndata import AnnData
from scipy.sparse import csr_matrix

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.data_splitter import split_data
from scxpand.data_util.prepare_data_for_train import prepare_data_for_training
from tests.test_utils import windows_safe_context_manager


@pytest.fixture
def mock_adata_with_required_columns() -> AnnData:
    """Create mock AnnData with all required columns for the data splitter."""
    n_cells, n_genes = 1000, 10

    # Create expression matrix with float32 dtype as a csr_matrix
    X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))

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
            "study": [patients[i][0] for i in patient_assignments],
            "patient": [patients[i][1] for i in patient_assignments],
            "cancer_type": [patients[i][2] for i in patient_assignments],
            "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells),
            "imputed_labels": np.random.choice(["label1", "label2"], size=n_cells),
            "sample": np.random.choice(["sample1", "sample2", "sample3", "sample4"], size=n_cells),
        }
    )
    obs.index = obs.index.astype(str)

    return AnnData(X=X, obs=obs)


class TestPrepareDataForTraining:
    def test_prepare_data_with_adata_input(self, mock_adata_with_required_columns: AnnData) -> None:
        """Test prepare_data_for_training with adata input."""
        with tempfile.TemporaryDirectory() as tmp_dir, windows_safe_context_manager() as ctx:
            # Save adata to file since data_path is now mandatory
            data_file = Path(tmp_dir) / "test_data.h5ad"
            mock_adata_with_required_columns.write_h5ad(data_file)
            ctx.register_file(data_file)

            result = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=("tissue_type", "imputed_labels"),
                use_log_transform=True,
                save_dir=tmp_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=False,
            )

            # Register the result's adata for cleanup
            ctx.register_adata(result.adata)

            # Verify basic properties
            assert result.data_format.n_genes == 10
            assert len(result.data_format.gene_names) == 10
            assert result.data_format.use_log_transform is True

            # Verify categorical mappings were created
            assert "tissue_type" in result.data_format.aux_categorical_mappings
            assert "imputed_labels" in result.data_format.aux_categorical_mappings

            # Verify data splits
            assert len(result.row_inds_train) + len(result.row_inds_dev) == 1000
            # Patient-based splitting may not result in exact percentages
            train_ratio = len(result.row_inds_train) / 1000
            assert 0.6 <= train_ratio <= 0.9  # Should be roughly 80% but allow variation due to patient boundaries

            # Verify adata is processed
            # Note: backing mode now depends on whether preprocessing was applied

    def test_prepare_data_with_data_path_input(self, mock_adata_with_required_columns: AnnData) -> None:
        """Test prepare_data_for_training with data_path input (no adata)."""
        with tempfile.TemporaryDirectory() as tmp_dir, windows_safe_context_manager() as ctx:
            # Save the adata to a file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            mock_adata_with_required_columns.write_h5ad(data_file)
            ctx.register_file(data_file)

            result = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=("tissue_type", "imputed_labels"),
                use_log_transform=False,
                save_dir=tmp_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=False,
            )

            # Register the result's adata for cleanup
            ctx.register_adata(result.adata)

            # Verify basic properties
            assert result.data_format.n_genes == 10
            assert len(result.data_format.gene_names) == 10
            assert result.data_format.use_log_transform is False
            # Verify adata is backed (data loaded from disk)
            assert result.adata.isbacked

    def test_prepare_data_separated_methods_equivalence(self, mock_adata_with_required_columns: AnnData) -> None:
        """Test that using separated methods produces the same result as the integrated function."""
        with tempfile.TemporaryDirectory() as tmp_dir, windows_safe_context_manager() as ctx:
            tmp_path = Path(tmp_dir)
            # Save the adata to a file
            data_file = tmp_path / "test_data.h5ad"
            mock_adata_with_required_columns.write_h5ad(data_file)
            ctx.register_file(data_file)

            # Create subdirectories
            integrated_dir = tmp_path / "integrated"
            manual_dir = tmp_path / "manual"
            integrated_dir.mkdir(exist_ok=True)
            manual_dir.mkdir(exist_ok=True)

            # Method 1: Use the integrated function
            result_integrated = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=("tissue_type", "imputed_labels"),
                use_log_transform=True,
                save_dir=integrated_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=False,
            )
            ctx.register_adata(result_integrated.adata)

            # Method 2: Use separated methods manually
            adata_loaded = ad.read_h5ad(data_file, backed="r")
            ctx.register_adata(adata_loaded)

            row_inds_train, row_inds_dev = split_data(
                adata=adata_loaded,
                dev_ratio=0.2,
                random_seed=42,
                save_path=manual_dir,
            )

            data_format_manual = DataFormat(
                use_log_transform=True,
                aux_categorical_types=("tissue_type", "imputed_labels"),
            )

            data_format_manual.create_data_format(
                data_path=data_file,
                adata=adata_loaded,
                row_inds_train=row_inds_train,
            )
            adata_manual = data_format_manual.prepare_adata_for_training(adata_loaded, reorder_genes=False)
            ctx.register_adata(adata_manual)

            # Compare results
            assert result_integrated.data_format.n_genes == data_format_manual.n_genes
            assert result_integrated.data_format.gene_names == data_format_manual.gene_names
            assert np.allclose(result_integrated.data_format.genes_mu, data_format_manual.genes_mu)
            assert np.allclose(result_integrated.data_format.genes_sigma, data_format_manual.genes_sigma)
            assert result_integrated.data_format.aux_categorical_mappings == data_format_manual.aux_categorical_mappings

            # Compare data splits (should be identical with same random seed)
            assert np.array_equal(result_integrated.row_inds_train, row_inds_train)
            assert np.array_equal(result_integrated.row_inds_dev, row_inds_dev)

            # Compare processed adata objects
            assert result_integrated.adata.n_obs == adata_manual.n_obs
            assert result_integrated.adata.n_vars == adata_manual.n_vars
            assert result_integrated.adata.var_names.tolist() == adata_manual.var_names.tolist()

            # Compare processed expression matrices
            # Handle different matrix types: sparse matrices, HDF5 datasets, CSRDatasets, dense arrays
            def convert_to_dense_array(X):
                if hasattr(X, "toarray"):  # Sparse matrix
                    return X.toarray()
                elif str(type(X)).endswith("_CSRDataset'>"):  # AnnData CSRDataset
                    matrix = X[:]  # This returns a sparse matrix
                    return matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
                elif hasattr(X, "shape") and hasattr(X, "__getitem__"):  # HDF5 dataset or array-like
                    return np.array(X[:])
                else:
                    return np.array(X)

            X_integrated = convert_to_dense_array(result_integrated.adata.X)
            X_manual = convert_to_dense_array(adata_manual.X)

            assert np.allclose(X_integrated, X_manual, rtol=1e-6, atol=1e-6)

    def test_prepare_data_resume_functionality(self, mock_adata_with_required_columns: AnnData) -> None:
        """Test the resume functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir, windows_safe_context_manager() as ctx:
            # Save the adata to a file
            data_file = Path(tmp_dir) / "test_data.h5ad"
            mock_adata_with_required_columns.write_h5ad(data_file)
            ctx.register_file(data_file)

            # First run: create data format and splits
            result1 = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=("tissue_type", "imputed_labels"),
                use_log_transform=True,
                save_dir=tmp_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=False,
            )
            ctx.register_adata(result1.adata)

            # Second run: resume from saved data
            result2 = prepare_data_for_training(
                data_path=data_file,
                aux_categorical_types=("tissue_type", "imputed_labels"),
                use_log_transform=True,
                save_dir=tmp_dir,
                dev_ratio=0.2,
                rand_seed=42,
                resume=True,
            )
            ctx.register_adata(result2.adata)

            # Verify that resumed data matches original
            assert result1.data_format.n_genes == result2.data_format.n_genes
            assert result1.data_format.gene_names == result2.data_format.gene_names
            assert np.array_equal(result1.row_inds_train, result2.row_inds_train)
            assert np.array_equal(result1.row_inds_dev, result2.row_inds_dev)

    def test_error_when_data_path_not_exists(self) -> None:
        """Test that appropriate error is raised when data_path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_file = Path(tmp_dir) / "non_existent.h5ad"
            with pytest.raises(FileNotFoundError):
                prepare_data_for_training(
                    data_path=non_existent_file,
                    aux_categorical_types=(),
                    use_log_transform=False,
                    save_dir=tmp_dir,
                    dev_ratio=0.2,
                    rand_seed=42,
                    resume=False,
                )
