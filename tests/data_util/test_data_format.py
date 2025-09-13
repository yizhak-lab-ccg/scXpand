import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anndata import AnnData
from scipy.sparse import csr_matrix

from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.data_util.transforms import DEFAULT_EPS, apply_zscore_normalization


class TestDataFormatGeneConversion:
    @pytest.fixture
    def data_format(self) -> DataFormat:
        """Create a simple DataFormat with predefined gene names."""
        data_format = DataFormat(
            gene_names=["gene1", "gene2", "gene3"],
            n_genes=3,
        )
        # Set dummy mu and sigma values
        data_format.genes_mu = np.array([0.1, 0.2, 0.3])
        data_format.genes_sigma = np.array([1.0, 1.0, 1.0])
        return data_format

    def test_exact_match(self, data_format: DataFormat):
        """Test when AnnData has exactly the same genes in the same order."""
        # Create test data with exact gene match
        X = csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
        adata = AnnData(X=X, var=var)

        # Convert
        result = data_format.convert_genes_expression_matrix(adata)

        # Verify genes are unchanged
        assert result.var_names.tolist() == data_format.gene_names
        # Verify data is unchanged
        np.testing.assert_array_equal(result.X.toarray(), X.toarray())

    def test_reordering_needed(self, data_format: DataFormat):
        """Test when AnnData has the same genes but in different order."""
        # Create test data with genes in different order
        X = csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene3", "gene1", "gene2"])
        adata = AnnData(X=X, var=var)

        # Convert
        result = data_format.convert_genes_expression_matrix(adata)

        # Verify genes are now in correct order
        assert result.var_names.tolist() == data_format.gene_names

        # The reordered data should be:
        # X = [[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]] -> reordered to gene1, gene2, gene3
        expected_data = np.array([[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_missing_genes(self, data_format: DataFormat):
        """Test when AnnData is missing some genes from data_format."""
        # Create test data with missing gene2
        X = csr_matrix(np.array([[1.0, 3.0], [4.0, 6.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene3"])
        adata = AnnData(X=X, var=var)

        # Convert
        result = data_format.convert_genes_expression_matrix(adata)

        # Verify genes are now in correct order with all genes
        assert result.var_names.tolist() == data_format.gene_names

        # The expected data should have zeros for gene2:
        # Original: [[1.0, 3.0], [4.0, 6.0]] for [gene1, gene3]
        # Expected: [[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]] for [gene1, gene2, gene3]
        expected_data = np.array([[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_superfluous_genes(self, data_format: DataFormat):
        """Test when AnnData has extra genes not in data_format."""
        # Create test data with extra gene4 and gene5
        X = csr_matrix(
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
            dtype=np.float32,
        )
        var = pd.DataFrame(index=["gene1", "gene2", "gene3", "gene4", "gene5"])
        adata = AnnData(X=X, var=var)

        # Convert
        result = data_format.convert_genes_expression_matrix(adata)

        # Verify only the genes in data_format are present and in correct order
        assert result.var_names.tolist() == data_format.gene_names

        # Expect only the first 3 columns to be kept
        expected_data = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_missing_and_superfluous_genes(self, data_format: DataFormat):
        """Test when AnnData is missing some genes and has extra genes."""
        # Create test data with missing gene2 and extra gene4
        X = csr_matrix(np.array([[1.0, 3.0, 4.0], [5.0, 6.0, 7.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene3", "gene4"])
        adata = AnnData(X=X, var=var)

        # Convert
        result = data_format.convert_genes_expression_matrix(adata)

        # Verify genes are now in correct order with all required genes
        assert result.var_names.tolist() == data_format.gene_names

        # The expected data should have zeros for gene2 and remove gene4:
        # Original: [[1.0, 3.0, 4.0], [5.0, 6.0, 7.0]] for [gene1, gene3, gene4]
        # Expected: [[1.0, 0.0, 3.0], [5.0, 0.0, 6.0]] for [gene1, gene2, gene3]
        expected_data = np.array([[1.0, 0.0, 3.0], [5.0, 0.0, 6.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_completely_different_genes(self, data_format: DataFormat):
        """Test when AnnData has completely different genes."""
        # Create test data with completely different genes
        X = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene4", "gene5"])
        adata = AnnData(X=X, var=var)

        # Convert
        result = data_format.convert_genes_expression_matrix(adata)

        # Verify genes are now in correct order with all required genes
        assert result.var_names.tolist() == data_format.gene_names

        # The expected data should have zeros for all genes in data_format
        expected_data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)


class TestZScoreNormalizationDense:
    """Test z-score normalization for dense arrays."""

    def test_dense_normalization_in_place(self):
        """Test that dense z-score normalization works in-place."""
        # Create test data
        X = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        X_original = X.copy()

        mu = np.array([1.5, 3.5, 5.5], dtype=np.float32)
        sigma = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Apply normalization
        result = apply_zscore_normalization(X, mu, sigma, in_place=True)

        # Check that normalization was applied correctly and result is returned
        expected = (X_original - mu) / (sigma + DEFAULT_EPS)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        np.testing.assert_array_almost_equal(X, expected, decimal=5)  # Also check in-place modification

    def test_dense_with_different_mu_sigma(self):
        """Test dense normalization with different mu and sigma values."""
        X = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        X_original = X.copy()

        mu = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        sigma = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        result = apply_zscore_normalization(X, mu, sigma, in_place=True)

        expected = (X_original - mu) / (sigma + DEFAULT_EPS)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        np.testing.assert_array_almost_equal(X, expected, decimal=5)


class TestDataFormatRefactoredMethods:
    """Test the new refactored methods in DataFormat."""

    @pytest.fixture
    def data_format(self) -> DataFormat:
        """Create a DataFormat for testing refactored methods."""
        data_format = DataFormat(
            gene_names=["gene1", "gene2", "gene3", "gene4"],
            n_genes=4,
            target_sum=1000.0,
        )
        data_format.genes_mu = np.array([0.1, 0.2, 0.3, 0.4])
        data_format.genes_sigma = np.array([1.0, 1.1, 1.2, 1.3])
        return data_format

    def test_analyze_gene_differences_exact_match(self, data_format: DataFormat):
        """Test _analyze_gene_differences when genes match exactly."""
        X = csr_matrix(np.array([[1.0, 2.0, 3.0, 4.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene2", "gene3", "gene4"])
        adata = AnnData(X=X, var=var)

        curr_genes, missing, superfluous = data_format._analyze_gene_differences(adata)

        assert curr_genes == ["gene1", "gene2", "gene3", "gene4"]
        assert len(missing) == 0
        assert len(superfluous) == 0

    def test_analyze_gene_differences_missing_genes(self, data_format: DataFormat):
        """Test _analyze_gene_differences when genes are missing."""
        X = csr_matrix(np.array([[1.0, 2.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene3"])
        adata = AnnData(X=X, var=var)

        curr_genes, missing, superfluous = data_format._analyze_gene_differences(adata)

        assert curr_genes == ["gene1", "gene3"]
        assert missing == {"gene2", "gene4"}
        assert len(superfluous) == 0

    def test_analyze_gene_differences_superfluous_genes(self, data_format: DataFormat):
        """Test _analyze_gene_differences when there are extra genes."""
        X = csr_matrix(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene2", "gene3", "gene4", "gene5"])
        adata = AnnData(X=X, var=var)

        curr_genes, missing, superfluous = data_format._analyze_gene_differences(adata)

        assert curr_genes == ["gene1", "gene2", "gene3", "gene4", "gene5"]
        assert len(missing) == 0
        assert superfluous == {"gene5"}

    def test_reorder_genes_only(self, data_format: DataFormat):
        """Test _reorder_genes_only method."""
        X = csr_matrix(np.array([[4.0, 3.0, 2.0, 1.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene4", "gene3", "gene2", "gene1"])
        adata = AnnData(X=X, var=var)

        curr_genes = ["gene4", "gene3", "gene2", "gene1"]
        result = data_format._reorder_genes_only(adata, curr_genes)

        assert result.var_names.tolist() == data_format.gene_names
        # Should reorder from [4,3,2,1] to [1,2,3,4]
        expected_data = np.array([[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_add_missing_genes(self, data_format: DataFormat):
        """Test _add_missing_genes method."""
        X = csr_matrix(np.array([[1.0, 3.0], [5.0, 7.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene3"])
        obs = pd.DataFrame(index=[0, 1])
        common_adata = AnnData(X=X, var=var, obs=obs)

        missing_genes = {"gene2", "gene4"}
        n_obs = 2

        result = data_format._add_missing_genes(common_adata, missing_genes, n_obs)

        # Should have 4 genes total (2 original + 2 missing)
        assert result.n_vars == 4
        assert result.n_obs == 2
        # Check that missing genes were added with zeros
        result_dense = result.X.toarray()
        assert result_dense.shape == (2, 4)

    def test_reorder_final_genes(self, data_format: DataFormat):
        """Test _reorder_final_genes method."""
        X = csr_matrix(np.array([[3.0, 1.0, 4.0, 2.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene3", "gene1", "gene4", "gene2"])
        obs = pd.DataFrame(index=[0])
        adata = AnnData(X=X, var=var, obs=obs)

        curr_genes = ["gene3", "gene1", "gene4", "gene2"]
        missing_genes = set()
        superfluous_genes = set()

        result = data_format._reorder_final_genes(adata, curr_genes, missing_genes, superfluous_genes)

        assert result.var_names.tolist() == data_format.gene_names
        # Should reorder from [3,1,4,2] to [1,2,3,4] based on gene order
        expected_data = np.array([[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_reorder_final_genes_missing_from_mapping_error(self, data_format: DataFormat):
        """Test _reorder_final_genes raises error when genes are missing from mapping."""
        X = csr_matrix(np.array([[1.0, 2.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene_unknown"])
        obs = pd.DataFrame(index=[0])
        adata = AnnData(X=X, var=var, obs=obs)

        curr_genes = ["gene1", "gene_unknown"]
        missing_genes = set()
        superfluous_genes = set()

        with pytest.raises(ValueError, match="Genes missing from mapping"):
            data_format._reorder_final_genes(adata, curr_genes, missing_genes, superfluous_genes)

    def test_handle_gene_differences_integration(self, data_format: DataFormat):
        """Test _handle_gene_differences method with missing and superfluous genes."""
        X = csr_matrix(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene1", "gene3", "gene_extra"])
        obs = pd.DataFrame(index=[0, 1])
        adata = AnnData(X=X, var=var, obs=obs)

        curr_genes = ["gene1", "gene3", "gene_extra"]
        missing_genes = {"gene2", "gene4"}
        superfluous_genes = {"gene_extra"}

        result = data_format._handle_gene_differences(adata, curr_genes, missing_genes, superfluous_genes)

        assert result.var_names.tolist() == data_format.gene_names
        assert result.n_vars == 4
        assert result.n_obs == 2

        # Check that the result has correct structure
        result_dense = result.X.toarray()
        assert result_dense.shape == (2, 4)


class TestDataFormatSaveLoad:
    """Test DataFormat save and load functionality."""

    def test_save_and_load_data_format(self):
        """Test saving and loading DataFormat objects."""
        # Create a DataFormat with all features
        original_df = DataFormat(
            gene_names=["gene1", "gene2", "gene3"],
            n_genes=3,
            target_sum=10000.0,
            use_log_transform=True,
            use_zscore_norm=True,
            aux_categorical_types=("tissue_type", "cell_type"),
            aux_categorical_mappings={
                "tissue_type": {"brain": 0, "liver": 1},
                "cell_type": {"neuron": 0, "hepatocyte": 1},
            },
        )
        original_df.genes_mu = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        original_df.genes_sigma = np.array([1.0, 1.1, 1.2], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "data_format.json"

            # Save the DataFormat
            original_df.save(save_path)

            # Load the DataFormat
            loaded_df = load_data_format(save_path)

            # Verify all attributes are correctly preserved
            assert loaded_df.gene_names == original_df.gene_names
            assert loaded_df.n_genes == original_df.n_genes
            assert loaded_df.target_sum == original_df.target_sum
            assert loaded_df.use_log_transform == original_df.use_log_transform
            assert loaded_df.use_zscore_norm == original_df.use_zscore_norm
            assert loaded_df.aux_categorical_types == original_df.aux_categorical_types
            assert loaded_df.aux_categorical_mappings == original_df.aux_categorical_mappings

            # Verify numpy arrays are correctly preserved
            np.testing.assert_array_equal(loaded_df.genes_mu, original_df.genes_mu)
            np.testing.assert_array_equal(loaded_df.genes_sigma, original_df.genes_sigma)

    def test_create_data_format_with_categorical_mappings(self):
        """Test create_data_format method with categorical features."""
        # Create test AnnData with categorical features
        X = csr_matrix(np.random.rand(100, 50), dtype=np.float32)
        obs = pd.DataFrame(
            {
                "tissue_type": ["brain"] * 30 + ["liver"] * 40 + ["kidney"] * 30,
                "cell_type": ["neuron"] * 50 + ["hepatocyte"] * 50,
            }
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
        adata = AnnData(X=X, obs=obs, var=var)

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(data_path)

            # Create DataFormat with categorical types
            data_format = DataFormat(
                use_log_transform=True,
                use_zscore_norm=True,
                aux_categorical_types=("tissue_type", "cell_type"),
                target_sum=10000.0,
            )

            train_indices = np.arange(80)  # Use first 80 cells for training

            # This should create the mappings and compute statistics
            data_format.create_data_format(
                data_path=data_path,
                adata=adata,
                row_inds_train=train_indices,
            )

            # Verify basic properties
            assert data_format.n_genes == 50
            assert len(data_format.gene_names) == 50
            assert len(data_format.genes_mu) == 50
            assert len(data_format.genes_sigma) == 50

            # Verify categorical mappings were created
            assert "tissue_type" in data_format.aux_categorical_mappings
            assert "cell_type" in data_format.aux_categorical_mappings

            tissue_mapping = data_format.aux_categorical_mappings["tissue_type"]
            cell_mapping = data_format.aux_categorical_mappings["cell_type"]

            # Check that all unique values from training data are in mappings
            train_obs = adata.obs.iloc[train_indices]
            expected_tissues = set(train_obs["tissue_type"].unique())
            expected_cells = set(train_obs["cell_type"].unique())

            assert set(tissue_mapping.keys()) == {str(t) for t in expected_tissues}
            assert set(cell_mapping.keys()) == {str(c) for c in expected_cells}


class TestDataFormatEdgeCases:
    """Test edge cases and error conditions in DataFormat."""

    def test_convert_genes_with_empty_adata(self):
        """Test converting genes with empty AnnData."""
        data_format = DataFormat(gene_names=["gene1", "gene2"], n_genes=2)

        # Create empty AnnData
        X = csr_matrix((0, 0), dtype=np.float32)
        var = pd.DataFrame(index=[])
        adata = AnnData(X=X, var=var)

        # This should handle the edge case gracefully
        result = data_format.convert_genes_expression_matrix(adata)
        assert result.n_obs == 0

    def test_reorder_genes_to_match_format_alias(self):
        """Test that reorder_genes_to_match_format calls convert_genes_expression_matrix."""
        data_format = DataFormat(gene_names=["gene1", "gene2"], n_genes=2)

        X = csr_matrix(np.array([[2.0, 1.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene2", "gene1"])
        adata = AnnData(X=X, var=var)

        result = data_format.reorder_genes_to_match_format(adata)

        # Should reorder genes correctly
        assert result.var_names.tolist() == ["gene1", "gene2"]
        expected_data = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(result.X.toarray(), expected_data)

    def test_prepare_adata_for_training_with_reorder(self):
        """Test prepare_adata_for_training with reorder_genes=True."""
        data_format = DataFormat(gene_names=["gene1", "gene2"], n_genes=2)

        X = csr_matrix(np.array([[2.0, 1.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene2", "gene1"])
        adata = AnnData(X=X, var=var)

        result = data_format.prepare_adata_for_training(adata, reorder_genes=True)

        # Should reorder genes
        assert result.var_names.tolist() == ["gene1", "gene2"]

    def test_prepare_adata_for_training_without_reorder(self):
        """Test prepare_adata_for_training with reorder_genes=False."""
        data_format = DataFormat(gene_names=["gene1", "gene2"], n_genes=2)

        X = csr_matrix(np.array([[2.0, 1.0]]), dtype=np.float32)
        var = pd.DataFrame(index=["gene2", "gene1"])
        adata = AnnData(X=X, var=var)

        result = data_format.prepare_adata_for_training(adata, reorder_genes=False)

        # Should NOT reorder genes
        assert result.var_names.tolist() == ["gene2", "gene1"]
        assert result is adata  # Should return the same object

    def test_data_format_str_representation(self):
        """Test the __str__ method of DataFormat."""
        data_format = DataFormat(
            gene_names=["gene1", "gene2"],
            n_genes=2,
            target_sum=10000.0,
            use_log_transform=True,
            aux_categorical_mappings={"tissue": {"brain": 0, "liver": 1}},
        )
        data_format.genes_mu = np.array([0.1, 0.2])
        data_format.genes_sigma = np.array([1.0, 1.1])

        str_repr = str(data_format)

        # Check that key information is in the string representation
        assert "n_genes=2" in str_repr
        assert "use_log_transform=True" in str_repr
        assert "target_sum=10000.0" in str_repr
        assert "tissue" in str_repr
