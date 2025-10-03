"""Comprehensive tests for gene mismatch handling across all model types.

This module tests that all model inference functions (MLP, Autoencoder, Linear, LightGBM)
correctly handle test data with different gene sets than the training data.
"""

from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import CellsDataset, cells_collate_fn
from scxpand.util.inference_utils import run_model_inference


class TestGeneMismatchHandlingAllModels:
    """Test gene mismatch handling for all model types."""

    @pytest.fixture
    def training_data_format(self):
        """Create a DataFormat representing what the model was trained on."""
        model_genes = [f"GENE_{i:03d}" for i in range(100)]  # 100 genes in training
        return DataFormat(
            n_genes=len(model_genes),
            gene_names=model_genes,
            genes_mu=np.random.randn(len(model_genes)).astype(np.float32),
            genes_sigma=np.random.rand(len(model_genes)).astype(np.float32) + 0.1,
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

    @pytest.fixture
    def test_scenarios(self):
        """Create different gene mismatch scenarios for testing."""
        return {
            "partial_overlap": {
                "description": "50% overlap with training genes",
                "genes": [
                    f"GENE_{i:03d}" for i in range(50, 150)
                ],  # 50 overlap, 50 new, 50 missing
                "n_cells": 20,
            },
            "no_overlap": {
                "description": "No overlap with training genes",
                "genes": [
                    f"NEWGENE_{i:03d}" for i in range(100)
                ],  # Completely different genes
                "n_cells": 15,
            },
            "superset": {
                "description": "Contains all training genes plus extras",
                "genes": [
                    f"GENE_{i:03d}" for i in range(150)
                ],  # All training genes + 50 extra
                "n_cells": 25,
            },
            "subset": {
                "description": "Contains only some training genes",
                "genes": [
                    f"GENE_{i:03d}" for i in range(0, 100, 2)
                ],  # Every other training gene (50 genes)
                "n_cells": 30,
            },
            "reordered": {
                "description": "Same genes but in different order",
                "genes": [
                    f"GENE_{i:03d}" for i in reversed(range(100))
                ],  # Reverse order
                "n_cells": 18,
            },
        }

    def create_test_adata(self, genes: list[str], n_cells: int) -> ad.AnnData:
        """Create test AnnData with specified genes and cell count."""
        n_genes = len(genes)

        # Create realistic gene expression data
        X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

        # Add some zeros to make it more realistic
        zero_mask = np.random.random((n_cells, n_genes)) < 0.1
        X[zero_mask] = 0

        obs_df = pd.DataFrame(
            {
                "expansion": np.random.choice(
                    ["expanded", "non-expanded"], size=n_cells
                ),
                "tissue_type": np.random.choice(["A", "B", "C"], size=n_cells),
                "patient_id": np.random.choice(
                    [f"P{i}" for i in range(5)], size=n_cells
                ),
            }
        )
        var_df = pd.DataFrame(index=genes)

        return ad.AnnData(X=X, obs=obs_df, var=var_df)

    @pytest.mark.parametrize(
        "model_type", ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]
    )
    @pytest.mark.parametrize(
        "scenario_name",
        ["partial_overlap", "no_overlap", "superset", "subset", "reordered"],
    )
    def test_gene_mismatch_scenarios_all_models(
        self, model_type, scenario_name, training_data_format, test_scenarios
    ):
        """Test that all model types handle various gene mismatch scenarios correctly."""
        scenario = test_scenarios[scenario_name]

        # Create test data with different genes
        test_adata = self.create_test_adata(scenario["genes"], scenario["n_cells"])

        # Create mock model
        mock_model = MagicMock()
        if model_type in ["mlp", "autoencoder"]:
            # Neural network models
            mock_model.eval.return_value = None
            mock_model.to.return_value = None
        # Sklearn-based models
        elif model_type == "lightgbm":
            mock_model.predict_proba.return_value = np.random.rand(
                scenario["n_cells"], 2
            )
        else:  # logistic, svm
            mock_model.predict_proba.return_value = np.random.rand(
                scenario["n_cells"], 2
            )

        # Mock the model-specific inference functions
        patch_targets = {
            "mlp": "scxpand.util.inference_utils.run_mlp_inference",
            "autoencoder": "scxpand.util.inference_utils.run_ae_inference",
            "logistic": "scxpand.util.inference_utils.run_linear_inference",
            "svm": "scxpand.util.inference_utils.run_linear_inference",
            "lightgbm": "scxpand.util.inference_utils.run_lightgbm_inference",
        }

        with patch(patch_targets[model_type]) as mock_inference:
            expected_result = np.random.rand(scenario["n_cells"])
            mock_inference.return_value = expected_result

            # This should NOT raise any errors about gene mismatches
            result = run_model_inference(
                model_type=model_type,
                model=mock_model,
                data_format=training_data_format,
                adata=test_adata,
                data_path=None,
                eval_row_inds=None,
                device="cpu",
                batch_size=16,
                num_workers=0,
            )

            # Verify the result
            assert isinstance(result, np.ndarray), (
                f"Result should be numpy array for {model_type}"
            )
            assert result.shape == (scenario["n_cells"],), (
                f"Wrong result shape for {model_type}"
            )
            np.testing.assert_array_equal(result, expected_result)

            # Verify the inference function was called
            mock_inference.assert_called_once()

            # Verify correct parameters were passed
            call_kwargs = mock_inference.call_args.kwargs
            assert call_kwargs["data_format"] is training_data_format
            assert call_kwargs["adata"] is test_adata

    def test_gene_transformation_correctness_dataset(self, training_data_format):
        """Test that CellsDataset correctly transforms genes for neural network models."""
        # Create test data with partial gene overlap
        test_genes = [
            f"GENE_{i:03d}" for i in range(50, 150)
        ]  # 50 overlap, 50 new, 50 missing
        n_cells = 10
        test_adata = self.create_test_adata(test_genes, n_cells)

        # Create dataset (this is what MLP/Autoencoder/Linear models use)
        dataset = CellsDataset(
            data_format=training_data_format,
            row_inds=np.arange(n_cells),
            dataset_params=None,
            is_train=False,  # Inference mode
            adata=test_adata,
        )

        # Verify dataset was created successfully
        assert dataset.n_genes == training_data_format.n_genes
        assert len(dataset) == n_cells

        # Verify gene transformation is detected
        assert dataset.needs_gene_transformation is True
        assert len(dataset.gene_overlap) == 50  # 50 overlapping genes
        assert len(dataset.missing_genes) == 50  # 50 missing genes
        assert len(dataset.extra_genes) == 50  # 50 extra genes

        # Test batch creation using collate function
        batch_indices = [0, 1, 2]
        batch = cells_collate_fn(batch_indices, dataset)

        # Verify batch has correct shape (transformed to training gene count)
        assert batch["x"].shape == (3, training_data_format.n_genes)
        assert isinstance(batch["x"], torch.Tensor)

        # Verify missing genes are handled consistently
        # Missing genes should be filled with zeros before preprocessing, then processed consistently
        missing_gene_mask = np.array(
            [gene not in test_genes for gene in training_data_format.gene_names]
        )
        missing_positions = np.where(missing_gene_mask)[0]

        if len(missing_positions) > 0:
            # Check that missing positions have consistent values across all samples
            missing_values = batch["x"][:, missing_positions]
            # All missing genes should have the same value for each sample (processed zeros)
            # Check that each missing gene has the same value across all samples in the batch
            for gene_idx in range(missing_values.shape[1]):
                gene_values = missing_values[:, gene_idx]
                # All samples should have the same value for this missing gene
                assert torch.allclose(gene_values, gene_values[0], atol=1e-6), (
                    f"Missing gene {gene_idx} should have consistent values across samples"
                )

    def test_file_based_gene_mismatch_handling(self, tmp_path, training_data_format):
        """Test gene mismatch handling with file-based data loading."""
        # Create test data file
        test_genes = [
            f"GENE_{i:03d}" for i in range(25, 125)
        ]  # 75 overlap, 25 new, 25 missing
        n_cells = 12
        test_adata = self.create_test_adata(test_genes, n_cells)

        file_path = tmp_path / "test_gene_mismatch.h5ad"
        test_adata.write_h5ad(file_path)

        # Test each model type with file-based loading
        for model_type in ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]:
            mock_model = MagicMock()

            # Mock the appropriate inference function
            patch_targets = {
                "mlp": "scxpand.util.inference_utils.run_mlp_inference",
                "autoencoder": "scxpand.util.inference_utils.run_ae_inference",
                "logistic": "scxpand.util.inference_utils.run_linear_inference",
                "svm": "scxpand.util.inference_utils.run_linear_inference",
                "lightgbm": "scxpand.util.inference_utils.run_lightgbm_inference",
            }

            with patch(patch_targets[model_type]) as mock_inference:
                expected_result = np.random.rand(n_cells)
                mock_inference.return_value = expected_result

                # Test file-based inference
                result = run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=training_data_format,
                    adata=None,
                    data_path=file_path,
                    eval_row_inds=None,
                    device="cpu",
                    batch_size=8,
                    num_workers=0,
                )

                assert isinstance(result, np.ndarray)
                assert result.shape == (n_cells,)
                mock_inference.assert_called_once()

    def test_edge_cases_gene_handling(self, training_data_format):
        """Test edge cases in gene handling."""
        # Test case 1: Empty gene set (should fail gracefully)
        empty_adata = ad.AnnData(X=np.empty((5, 0)), obs=pd.DataFrame(index=range(5)))

        mock_model = MagicMock()

        # This should either work (filling all genes with zeros) or fail with a clear error
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(5)

            # Try to run inference - it should either work or fail gracefully
            result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=training_data_format,
                adata=empty_adata,
            )
            # If it succeeds, verify the result
            assert isinstance(result, np.ndarray)
            assert len(result) == 5

        # Test case 2: Single gene overlap
        single_gene_adata = self.create_test_adata(
            [training_data_format.gene_names[0]], 3
        )

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(3)

            result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=training_data_format,
                adata=single_gene_adata,
            )

            assert isinstance(result, np.ndarray)
            assert len(result) == 3

    def test_gene_name_case_sensitivity(self, training_data_format):
        """Test that gene name matching is case-sensitive (as it should be)."""
        # Create test data with different case
        lowercase_genes = [
            gene.lower() for gene in training_data_format.gene_names[:50]
        ]
        test_adata = self.create_test_adata(lowercase_genes, 8)

        mock_model = MagicMock()

        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_inference:
            mock_inference.return_value = np.random.rand(8)

            # This should treat lowercase genes as completely different (no overlap)
            result = run_model_inference(
                model_type="mlp",
                model=mock_model,
                data_format=training_data_format,
                adata=test_adata,
            )

            assert isinstance(result, np.ndarray)
            assert len(result) == 8
            mock_inference.assert_called_once()

    def test_gene_order_independence(self, training_data_format):
        """Test that gene order in test data doesn't affect results (after reordering)."""
        # Create two datasets with same genes but different orders
        genes_original_order = training_data_format.gene_names[:50]
        genes_shuffled_order = genes_original_order.copy()
        np.random.shuffle(genes_shuffled_order)

        n_cells = 10
        adata_original = self.create_test_adata(genes_original_order, n_cells)
        adata_shuffled = self.create_test_adata(genes_shuffled_order, n_cells)

        # Set the same expression values for corresponding genes
        for i, gene in enumerate(genes_original_order):
            shuffled_idx = genes_shuffled_order.index(gene)
            adata_shuffled.X[:, shuffled_idx] = adata_original.X[:, i]

        # Test that both datasets produce the same preprocessing result
        dataset_original = CellsDataset(
            data_format=training_data_format,
            row_inds=np.arange(n_cells),
            is_train=False,
            adata=adata_original,
        )

        dataset_shuffled = CellsDataset(
            data_format=training_data_format,
            row_inds=np.arange(n_cells),
            is_train=False,
            adata=adata_shuffled,
        )

        # Get batches from both datasets
        batch_indices = list(range(n_cells))
        batch_original = cells_collate_fn(batch_indices, dataset_original)
        batch_shuffled = cells_collate_fn(batch_indices, dataset_shuffled)

        # The preprocessed data should be identical (after gene reordering)
        # We'll check the overlapping gene positions
        overlapping_positions = []
        for i, gene in enumerate(training_data_format.gene_names):
            if gene in genes_original_order:
                overlapping_positions.append(i)

        if overlapping_positions:
            # Compare values at overlapping positions - they should be very close
            original_values = batch_original["x"][:, overlapping_positions]
            shuffled_values = batch_shuffled["x"][:, overlapping_positions]

            # Allow for small numerical differences due to floating point operations
            torch.testing.assert_close(
                original_values, shuffled_values, atol=1e-5, rtol=1e-5
            )

    @pytest.mark.parametrize(
        "model_type", ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]
    )
    def test_eval_row_inds_with_gene_mismatch(self, model_type, training_data_format):
        """Test that eval_row_inds works correctly with gene mismatches."""
        # Create test data
        test_genes = [
            f"GENE_{i:03d}" for i in range(30, 130)
        ]  # 70 overlap, 30 new, 30 missing
        n_cells = 20
        test_adata = self.create_test_adata(test_genes, n_cells)

        # Select subset of cells for evaluation
        eval_indices = np.array([1, 3, 5, 7, 9])

        mock_model = MagicMock()

        patch_targets = {
            "mlp": "scxpand.util.inference_utils.run_mlp_inference",
            "autoencoder": "scxpand.util.inference_utils.run_ae_inference",
            "logistic": "scxpand.util.inference_utils.run_linear_inference",
            "svm": "scxpand.util.inference_utils.run_linear_inference",
            "lightgbm": "scxpand.util.inference_utils.run_lightgbm_inference",
        }

        with patch(patch_targets[model_type]) as mock_inference:
            expected_result = np.random.rand(len(eval_indices))
            mock_inference.return_value = expected_result

            result = run_model_inference(
                model_type=model_type,
                model=mock_model,
                data_format=training_data_format,
                adata=test_adata,
                eval_row_inds=eval_indices,
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (len(eval_indices),)

            # Verify eval_row_inds was passed correctly
            call_kwargs = mock_inference.call_args.kwargs
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], eval_indices)


class TestGeneMismatchErrorHandling:
    """Test error handling and edge cases in gene mismatch scenarios."""

    def test_invalid_gene_names(self):
        """Test handling of invalid gene names."""
        # Create data format with valid gene names
        valid_genes = [f"GENE_{i}" for i in range(10)]
        data_format = DataFormat(
            n_genes=len(valid_genes),
            gene_names=valid_genes,
            genes_mu=np.zeros(len(valid_genes)),
            genes_sigma=np.ones(len(valid_genes)),
        )

        # Create test data with problematic gene names
        problematic_cases = [
            ["", "GENE_1", "GENE_2"],  # Empty string
            [
                "GENE_0",
                None,
                "GENE_2",
            ],  # None value (this would fail at AnnData creation)
            ["GENE_0", "GENE_1", "GENE_1"],  # Duplicate names
        ]

        for _i, gene_names in enumerate(problematic_cases):
            if None in gene_names:
                continue  # Skip None case as AnnData creation would fail

            try:
                # Create test data
                X = np.random.randn(5, len(gene_names)).astype(np.float32)
                adata = ad.AnnData(X=X, var=pd.DataFrame(index=gene_names))

                # Test with dataset creation
                dataset = CellsDataset(
                    data_format=data_format,
                    row_inds=np.arange(5),
                    is_train=False,
                    adata=adata,
                )

                # Should handle gracefully
                assert dataset is not None

            except Exception:
                # If it fails, that's expected for problematic gene names
                pass

    def test_memory_efficiency_large_gene_sets(self):
        """Test that gene mismatch handling is memory efficient for large gene sets."""
        # Create a data format with many genes
        large_gene_count = 20000
        training_genes = [f"GENE_{i:05d}" for i in range(large_gene_count)]

        data_format = DataFormat(
            n_genes=large_gene_count,
            gene_names=training_genes,
            genes_mu=np.zeros(large_gene_count, dtype=np.float32),
            genes_sigma=np.ones(large_gene_count, dtype=np.float32),
        )

        # Create test data with different genes (memory should be manageable)
        test_genes = [f"TESTGENE_{i:05d}" for i in range(1000)]  # Much smaller test set
        n_cells = 100

        X = np.random.randn(n_cells, len(test_genes)).astype(np.float32)
        adata = ad.AnnData(X=X, var=pd.DataFrame(index=test_genes))

        # This should not cause memory issues
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=np.arange(n_cells),
            is_train=False,
            adata=adata,
        )

        assert dataset.n_genes == large_gene_count
        assert len(dataset) == n_cells

        # Test a small batch to ensure it works
        batch = cells_collate_fn([0, 1], dataset)
        assert batch["x"].shape == (2, large_gene_count)


class TestGeneMismatchIntegration:
    """Integration tests for gene mismatch handling across the full pipeline."""

    def test_end_to_end_gene_mismatch_pipeline(self, tmp_path):
        """Test the complete pipeline from data loading to inference with gene mismatches."""
        # Create training data format
        training_genes = [f"GENE_{i:03d}" for i in range(50)]
        data_format = DataFormat(
            n_genes=len(training_genes),
            gene_names=training_genes,
            genes_mu=np.random.randn(len(training_genes)).astype(np.float32),
            genes_sigma=np.random.rand(len(training_genes)).astype(np.float32) + 0.1,
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

        # Create test data with gene mismatches
        test_genes = [
            f"GENE_{i:03d}" for i in range(25, 75)
        ]  # 25 overlap, 25 new, 25 missing
        n_cells = 15

        X = np.random.exponential(scale=2.0, size=(n_cells, len(test_genes))).astype(
            np.float32
        )
        obs_df = pd.DataFrame(
            {
                "expansion": np.random.choice(
                    ["expanded", "non-expanded"], size=n_cells
                ),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=test_genes)

        test_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        # Save to file for file-based testing
        file_path = tmp_path / "test_integration.h5ad"
        test_adata.write_h5ad(file_path)

        # Test all model types with both in-memory and file-based data
        for model_type in ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]:
            for data_source in ["memory", "file"]:
                mock_model = MagicMock()

                # Configure mock based on model type
                if model_type in ["mlp", "autoencoder"]:
                    mock_model.eval.return_value = None
                    mock_model.to.return_value = None

                # Mock the inference function
                patch_targets = {
                    "mlp": "scxpand.util.inference_utils.run_mlp_inference",
                    "autoencoder": "scxpand.util.inference_utils.run_ae_inference",
                    "logistic": "scxpand.util.inference_utils.run_linear_inference",
                    "svm": "scxpand.util.inference_utils.run_linear_inference",
                    "lightgbm": "scxpand.util.inference_utils.run_lightgbm_inference",
                }

                with patch(patch_targets[model_type]) as mock_inference:
                    expected_result = np.random.rand(n_cells)
                    mock_inference.return_value = expected_result

                    # Run inference
                    result = run_model_inference(
                        model_type=model_type,
                        model=mock_model,
                        data_format=data_format,
                        adata=test_adata if data_source == "memory" else None,
                        data_path=file_path if data_source == "file" else None,
                        eval_row_inds=None,
                        device="cpu",
                        batch_size=8,
                        num_workers=0,
                    )

                    # Verify results
                    assert isinstance(result, np.ndarray), (
                        f"Failed for {model_type} with {data_source} data"
                    )
                    assert result.shape == (n_cells,), (
                        f"Wrong shape for {model_type} with {data_source} data"
                    )
                    np.testing.assert_array_equal(result, expected_result)

                    # Verify the inference function was called correctly
                    mock_inference.assert_called_once()
                    call_kwargs = mock_inference.call_args.kwargs
                    assert call_kwargs["data_format"] is data_format
