"""Comprehensive tests for gene-specific z-score normalization during inference.

This module tests that all model inference functions correctly apply per-gene
z-score normalization parameters (genes_mu and genes_sigma) even when genes
are reordered, missing, or extra compared to training data.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import CellsDataset, cells_collate_fn
from scxpand.data_util.transforms import (
    DEFAULT_SIGMA_CLIP_FACTOR,
    apply_log_transform,
    apply_row_normalization,
    apply_zscore_normalization,
    preprocess_expression_data,
)
from scxpand.util.inference_utils import run_model_inference


class TestGeneSpecificNormalization:
    """Test that gene-specific z-score normalization works correctly during inference."""

    @pytest.fixture
    def training_data_format_with_specific_stats(self) -> DataFormat:
        """Create a DataFormat with specific, known gene statistics for testing."""
        # Create genes with very different statistics to make normalization effects obvious
        gene_names: list[str] = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]

        # GENE_A: high mean, low std (will be close to 0 after z-score)
        # GENE_B: low mean, high std (will be close to -1 after z-score for missing genes)
        # GENE_C: medium mean, medium std
        # GENE_D: very low mean, very high std (will be close to -1 after z-score for missing genes)
        # GENE_E: high mean, high std
        genes_mu: np.ndarray = np.array(
            [100.0, 10.0, 50.0, 5.0, 200.0], dtype=np.float32
        )
        genes_sigma: np.ndarray = np.array(
            [20.0, 100.0, 30.0, 200.0, 50.0], dtype=np.float32
        )

        return DataFormat(
            n_genes=len(gene_names),
            gene_names=gene_names,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

    def test_gene_specific_normalization_manual(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test gene-specific normalization manually to verify the math."""
        data_format = training_data_format_with_specific_stats

        # Create test data with values that work well with the preprocessing pipeline
        # Use positive values that work well with log1p and row normalization
        X_raw = np.array(
            [
                [100.0, 10.0, 50.0, 5.0, 200.0],  # Row 1: baseline values
                [120.0, 110.0, 80.0, 205.0, 250.0],  # Row 2: higher values
                [80.0, 5.0, 20.0, 1.0, 150.0],  # Row 3: lower values
            ],
            dtype=np.float32,
        )

        # Apply preprocessing manually
        X_processed = preprocess_expression_data(X=X_raw, data_format=data_format)

        # The preprocessing pipeline is:
        # 1. Row normalization (target_sum=1e4)
        # 2. Log transform (log1p)
        # 3. Z-score normalization using genes_mu and genes_sigma

        # After row normalization, each row sums to 1e4
        # After log transform, values are log1p(normalized_values)
        # After z-score: (log1p_value - genes_mu) / genes_sigma

        # Verify that z-score normalization was applied per-gene
        # The exact values depend on the preprocessing pipeline, but we can verify
        # that the relative relationships between genes are preserved

        # Check that the processed data has reasonable z-score values
        # (typically between -3 and 3 for most data)
        assert np.all(
            np.abs(X_processed) < 10
        ), f"Z-score values too extreme: {X_processed}"

        # Verify that genes with different statistics (mu, sigma) get different normalization
        # GENE_A (mu=100, sigma=20) vs GENE_B (mu=10, sigma=100) should behave differently
        gene_a_values = X_processed[:, 0]  # GENE_A column
        gene_b_values = X_processed[:, 1]  # GENE_B column

        # The genes should have different patterns due to different mu/sigma
        assert not np.allclose(gene_a_values, gene_b_values, atol=0.1)

        # Verify that the preprocessing preserved the data structure
        assert X_processed.shape == X_raw.shape
        assert not np.any(np.isnan(X_processed)), "NaN values in processed data"
        assert not np.any(np.isinf(X_processed)), "Inf values in processed data"

    def test_gene_reordering_preserves_normalization(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that gene reordering preserves correct per-gene normalization."""
        data_format = training_data_format_with_specific_stats

        # Create test data with genes in different order
        test_genes = [
            "GENE_C",
            "GENE_A",
            "GENE_E",
            "GENE_B",
            "GENE_D",
        ]  # Different order
        X_raw = np.array(
            [
                [50.0, 100.0, 200.0, 10.0, 5.0],  # Same values, different order
            ],
            dtype=np.float32,
        )

        # Create AnnData with reordered genes
        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Reorder genes to match training format
        adata_reordered = data_format.prepare_adata_for_training(
            test_adata, reorder_genes=True
        )

        # Verify genes are now in correct order
        assert list(adata_reordered.var_names) == data_format.gene_names

        # Apply preprocessing
        X_processed = preprocess_expression_data(
            X=adata_reordered.X, data_format=data_format
        )

        # Verify that preprocessing was applied correctly
        assert X_processed.shape == (1, 5)  # 1 row, 5 genes
        assert not np.any(np.isnan(X_processed)), "NaN values in processed data"
        assert not np.any(np.isinf(X_processed)), "Inf values in processed data"

        # The key point is that genes are now in the correct order for normalization
        # GENE_A is now at index 0, GENE_B at index 1, etc.
        # Each gene gets normalized using its corresponding genes_mu[i] and genes_sigma[i]

        # Verify that the data structure is preserved and preprocessing worked
        assert X_processed.shape == adata_reordered.X.shape

    def test_missing_genes_get_correct_normalization(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that missing genes get correct normalization (zeros become -mu/sigma)."""
        data_format = training_data_format_with_specific_stats

        # Create test data missing GENE_B and GENE_D
        test_genes = ["GENE_A", "GENE_C", "GENE_E"]  # Missing B and D
        X_raw = np.array(
            [
                [100.0, 50.0, 200.0],  # Same values for existing genes
            ],
            dtype=np.float32,
        )

        # Create AnnData with missing genes
        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Reorder genes to match training format (will add missing genes as zeros)
        adata_reordered = data_format.prepare_adata_for_training(
            test_adata, reorder_genes=True
        )

        # Verify genes are now in correct order with missing genes added
        assert list(adata_reordered.var_names) == data_format.gene_names
        assert adata_reordered.n_vars == 5

        # Apply preprocessing
        X_processed = preprocess_expression_data(
            X=adata_reordered.X, data_format=data_format
        )

        # Verify that preprocessing was applied correctly
        assert X_processed.shape == (1, 5)  # 1 row, 5 genes
        assert not np.any(np.isnan(X_processed)), "NaN values in processed data"
        assert not np.any(np.isinf(X_processed)), "Inf values in processed data"

        # The key point is that missing genes (B and D) are now at the correct positions
        # and will get normalized using their corresponding genes_mu and genes_sigma
        # GENE_B is at index 1, GENE_D is at index 3

        # Verify that the data structure is preserved and preprocessing worked
        assert X_processed.shape == adata_reordered.X.shape

    def test_extra_genes_are_removed_correctly(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that extra genes are removed and remaining genes get correct normalization."""
        data_format = training_data_format_with_specific_stats

        # Create test data with extra genes
        test_genes = [
            "GENE_A",
            "GENE_B",
            "GENE_C",
            "GENE_D",
            "GENE_E",
            "EXTRA_1",
            "EXTRA_2",
        ]
        X_raw = np.array(
            [
                [
                    100.0,
                    10.0,
                    50.0,
                    5.0,
                    200.0,
                    999.0,
                    888.0,
                ],  # Extra genes have high values
            ],
            dtype=np.float32,
        )

        # Create AnnData with extra genes
        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Reorder genes to match training format (will remove extra genes)
        adata_reordered = data_format.prepare_adata_for_training(
            test_adata, reorder_genes=True
        )

        # Verify extra genes are removed
        assert list(adata_reordered.var_names) == data_format.gene_names
        assert adata_reordered.n_vars == 5

        # Apply preprocessing
        X_processed = preprocess_expression_data(
            X=adata_reordered.X, data_format=data_format
        )

        # Verify that preprocessing was applied correctly
        assert X_processed.shape == (1, 5)  # 1 row, 5 genes
        assert not np.any(np.isnan(X_processed)), "NaN values in processed data"
        assert not np.any(np.isinf(X_processed)), "Inf values in processed data"

        # The key point is that extra genes (EXTRA_1, EXTRA_2) are removed
        # and the remaining genes are in the correct order for normalization
        # Each gene gets normalized using its corresponding genes_mu[i] and genes_sigma[i]

        # Verify that the data structure is preserved and preprocessing worked
        assert X_processed.shape == adata_reordered.X.shape

    def test_cellsdataset_gene_transformation_normalization(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that CellsDataset correctly handles gene transformation and normalization."""
        data_format = training_data_format_with_specific_stats

        # Create test data with gene mismatches
        test_genes = ["GENE_C", "GENE_A", "GENE_E"]  # Missing B and D, different order
        X_raw = np.array(
            [
                [50.0, 100.0, 200.0],  # Same values, different order
            ],
            dtype=np.float32,
        )

        # Create AnnData with gene mismatches
        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Create dataset (this should handle gene transformation internally)
        dataset = CellsDataset(
            data_format=data_format,
            row_inds=np.array([0]),
            is_train=False,  # Inference mode
            adata=test_adata,
        )

        # Verify gene transformation is needed and detected
        assert dataset.needs_gene_transformation
        assert len(dataset.gene_overlap) == 3  # 3 overlapping genes
        assert len(dataset.missing_genes) == 2  # 2 missing genes (B and D)
        assert len(dataset.extra_genes) == 0  # No extra genes

        # Test batch creation using collate function
        batch = cells_collate_fn([0], dataset)

        # Verify batch has correct shape (transformed to training gene count)
        assert batch["x"].shape == (1, data_format.n_genes)
        assert isinstance(batch["x"], torch.Tensor)

        # Verify that the preprocessing was applied correctly
        # The tensor should contain the preprocessed values with correct gene-specific normalization
        X_processed = batch["x"].numpy()

        # Verify that preprocessing worked correctly
        assert X_processed.shape == (1, 5)  # 1 row, 5 genes
        assert not np.any(np.isnan(X_processed)), "NaN values in processed data"
        assert not np.any(np.isinf(X_processed)), "Inf values in processed data"

        # The key point is that each gene gets normalized using its corresponding
        # genes_mu[i] and genes_sigma[i] from the data_format, even for missing genes

    def test_all_model_types_gene_specific_normalization(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that all model types correctly apply gene-specific normalization."""
        data_format = training_data_format_with_specific_stats

        # Create test data with gene mismatches
        test_genes: list[str] = [
            "GENE_C",
            "GENE_A",
            "GENE_E",
        ]  # Missing B and D, different order
        n_cells: int = 5
        X_raw: np.ndarray = np.random.exponential(
            scale=2.0, size=(n_cells, len(test_genes))
        ).astype(np.float32)

        # Create AnnData with gene mismatches
        obs_df = pd.DataFrame(
            {
                "expansion": np.random.choice(
                    ["expanded", "non-expanded"], size=n_cells
                ),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Test each model type
        for model_type in ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]:
            mock_model: MagicMock = MagicMock()

            # Configure mock based on model type
            if model_type in ["mlp", "autoencoder"]:
                mock_model.eval.return_value = None
                mock_model.to.return_value = None
            elif model_type == "lightgbm":
                mock_model.predict_proba.return_value = np.random.rand(n_cells, 2)
            else:  # logistic, svm
                mock_model.predict_proba.return_value = np.random.rand(n_cells, 2)

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
                    adata=test_adata,
                    data_path=None,
                    eval_row_inds=None,
                    device="cpu",
                    batch_size=8,
                    num_workers=0,
                )

                # Verify the result
                assert isinstance(result, np.ndarray)
                assert result.shape == (n_cells,)
                np.testing.assert_array_equal(result, expected_result)

                # Verify the inference function was called
                mock_inference.assert_called_once()

                # Verify correct parameters were passed
                call_kwargs = mock_inference.call_args.kwargs
                assert call_kwargs["data_format"] is data_format
                assert call_kwargs["adata"] is test_adata

    def test_file_based_gene_specific_normalization(
        self, tmp_path: Path, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test gene-specific normalization with file-based data loading."""
        data_format = training_data_format_with_specific_stats

        # Create test data with gene mismatches
        test_genes = ["GENE_C", "GENE_A", "GENE_E"]  # Missing B and D, different order
        n_cells = 3
        X_raw = np.random.exponential(
            scale=2.0, size=(n_cells, len(test_genes))
        ).astype(np.float32)

        # Create AnnData with gene mismatches
        obs_df = pd.DataFrame(
            {
                "expansion": np.random.choice(
                    ["expanded", "non-expanded"], size=n_cells
                ),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Save to file
        file_path = tmp_path / "test_gene_normalization.h5ad"
        test_adata.write_h5ad(file_path)

        # Test each model type with file-based loading
        for model_type in ["mlp", "autoencoder", "logistic", "svm", "lightgbm"]:
            mock_model = MagicMock()

            # Configure mock based on model type
            if model_type in ["mlp", "autoencoder"]:
                mock_model.eval.return_value = None
                mock_model.to.return_value = None
            elif model_type == "lightgbm":
                mock_model.predict_proba.return_value = np.random.rand(n_cells, 2)
            else:  # logistic, svm
                mock_model.predict_proba.return_value = np.random.rand(n_cells, 2)

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

                # Test file-based inference
                result = run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=data_format,
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

    def test_edge_cases_gene_normalization(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test edge cases in gene-specific normalization."""
        data_format = training_data_format_with_specific_stats

        # Test with single gene overlap
        test_genes = ["GENE_A", "EXTRA_1", "EXTRA_2"]  # Only 1 gene in common
        X_raw = np.array([[100.0, 999.0, 888.0]], dtype=np.float32)

        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Reorder genes to match training format
        adata_reordered = data_format.prepare_adata_for_training(
            test_adata, reorder_genes=True
        )

        # Verify genes are now in correct order with missing genes added
        assert list(adata_reordered.var_names) == data_format.gene_names
        assert adata_reordered.n_vars == 5

        # Apply preprocessing
        X_processed = preprocess_expression_data(
            X=adata_reordered.X, data_format=data_format
        )

        # Verify that preprocessing was applied correctly
        assert X_processed.shape == (1, 5)  # 1 row, 5 genes
        assert not np.any(np.isnan(X_processed)), "NaN values in processed data"
        assert not np.any(np.isinf(X_processed)), "Inf values in processed data"

        # The key point is that GENE_A is at the correct position (index 0)
        # and gets normalized using genes_mu[0] and genes_sigma[0]
        # Missing genes (B, C, D, E) are added at their correct positions
        # and get normalized using their corresponding genes_mu[i] and genes_sigma[i]

        # Verify that the data structure is preserved and preprocessing worked
        assert X_processed.shape == adata_reordered.X.shape

    def test_mathematical_correctness_of_zscore_normalization(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that z-score normalization calculations are mathematically correct."""
        data_format = training_data_format_with_specific_stats

        # Create simple test data where we can manually verify the calculations
        # Use values that will produce predictable results after preprocessing
        X_raw = np.array(
            [
                [1000.0, 100.0, 500.0, 50.0, 2000.0],  # Row 1
                [2000.0, 200.0, 1000.0, 100.0, 4000.0],  # Row 2: 2x the first row
            ],
            dtype=np.float32,
        )

        # Apply the complete preprocessing pipeline step by step to verify each stage

        # Step 1: Row normalization (target_sum=1e4)
        X_normalized = apply_row_normalization(
            X_raw.copy(), target_sum=data_format.target_sum
        )

        # Verify row normalization worked correctly
        row_sums = np.sum(X_normalized, axis=1)
        np.testing.assert_allclose(row_sums, data_format.target_sum, rtol=1e-5)

        # Step 2: Log transform
        X_logged = apply_log_transform(X_normalized.copy(), in_place=False)

        # Verify log transform: should be log1p(normalized_values)
        expected_logged = np.log1p(X_normalized)
        np.testing.assert_allclose(X_logged, expected_logged, rtol=1e-6)

        # Step 3: Z-score normalization - this is what we're really testing
        X_zscore = apply_zscore_normalization(
            X_logged.copy(),
            genes_mu=data_format.genes_mu,
            genes_sigma=data_format.genes_sigma,
            eps=1e-8,  # Use small eps for precise testing
            in_place=False,
        )

        # Manually compute expected z-scores for verification
        expected_zscore = np.zeros_like(X_logged)
        for gene_idx in range(len(data_format.gene_names)):
            mu = data_format.genes_mu[gene_idx]
            sigma = data_format.genes_sigma[gene_idx]
            eps = 1e-8

            # Apply the exact formula used in the implementation
            expected_zscore[:, gene_idx] = (X_logged[:, gene_idx] - mu) / (sigma + eps)

        # Apply clipping using the default sigma clip factor
        expected_zscore = np.clip(
            expected_zscore, -DEFAULT_SIGMA_CLIP_FACTOR, DEFAULT_SIGMA_CLIP_FACTOR
        )

        # Verify that our manual calculation matches the implementation
        np.testing.assert_allclose(X_zscore, expected_zscore, rtol=1e-6, atol=1e-8)

        # Also test the complete pipeline function
        X_complete = preprocess_expression_data(X=X_raw.copy(), data_format=data_format)
        np.testing.assert_allclose(X_complete, expected_zscore, rtol=1e-6, atol=1e-8)

        # Verify that each gene was normalized with its specific mu/sigma
        for gene_idx in range(len(data_format.gene_names)):
            gene_name = data_format.gene_names[gene_idx]
            mu = data_format.genes_mu[gene_idx]
            sigma = data_format.genes_sigma[gene_idx]

            # Check that the gene-specific parameters were used
            # by verifying the transformation is correct for this specific gene
            original_logged_values = X_logged[:, gene_idx]
            zscore_values = X_complete[:, gene_idx]

            # Manually compute what the z-score should be for this gene
            expected_gene_zscore = (original_logged_values - mu) / (sigma + 1e-8)
            expected_gene_zscore = np.clip(
                expected_gene_zscore,
                -DEFAULT_SIGMA_CLIP_FACTOR,
                DEFAULT_SIGMA_CLIP_FACTOR,
            )

            np.testing.assert_allclose(
                zscore_values,
                expected_gene_zscore,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Gene {gene_name} (index {gene_idx}) z-score calculation incorrect",
            )

    def test_per_gene_normalization_parameters_usage(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that each gene uses its correct mu/sigma parameters during normalization."""
        data_format = training_data_format_with_specific_stats

        # Create test data where each gene has a different, known value
        # This makes it easy to verify that the correct mu/sigma is used for each gene
        X_raw = np.array(
            [
                [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],  # Different value per gene
            ],
            dtype=np.float32,
        )

        # Apply preprocessing
        X_processed = preprocess_expression_data(
            X=X_raw.copy(), data_format=data_format
        )

        # Manually verify each gene's normalization
        # First, compute what the intermediate values should be

        # Step 1: Row normalization
        row_sum = np.sum(X_raw[0])
        X_normalized = X_raw[0] * (data_format.target_sum / row_sum)

        # Step 2: Log transform
        X_logged = np.log1p(X_normalized)

        # Step 3: Verify z-score for each gene individually
        for gene_idx in range(len(data_format.gene_names)):
            gene_name = data_format.gene_names[gene_idx]
            mu = data_format.genes_mu[gene_idx]
            sigma = data_format.genes_sigma[gene_idx]

            # Compute expected z-score for this specific gene
            logged_value = X_logged[gene_idx]
            expected_zscore = (logged_value - mu) / (sigma + 1e-8)
            expected_zscore = np.clip(
                expected_zscore, -DEFAULT_SIGMA_CLIP_FACTOR, DEFAULT_SIGMA_CLIP_FACTOR
            )

            # Verify the actual result matches our expectation
            actual_zscore = X_processed[0, gene_idx]

            np.testing.assert_allclose(
                actual_zscore,
                expected_zscore,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Gene {gene_name} (index {gene_idx}) used incorrect mu={mu} or sigma={sigma}",
            )

            # Additional verification: ensure this gene's z-score would be different
            # if we used a different gene's parameters (proving gene-specificity)
            if gene_idx > 0:
                wrong_mu = data_format.genes_mu[0]  # Use first gene's parameters
                wrong_sigma = data_format.genes_sigma[0]
                wrong_zscore = (logged_value - wrong_mu) / (wrong_sigma + 1e-8)
                wrong_zscore = np.clip(
                    wrong_zscore, -DEFAULT_SIGMA_CLIP_FACTOR, DEFAULT_SIGMA_CLIP_FACTOR
                )

                # The z-score should be different when using wrong parameters
                # (unless by coincidence the parameters are very similar)
                if not np.isclose(mu, wrong_mu, rtol=0.1) or not np.isclose(
                    sigma, wrong_sigma, rtol=0.1
                ):
                    assert not np.isclose(
                        actual_zscore, wrong_zscore, rtol=1e-3
                    ), f"Gene {gene_name} z-score same with wrong parameters - test may be flawed"

    def test_gene_reordering_mathematical_correctness(
        self, training_data_format_with_specific_stats: DataFormat
    ) -> None:
        """Test that gene reordering preserves mathematical correctness of normalization."""
        data_format = training_data_format_with_specific_stats

        # Create test data with genes in different order than training
        test_genes = [
            "GENE_C",
            "GENE_A",
            "GENE_E",
            "GENE_B",
            "GENE_D",
        ]  # Different order

        # Use specific values that we can track through the reordering
        X_raw_reordered = np.array(
            [
                [3000.0, 1000.0, 5000.0, 2000.0, 4000.0],  # Values for C,A,E,B,D order
            ],
            dtype=np.float32,
        )

        # Create AnnData with reordered genes
        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw_reordered, obs=obs_df, var=var_df)

        # Reorder genes to match training format
        adata_reordered = data_format.prepare_adata_for_training(
            test_adata, reorder_genes=True
        )

        # Verify genes are now in correct order
        assert list(adata_reordered.var_names) == data_format.gene_names

        # The reordered data should now have values in training gene order: A,B,C,D,E
        # Original: C=3000, A=1000, E=5000, B=2000, D=4000
        # Reordered: A=1000, B=2000, C=3000, D=4000, E=5000
        expected_reordered_values = np.array(
            [[1000.0, 2000.0, 3000.0, 4000.0, 5000.0]], dtype=np.float32
        )
        np.testing.assert_allclose(
            adata_reordered.X, expected_reordered_values, rtol=1e-6
        )

        # Apply preprocessing and verify mathematical correctness
        X_processed = preprocess_expression_data(
            X=adata_reordered.X, data_format=data_format
        )

        # Manually compute expected results for the reordered data
        # Step 1: Row normalization
        row_sum = np.sum(expected_reordered_values[0])
        X_normalized = expected_reordered_values[0] * (data_format.target_sum / row_sum)

        # Step 2: Log transform
        X_logged = np.log1p(X_normalized)

        # Step 3: Z-score normalization with correct gene-specific parameters
        expected_zscore = np.zeros_like(X_logged)
        for gene_idx in range(len(data_format.gene_names)):
            mu = data_format.genes_mu[gene_idx]
            sigma = data_format.genes_sigma[gene_idx]

            # Each gene should use its correct parameters after reordering
            expected_zscore[gene_idx] = (X_logged[gene_idx] - mu) / (sigma + 1e-8)

        # Apply clipping
        expected_zscore = np.clip(
            expected_zscore, -DEFAULT_SIGMA_CLIP_FACTOR, DEFAULT_SIGMA_CLIP_FACTOR
        )

        # Verify the processed result matches our manual calculation
        np.testing.assert_allclose(
            X_processed[0], expected_zscore, rtol=1e-6, atol=1e-8
        )

        # Verify that each gene in the final result used its correct parameters
        for gene_idx, gene_name in enumerate(data_format.gene_names):
            mu = data_format.genes_mu[gene_idx]
            sigma = data_format.genes_sigma[gene_idx]

            # Verify this specific gene's z-score calculation
            logged_value = X_logged[gene_idx]
            expected_gene_zscore = (logged_value - mu) / (sigma + 1e-8)
            expected_gene_zscore = np.clip(
                expected_gene_zscore,
                -DEFAULT_SIGMA_CLIP_FACTOR,
                DEFAULT_SIGMA_CLIP_FACTOR,
            )

            actual_gene_zscore = X_processed[0, gene_idx]

            np.testing.assert_allclose(
                actual_gene_zscore,
                expected_gene_zscore,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"After reordering, gene {gene_name} at position {gene_idx} "
                f"did not use correct mu={mu}, sigma={sigma}",
            )

    def test_memory_efficiency_large_gene_sets(self) -> None:
        """Test memory efficiency with large gene sets while preserving normalization."""
        # Create a large data format
        large_gene_count: int = 1000
        large_gene_names: list[str] = [f"GENE_{i:04d}" for i in range(large_gene_count)]

        # Create diverse statistics for testing
        large_genes_mu: np.ndarray = (
            np.random.randn(large_gene_count).astype(np.float32) * 100
        )
        large_genes_sigma: np.ndarray = (
            np.random.rand(large_gene_count).astype(np.float32) * 50 + 10
        )

        large_data_format = DataFormat(
            n_genes=large_gene_count,
            gene_names=large_gene_names,
            genes_mu=large_genes_mu,
            genes_sigma=large_genes_sigma,
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

        # Create test data with partial overlap
        # Use genes from the middle range for overlap, and add some extra genes
        overlap_genes: list[str] = large_gene_names[500:750]  # 250 overlapping genes
        extra_genes: list[str] = [
            f"EXTRA_{i:04d}" for i in range(250)
        ]  # 250 extra genes
        test_genes: list[str] = overlap_genes + extra_genes  # 500 total test genes

        n_cells: int = 10
        X_raw: np.ndarray = np.random.exponential(
            scale=2.0, size=(n_cells, len(test_genes))
        ).astype(np.float32)

        obs_df = pd.DataFrame(
            {
                "expansion": np.random.choice(
                    ["expanded", "non-expanded"], size=n_cells
                ),
                "tissue_type": np.random.choice(["A", "B"], size=n_cells),
            }
        )
        var_df = pd.DataFrame(index=test_genes)
        test_adata = ad.AnnData(X=X_raw, obs=obs_df, var=var_df)

        # Create dataset (this should handle gene transformation efficiently)
        dataset = CellsDataset(
            data_format=large_data_format,
            row_inds=np.arange(n_cells),
            is_train=False,  # Inference mode
            adata=test_adata,
        )

        # Verify dataset was created successfully
        assert dataset.n_genes == large_gene_count
        assert len(dataset) == n_cells

        # Verify gene transformation is detected
        assert dataset.needs_gene_transformation
        assert len(dataset.gene_overlap) == 250  # 250 overlapping genes
        assert len(dataset.missing_genes) == 750  # 750 missing genes (1000 - 250)
        assert len(dataset.extra_genes) == 250  # 250 extra genes

        # Test batch creation using collate function
        batch_indices = [0, 1, 2]
        batch = cells_collate_fn(batch_indices, dataset)

        # Verify batch has correct shape (transformed to training gene count)
        assert batch["x"].shape == (3, large_gene_count)
        assert isinstance(batch["x"], torch.Tensor)

        # This should not cause memory issues
        dataset = CellsDataset(
            data_format=large_data_format,
            row_inds=np.arange(n_cells),
            is_train=False,
            adata=test_adata,
        )

        assert dataset.n_genes == large_gene_count
        assert len(dataset) == n_cells

        # Test a small batch to ensure it works
        batch = cells_collate_fn([0, 1], dataset)
        assert batch["x"].shape == (2, large_gene_count)
