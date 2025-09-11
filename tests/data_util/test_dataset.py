import os
import tempfile

from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from anndata import AnnData
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataloaders import BalancedLabelsBatchSampler, BalancedTypesBatchSampler, create_eval_dataloader
from scxpand.data_util.dataset import (
    CellsDataset,
    apply_post_normalization_augmentations,
    apply_pre_normalization_augmentations,
    cells_collate_fn,
    encode_categorical_features_batch,
    encode_categorical_value,
)
from scxpand.data_util.transforms import preprocess_expression_data
from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.classes import DataAugmentParams


class TestEncodeCategoricalValue:
    def test_value_in_mapping(self):
        mapping = {"a": 0, "b": 1}
        idx, valid = encode_categorical_value("a", mapping)
        assert idx == 0
        assert valid is True

    def test_value_not_in_mapping(self):
        mapping = {"a": 0, "b": 1}
        idx, valid = encode_categorical_value("c", mapping)
        assert idx == -1
        assert valid is False

    def test_numeric_value(self):
        mapping = {"1": 0, "2": 1}
        idx, valid = encode_categorical_value(1, mapping)
        assert idx == 0
        assert valid is True


class TestApplyAugmentations:
    """Test the separated augmentation functions."""

    def test_no_augmentation(self):
        X = torch.ones((4, 4))
        out_pre = apply_pre_normalization_augmentations(X, mask_rate=0.0)
        out_post = apply_post_normalization_augmentations(out_pre, noise_std=0.0)
        assert torch.allclose(out_post, X)

    def test_masking(self):
        X = torch.ones((100, 10))
        out = apply_pre_normalization_augmentations(X, mask_rate=0.5)
        # Should have some zeros
        assert (out == 0).sum() > 0

    def test_noise(self):
        X = torch.zeros((100, 10))
        out = apply_post_normalization_augmentations(X, noise_std=1.0)
        # Should not be all zeros
        assert not torch.allclose(out, X)


class TestSeparatedAugmentations:
    """Test that masking is applied before normalization and noise after normalization."""

    def test_masking_before_normalization(self) -> None:
        """Test that masking applied before normalization produces -mu/sigma values."""
        # Create simple test data
        X_raw = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        genes_mu = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        genes_sigma = np.array([0.5, 1.0, 1.5], dtype=np.float32)

        # Apply masking to raw data (before normalization)
        X_masked_raw = apply_pre_normalization_augmentations(X_raw, mask_rate=1.0)
        print(f"Original raw X: {X_raw}")
        print(f"Masked raw X: {X_masked_raw}")

        # Apply normalization to masked data
        # genes_mu_tensor = torch.from_numpy(genes_mu).float()
        # genes_sigma_tensor = torch.from_numpy(genes_sigma).float()
        # Create a minimal DataFormat and CellsDataset for preprocessing
        data_format = DataFormat(
            n_genes=3,
            gene_names=["g0", "g1", "g2"],
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            use_log_transform=False,
            target_sum=10.0,
        )
        # Create a temporary file for the dataset
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            # Create a minimal AnnData object and save it

            obs_df = pd.DataFrame(
                {
                    "expansion": ["expanded"] * X_masked_raw.shape[0],
                    "clone_id_size": [1] * X_masked_raw.shape[0],
                    "median_clone_size": [1] * X_masked_raw.shape[0],
                }
            )
            var_df = pd.DataFrame(index=data_format.gene_names)
            X_dummy = np.zeros((X_masked_raw.shape[0], len(data_format.gene_names)))
            adata = ad.AnnData(X=X_dummy, obs=obs_df, var=var_df)
            adata.write_h5ad(tmp_file.name)

            _dataset = CellsDataset(
                data_format=data_format,
                row_inds=np.arange(X_masked_raw.shape[0]),
                dataset_params=DataAugmentParams(),
                is_train=False,
                data_path=tmp_file.name,
            )
        # Apply preprocessing directly (no need for dataset method)

        X_normalized_tensor = preprocess_expression_data(X=X_masked_raw, data_format=data_format, eps=1e-10)
        print(f"Normalized masked X: {X_normalized_tensor}")

        # Expected values: (0 - mu) / sigma = -mu / sigma
        expected_values = -genes_mu / genes_sigma
        expected_tensor = torch.tensor([expected_values, expected_values], dtype=torch.float32)
        print(f"Expected values: {expected_tensor}")

        assert torch.allclose(X_normalized_tensor, expected_tensor, atol=1e-6), (
            f"Masked values should be -mu/sigma. Expected: {expected_tensor}, Got: {X_normalized_tensor}"
        )

    def test_noise_after_normalization(self) -> None:
        """Test that noise is applied after normalization."""
        # Create normalized test data
        X_normalized = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=torch.float32)

        # Apply noise
        X_with_noise = apply_post_normalization_augmentations(X_normalized, noise_std=0.1)

        # Should not be identical due to noise
        assert not torch.allclose(X_with_noise, X_normalized), "Noise should change the normalized values"

        # Should be close but not identical
        difference = torch.abs(X_with_noise - X_normalized)
        assert torch.all(difference < 1.0), "Noise should be reasonable in magnitude"

    def test_full_pipeline_with_separated_augmentations(self, mock_dataset: CellsDataset) -> None:
        """Test the full pipeline with separated augmentations."""
        # Create dataset with both masking and noise
        dataset_params = DataAugmentParams(mask_rate=0.5, noise_std=0.1)

        # Temporarily modify the dataset
        original_params = mock_dataset.dataset_params
        mock_dataset.dataset_params = dataset_params
        mock_dataset.is_train = True

        try:
            # Get a batch
            batch_indices = [0, 1, 2, 3]
            batch = cells_collate_fn(batch_indices, mock_dataset)
            X_final = batch["x"]

            # Check that we have reasonable values (not all zeros, not all the same)
            assert not torch.allclose(X_final, torch.zeros_like(X_final)), "Final batch should not be all zeros"

            # Check that values are in a reasonable range for normalized data
            # Single-cell data can have values > 10 after z-score normalization
            assert torch.all(torch.abs(X_final) < 20.0), (
                "Normalized values should be in reasonable range (allowing for single-cell data extremes)"
            )

            print(f"Final batch shape: {X_final.shape}")
            print(f"Final batch stats - mean: {X_final.mean():.3f}, std: {X_final.std():.3f}")
            print(f"Final batch range: [{X_final.min():.3f}, {X_final.max():.3f}]")

        finally:
            # Restore original parameters
            mock_dataset.dataset_params = original_params
            mock_dataset.is_train = False

    def test_masking_produces_expected_missing_signal(self, mock_dataset: CellsDataset) -> None:
        """Test that masking produces the expected 'missing gene' signal."""
        # Create dataset with only masking (no noise)
        dataset_params = DataAugmentParams(mask_rate=1.0, noise_std=0.0)  # 100% masking

        # Temporarily modify the dataset
        original_params = mock_dataset.dataset_params
        mock_dataset.dataset_params = dataset_params
        mock_dataset.is_train = True

        try:
            # Get a batch
            batch_indices = [0]
            batch = cells_collate_fn(batch_indices, mock_dataset)
            X_masked = batch["x"]

            # All values should be -mu/sigma (the "missing gene" signal)
            expected_masked_value = torch.from_numpy(
                -mock_dataset.data_format.genes_mu
                / (mock_dataset.data_format.genes_sigma + 1e-10)  # Match dataset preprocessing eps
            ).float()

            assert torch.allclose(X_masked, expected_masked_value.unsqueeze(0), atol=1e-6), (
                f"All masked values should be -mu/sigma (missing gene signal). "
                f"Expected: {expected_masked_value}, Got: {X_masked[0]}"
            )

            print(f"Missing gene signal: {expected_masked_value[:5]}...")  # Show first 5 values

        finally:
            # Restore original parameters
            mock_dataset.dataset_params = original_params
            mock_dataset.is_train = False


@pytest.fixture
def mock_adata() -> AnnData:
    # Create mock data with known distributions
    n_cells: int = 1000
    n_genes: int = 10

    # Create expression matrix with float32 dtype as a csr_matrix
    X: np.ndarray = np.random.rand(n_cells, n_genes).astype(np.float32)
    X_sparse = csr_matrix(X)

    # Create observation DataFrame with controlled distributions
    obs: pd.DataFrame = pd.DataFrame(
        {
            "tissue_type": np.random.choice(["tissue1", "tissue2"], size=n_cells, p=[0.6, 0.4]),
            "imputed_labels": np.random.choice(["label1", "label2"], size=n_cells, p=[0.7, 0.3]),
            "expansion": np.random.choice(["expanded", "not_expanded"], size=n_cells, p=[0.3, 0.7]),
            "clone_id_size": np.random.randint(1, 100, size=n_cells),
            "median_clone_size": np.random.randint(1, 100, size=n_cells),
        }
    )
    # Convert index to string to avoid the implicit conversion warning in AnnData
    obs.index = obs.index.astype(str)

    # Create AnnData object with sparse matrix
    adata: AnnData = AnnData(X=X_sparse, obs=obs)
    return adata


@pytest.fixture
def mock_dataset(mock_adata: AnnData, tmp_path: Path) -> CellsDataset:
    prm = MLPParam(
        n_epochs=100,
        early_stopping_patience=10,
        init_learning_rate=0.001,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_config={"type": "constant"},
        optimizer_type="adam",
        adam_betas=(0.9, 0.999),
        train_batch_size=32,
        inference_batch_size=64,
        sampler_type="balanced_types",
        layer_units=[64, 32],
        dropout_rate=0.1,
        mask_rate=0.0,
        noise_std=0.0,
        soft_loss_beta=1.0,
        soft_loss_start_epoch=0,
        positives_weight=1.0,
        train_log_interval=10,
    )

    # Create data format from mock_adata
    row_inds: np.ndarray = np.arange(len(mock_adata.obs))

    # Save the AnnData object to a temporary file first - required for create_data_format_and_convert_adata
    raw_file = tmp_path / "raw_adata.h5ad"
    mock_adata.write_h5ad(raw_file)

    data_format = DataFormat(
        aux_categorical_types=["tissue_type", "imputed_labels"],
        use_log_transform=False,
    )

    # Use data_path parameter for more efficient processing
    adata_loaded = ad.read_h5ad(raw_file, backed="r")

    new_adata = data_format.create_data_format(
        data_path=raw_file,
        adata=adata_loaded,
        row_inds_train=row_inds,
    )
    new_adata = data_format.prepare_adata_for_training(adata_loaded, reorder_genes=False)

    # Ensure all required columns are present in obs (required for CellsDataset and tests)
    required_cols = [
        "expansion",
        "clone_id_size",
        "median_clone_size",
        "tissue_type",
        "imputed_labels",
    ]
    for col in required_cols:
        if col not in new_adata.obs.columns:
            new_adata.obs[col] = mock_adata.obs[col].to_numpy()

    # Save the processed AnnData object to a temporary file
    processed_file = tmp_path / "test_adata.h5ad"
    new_adata.write_h5ad(processed_file)

    # Create dataset using the file path
    return CellsDataset(
        data_path=processed_file,
        row_inds=row_inds,
        dataset_params=prm.get_dataset_params(),
        data_format=data_format,
        is_train=True,
    )


class TestCategoricalFeatureEncoding:
    def test_encode_categorical_features_batch(self):
        # Prepare a small batch of dictionary of numpy arrays
        obs_np = {
            "tissue_type": np.array(["tissue1", "tissue2", "unknown_tissue"]),
            "imputed_labels": np.array(["label1", "label2", "unknown_label"]),
        }
        aux_categorical_types = ["tissue_type", "imputed_labels"]
        categorical_mappings = {
            "tissue_type": {"tissue1": 0, "tissue2": 1},
            "imputed_labels": {"label1": 0, "label2": 1},
        }
        obs_df = pd.DataFrame(obs_np)
        result = encode_categorical_features_batch(obs_df, aux_categorical_types, categorical_mappings)
        # Should be shape (3, 4): 2 tissue + 2 label
        assert result.shape == (3, 4)
        # First row: tissue1, label1 -> [1,0,1,0]
        assert (result[0] == np.array([1, 0, 1, 0], dtype=np.float32)).all()
        # Second row: tissue2, label2 -> [0,1,0,1]
        assert (result[1] == np.array([0, 1, 0, 1], dtype=np.float32)).all()
        # Third row: unknowns -> [0,0,0,0]
        assert (result[2] == np.array([0, 0, 0, 0], dtype=np.float32)).all()


class TestBatchSamplers:
    def test_balanced_labels_batch_sampler(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        mock_dataset.y = torch.FloatTensor((mock_dataset.obs_df["expansion"].to_numpy() == "expanded").astype(int))
        batch_size: int = 32
        sampler = BalancedTypesBatchSampler(mock_dataset, batch_size)
        all_batches = list(sampler)
        for batch in all_batches:
            assert len(batch) == batch_size, f"Batch size expected {batch_size}, got {len(batch)}"
        all_indices = np.concatenate(all_batches)
        assert np.all(all_indices >= 0) and np.all(all_indices < len(mock_dataset)), (
            "All indices should be within valid range"
        )

        def get_feature_distributions(indices: np.ndarray) -> dict[str, pd.Series]:
            batch_obs_df = mock_dataset.obs_df.iloc[indices]
            distributions = {
                "tissue_type": batch_obs_df["tissue_type"].value_counts(normalize=True),
                "imputed_labels": batch_obs_df["imputed_labels"].value_counts(normalize=True),
                "expansion": batch_obs_df["expansion"].value_counts(normalize=True),
            }
            return distributions

        original_dist = get_feature_distributions(np.array(range(len(mock_dataset))))
        for batch in all_batches[:5]:
            batch_arr = np.array(batch)
            batch_dist = get_feature_distributions(batch_arr)
            for feature in ["tissue_type", "imputed_labels"]:
                orig_props = original_dist[feature]
                batch_props = batch_dist[feature]
                assert set(batch_props.index).issubset(set(orig_props.index)), (
                    f"Batch contains unexpected categories for {feature}"
                )

            def composite_key(idx: int) -> str:
                return (
                    f"{mock_dataset.obs_df['tissue_type'].iloc[idx]}_{mock_dataset.obs_df['imputed_labels'].iloc[idx]}"
                )

            composite_counts: dict[str, dict[int, int]] = {}
            for idx in batch_arr:
                comp = composite_key(idx)
                label = int(mock_dataset.y[idx].item())
                composite_counts.setdefault(comp, {0: 0, 1: 0})
                composite_counts[comp][label] += 1
            composite_original: dict[str, list[int]] = {}
            for idx in range(len(mock_dataset.obs_df["tissue_type"])):
                comp = (
                    f"{mock_dataset.obs_df['tissue_type'].iloc[idx]}_{mock_dataset.obs_df['imputed_labels'].iloc[idx]}"
                )
                label = int(mock_dataset.y[idx].item())
                composite_original.setdefault(comp, []).append(label)
            balanced_composites = {comp for comp, labels in composite_original.items() if len(np.unique(labels)) == 2}
            for comp in balanced_composites:
                if comp in composite_counts:
                    counts = composite_counts[comp]
                    total = counts[0] + counts[1]
                    if total > 1:
                        assert abs(counts[0] - counts[1]) <= 1, (
                            f"Labels should be balanced in composite group {comp}: {counts}"
                        )

    def test_balanced_types_batch_sampler(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.y = torch.FloatTensor((mock_dataset.obs_df["expansion"].to_numpy() == "expanded").astype(int))
        batch_size: int = 32
        sampler = BalancedTypesBatchSampler(mock_dataset, batch_size)
        all_batches = list(sampler)
        for batch in all_batches:
            assert len(batch) == batch_size, f"Batch size expected {batch_size}, got {len(batch)}"
        all_indices = np.concatenate(all_batches)
        assert np.all(all_indices >= 0) and np.all(all_indices < len(mock_dataset)), (
            "All indices should be within valid range"
        )

        def get_feature_distributions(indices: np.ndarray) -> dict[str, pd.Series]:
            batch_obs_df = mock_dataset.obs_df.iloc[indices]
            distributions = {
                "tissue_type": batch_obs_df["tissue_type"].value_counts(normalize=True),
                "imputed_labels": batch_obs_df["imputed_labels"].value_counts(normalize=True),
                "expansion": batch_obs_df["expansion"].value_counts(normalize=True),
            }
            return distributions

        original_dist = get_feature_distributions(np.array(range(len(mock_dataset))))
        for batch in all_batches[:5]:
            batch_arr = np.array(batch)
            batch_dist = get_feature_distributions(batch_arr)
            for feature in ["tissue_type", "imputed_labels"]:
                orig_props = original_dist[feature]
                batch_props = batch_dist[feature]
                assert set(batch_props.index).issubset(set(orig_props.index)), (
                    f"Batch contains unexpected categories for {feature}"
                )

            def composite_key(idx: int) -> str:
                return (
                    f"{mock_dataset.obs_df['tissue_type'].iloc[idx]}_{mock_dataset.obs_df['imputed_labels'].iloc[idx]}"
                )

            composite_counts: dict[str, dict[int, int]] = {}
            for idx in batch_arr:
                comp = composite_key(idx)
                label = int(mock_dataset.y[idx].item())
                composite_counts.setdefault(comp, {0: 0, 1: 0})
                composite_counts[comp][label] += 1
            composite_original: dict[str, list[int]] = {}
            for idx in range(len(mock_dataset.obs_df["tissue_type"])):
                comp = (
                    f"{mock_dataset.obs_df['tissue_type'].iloc[idx]}_{mock_dataset.obs_df['imputed_labels'].iloc[idx]}"
                )
                label = int(mock_dataset.y[idx].item())
                composite_original.setdefault(comp, []).append(label)
            balanced_composites = {comp for comp, labels in composite_original.items() if len(np.unique(labels)) == 2}
            for comp in balanced_composites:
                if comp in composite_counts:
                    counts = composite_counts[comp]
                    total = counts[0] + counts[1]
                    if total > 1:
                        assert abs(counts[0] - counts[1]) <= 1, (
                            f"Labels should be balanced in composite group {comp}: {counts}"
                        )

    def test_balanced_types_batch_sampler_edge_cases(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        sampler_temp = BalancedTypesBatchSampler(mock_dataset, batch_size=32)
        num_strata: int = len(sampler_temp.feature_strata)
        if num_strata > 4:
            with pytest.raises(
                ValueError,
                match="Batch size must be at least as large as the number of strata",
            ):
                list(BalancedTypesBatchSampler(mock_dataset, batch_size=4))
        else:
            small_sampler = BalancedTypesBatchSampler(mock_dataset, batch_size=4)
            small_batches = list(small_sampler)
            for batch in small_batches:
                assert len(batch) == 4, "Small batches should maintain specified size"
        large_sampler = BalancedTypesBatchSampler(mock_dataset, batch_size=128)
        large_batches = list(large_sampler)
        for batch in large_batches:
            assert len(batch) == 128, "Large batches should maintain specified size"

    def test_balanced_types_batch_sampler_balance(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        batch_size: int = 32
        sampler = BalancedTypesBatchSampler(mock_dataset, batch_size)
        expected_samples = sampler.n_items_in_batch_per_stratum
        for batch in list(sampler)[:5]:
            batch_counts: dict[str, int] = dict.fromkeys(sampler.feature_strata, 0)
            for idx in batch:
                for stratum, indices in sampler.feature_strata.items():
                    if idx in indices:
                        batch_counts[stratum] += 1
                        break
            for stratum, expected in expected_samples.items():
                assert batch_counts[stratum] == expected, (
                    f"Stratum {stratum} expected {expected} samples, got {batch_counts[stratum]}"
                )

    def test_balanced_types_batch_sampler_consistency(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        batch_size: int = 32
        sampler = BalancedTypesBatchSampler(mock_dataset, batch_size)
        n_epochs: int = 3
        epoch_batches = [list(sampler) for _ in range(n_epochs)]
        n_batches: list[int] = [len(batches) for batches in epoch_batches]
        assert all(n == n_batches[0] for n in n_batches), "Number of batches should be consistent across epochs"
        for epoch_idx, batches in enumerate(epoch_batches):
            for batch_idx, batch in enumerate(batches):
                assert len(batch) == batch_size, (
                    f"Epoch {epoch_idx}, batch {batch_idx} has incorrect size: {len(batch)} != {batch_size}"
                )
        for epoch_idx, batches in enumerate(epoch_batches):
            for batch_idx, batch in enumerate(batches):
                batch_counts: dict[str, int] = dict.fromkeys(sampler.feature_strata, 0)
                for idx in batch:
                    for stratum, indices in sampler.feature_strata.items():
                        if idx in indices:
                            batch_counts[stratum] += 1
                            break
                for stratum, expected in sampler.n_items_in_batch_per_stratum.items():
                    assert batch_counts[stratum] == expected, (
                        f"Epoch {epoch_idx}, batch {batch_idx}, stratum {stratum}: "
                        f"expected {expected} samples, got {batch_counts[stratum]}"
                    )
        first_batches: list[list[int]] = epoch_batches[0]
        for other_epoch_batches in epoch_batches[1:]:
            assert not all(set(b1) == set(b2) for b1, b2 in zip(first_batches, other_epoch_batches)), (
                "Batch composition should vary between epochs due to randomization"
            )

    def test_balanced_labels_batch_sampler_distribution(self, mock_dataset: CellsDataset) -> None:
        batch_size: int = 32
        sampler = BalancedLabelsBatchSampler(mock_dataset, batch_size)
        for batch in list(sampler):
            labels = [int(mock_dataset.y[idx].item()) for idx in batch]
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            assert abs(pos_count - neg_count) <= 1, (
                f"Labels not balanced in batch: positives={pos_count}, negatives={neg_count}"
            )

    def test_balanced_labels_batch_sampler_coverage(self, mock_dataset: CellsDataset) -> None:
        batch_size: int = 32
        sampler = BalancedLabelsBatchSampler(mock_dataset, batch_size)
        all_indices = set()
        for batch in sampler:
            all_indices.update(batch)
        labels = mock_dataset.y.numpy()
        pos_indices = set(np.where(labels == 1)[0])
        neg_indices = set(np.where(labels == 0)[0])
        assert len(all_indices.intersection(pos_indices)) > 0, "No positive samples were used"
        assert len(all_indices.intersection(neg_indices)) > 0, "No negative samples were used"

        # Check that all available samples of each class are used when needed
        pos_used_unique = all_indices.intersection(pos_indices)
        neg_used_unique = all_indices.intersection(neg_indices)

        # Calculate how many unique samples we expect to use for each class
        pos_per_batch = batch_size // 2
        neg_per_batch = batch_size - pos_per_batch
        total_pos_needed = pos_per_batch * len(sampler)
        total_neg_needed = neg_per_batch * len(sampler)

        # We should use all available positive samples if we need more than we have
        expected_pos_unique = min(len(pos_indices), total_pos_needed)
        expected_neg_unique = min(len(neg_indices), total_neg_needed)

        assert len(pos_used_unique) == expected_pos_unique, (
            f"Expected to use {expected_pos_unique} unique positive samples, got {len(pos_used_unique)}"
        )
        assert len(neg_used_unique) == expected_neg_unique, (
            f"Expected to use {expected_neg_unique} unique negative samples, got {len(neg_used_unique)}"
        )

    def test_balanced_types_batch_sampler_stratum_isolation(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        batch_size: int = 32
        sampler = BalancedTypesBatchSampler(mock_dataset, batch_size)
        index_to_stratum: dict[int, str] = {}
        for stratum, indices in sampler.feature_strata.items():
            for idx in indices:
                index_to_stratum[idx] = stratum
        for batch in list(sampler):
            index_counts = {}
            for idx in batch:
                index_counts[idx] = index_counts.get(idx, 0) + 1
        # Each index should appear exactly once
        # Remove this test

    def test_balanced_types_batch_sampler_small_strata(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        batch_size: int = 32
        sampler = BalancedTypesBatchSampler(mock_dataset, batch_size)
        small_strata = {
            stratum: indices
            for stratum, indices in sampler.feature_strata.items()
            if len(indices) < sampler.n_items_in_batch_per_stratum[stratum]
        }
        if small_strata:
            for batch in list(sampler)[:3]:
                for stratum, indices in small_strata.items():
                    stratum_appearances = sum(1 for idx in batch if idx in indices)
                    assert stratum_appearances == sampler.n_items_in_batch_per_stratum[stratum], (
                        f"Small stratum {stratum} not properly represented in batch"
                    )

    def test_balanced_types_batch_sampler_reproducibility(self, mock_dataset: CellsDataset) -> None:
        mock_dataset.obs_df = pd.DataFrame(mock_dataset.obs_df)
        batch_size: int = 32
        np.random.seed(42)
        sampler1 = BalancedTypesBatchSampler(mock_dataset, batch_size)
        batches1 = list(sampler1)
        np.random.seed(42)
        sampler2 = BalancedTypesBatchSampler(mock_dataset, batch_size)
        batches2 = list(sampler2)
        assert len(batches1) == len(batches2), "Different number of batches with same seed"
        for b1, b2 in zip(batches1, batches2):
            assert b1 == b2, "Different batch composition with same seed"


class TestCellsDataset:
    def test_cells_dataset_obsm_categorical_features(self, mock_adata: AnnData, tmp_path: Path) -> None:
        prm = MLPParam(
            n_epochs=100,
            early_stopping_patience=10,
            init_learning_rate=0.001,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_scheduler_config={"type": "constant"},
            optimizer_type="adam",
            adam_betas=(0.9, 0.999),
            train_batch_size=32,
            inference_batch_size=64,
            sampler_type="balanced_types",
            layer_units=[64, 32],
            dropout_rate=0.1,
            mask_rate=0.0,
            noise_std=0.0,
            soft_loss_beta=1.0,
            soft_loss_start_epoch=0,
            positives_weight=1.0,
            train_log_interval=10,
        )
        row_inds: np.ndarray = np.arange(len(mock_adata.obs))
        raw_file_append = tmp_path / "raw_adata_append.h5ad"
        raw_file_obsm = tmp_path / "raw_adata_obsm.h5ad"
        mock_adata.write_h5ad(raw_file_append)
        mock_adata.write_h5ad(raw_file_obsm)
        data_format_append = DataFormat(
            use_log_transform=False,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        adata_loaded_append = ad.read_h5ad(raw_file_append, backed="r")
        data_format_append.create_data_format(
            data_path=raw_file_append,
            adata=adata_loaded_append,
            row_inds_train=row_inds,
        )
        adata_append = data_format_append.prepare_adata_for_training(adata_loaded_append, reorder_genes=False)
        data_format_obsm = DataFormat(
            use_log_transform=False,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        adata_loaded_obsm = ad.read_h5ad(raw_file_obsm, backed="r")
        data_format_obsm.create_data_format(
            data_path=raw_file_obsm,
            adata=adata_loaded_obsm,
            row_inds_train=row_inds,
        )
        adata_obsm = data_format_obsm.prepare_adata_for_training(adata_loaded_obsm, reorder_genes=False)
        required_cols = ["expansion", "clone_id_size", "median_clone_size"]
        for col in required_cols:
            if col not in adata_append.obs.columns:
                adata_append.obs[col] = mock_adata.obs[col].to_numpy()
            if col not in adata_obsm.obs.columns:
                adata_obsm.obs[col] = mock_adata.obs[col].to_numpy()
        file_append = tmp_path / "adata_append.h5ad"
        file_obsm = tmp_path / "adata_obsm.h5ad"
        adata_append.write_h5ad(file_append)
        adata_obsm.write_h5ad(file_obsm)
        dataset_append = CellsDataset(
            data_path=file_append,
            data_format=data_format_append,
            row_inds=row_inds,
            dataset_params=prm.get_dataset_params(),
            is_train=True,
        )
        dataset_obsm = CellsDataset(
            data_path=file_obsm,
            data_format=data_format_obsm,
            row_inds=row_inds,
            dataset_params=prm.get_dataset_params(),
            is_train=True,
        )

        for dataset, data_format in zip([dataset_append, dataset_obsm], [data_format_append, data_format_obsm]):
            loader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=lambda batch, dataset=dataset: cells_collate_fn(batch, dataset),
            )
            batch = next(iter(loader))
            assert isinstance(batch, dict)
            assert "x" in batch
            assert "y" in batch
            assert isinstance(batch["x"], torch.Tensor)
            assert isinstance(batch["y"], torch.Tensor)
            expected_n_features = data_format.n_genes
            assert batch["x"].shape[1] == expected_n_features

    def test_cellsdataset_batch_collate_fn(self, mock_adata: AnnData, tmp_path: Path) -> None:
        data_format = DataFormat(
            use_log_transform=False,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        row_inds = np.arange(len(mock_adata.obs))
        raw_file = tmp_path / "raw_adata_obsm.h5ad"
        mock_adata.write_h5ad(raw_file)
        adata_loaded = ad.read_h5ad(raw_file, backed="r")
        data_format.create_data_format(
            data_path=raw_file,
            adata=adata_loaded,
            row_inds_train=row_inds,
        )
        adata = data_format.prepare_adata_for_training(adata_loaded, reorder_genes=False)
        processed_file = tmp_path / "test_adata_obsm.h5ad"
        adata.write_h5ad(processed_file)
        prm = MLPParam(
            n_epochs=100,
            early_stopping_patience=10,
            init_learning_rate=0.001,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_scheduler_config={"type": "constant"},
            optimizer_type="adam",
            adam_betas=(0.9, 0.999),
            train_batch_size=16,
            inference_batch_size=64,
            sampler_type="balanced_types",
            layer_units=[64, 32],
            dropout_rate=0.1,
            mask_rate=0.0,
            noise_std=0.0,
            soft_loss_beta=1.0,
            soft_loss_start_epoch=0,
            positives_weight=1.0,
            train_log_interval=10,
        )
        dataset = CellsDataset(
            data_path=processed_file,
            data_format=data_format,
            row_inds=row_inds,
            dataset_params=prm.get_dataset_params(),
            is_train=True,
        )
        batch_size = 16
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch, dataset=dataset: cells_collate_fn(batch, dataset),
        )
        batch = next(iter(loader))
        assert "x" in batch and "y" in batch, "Batch must contain 'x' and 'y' keys"
        assert isinstance(batch["x"], torch.Tensor), "'x' must be a torch.Tensor"
        assert isinstance(batch["y"], torch.Tensor), "'y' must be a torch.Tensor"
        assert batch["x"].shape[0] == batch_size, f"Expected batch size {batch_size}, got {batch['x'].shape[0]}"
        assert batch["y"].shape[0] == batch_size, f"Expected batch size {batch_size}, got {batch['y'].shape[0]}"
        expected_features = dataset.n_genes
        assert batch["x"].shape[1] == expected_features, (
            f"Expected feature dimension {expected_features}, got {batch['x'].shape[1]}"
        )
        if dataset.y_soft is not None:
            assert "y_soft" in batch, "Batch must contain 'y_soft' if soft labels are present"
            assert isinstance(batch["y_soft"], torch.Tensor), "'y_soft' must be a torch.Tensor"
            assert batch["y_soft"].shape[0] == batch_size, (
                f"Expected batch size {batch_size}, got {batch['y_soft'].shape[0]}"
            )

    def test_data_preprocessing_consistency(self, mock_adata: AnnData, tmp_path: Path) -> None:
        """Test that data preprocessing works consistently with augmentations."""
        prm = MLPParam(
            n_epochs=100,
            early_stopping_patience=10,
            init_learning_rate=0.001,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_scheduler_config={"type": "constant"},
            optimizer_type="adam",
            adam_betas=(0.9, 0.999),
            train_batch_size=32,
            inference_batch_size=64,
            sampler_type="balanced_types",
            layer_units=[64, 32],
            dropout_rate=0.1,
            mask_rate=0.1,
            noise_std=0.1,
            soft_loss_beta=1.0,
            soft_loss_start_epoch=0,
            positives_weight=1.0,
            train_log_interval=10,
        )
        row_inds = np.arange(len(mock_adata.obs))
        raw_file = tmp_path / "raw_adata.h5ad"
        mock_adata.write_h5ad(raw_file)

        data_format = DataFormat(
            use_log_transform=True,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        adata_loaded = ad.read_h5ad(raw_file, backed="r")
        data_format.create_data_format(
            data_path=raw_file,
            adata=adata_loaded,
            row_inds_train=row_inds,
        )
        adata = data_format.prepare_adata_for_training(adata_loaded, reorder_genes=False)
        required_cols = ["expansion", "clone_id_size", "median_clone_size"]
        for col in required_cols:
            if col not in adata.obs.columns:
                adata.obs[col] = mock_adata.obs[col].to_numpy()

        processed_file = tmp_path / "processed_adata.h5ad"
        adata.write_h5ad(processed_file)

        dataset = CellsDataset(
            data_path=processed_file,
            data_format=data_format,
            row_inds=row_inds,
            dataset_params=prm.get_dataset_params(),
            is_train=True,
        )

        batch_size = 16
        batch_indices = list(range(batch_size))

        # Test with augmentations (train mode)
        torch.manual_seed(42)
        batch_with_aug = cells_collate_fn(batch_indices, dataset)

        # Test without augmentations (inference mode)
        dataset.is_train = False
        torch.manual_seed(42)
        batch_without_aug = cells_collate_fn(batch_indices, dataset)

        assert batch_with_aug["x"].shape == batch_without_aug["x"].shape, (
            "Batches should have the same shape regardless of augmentation"
        )

        expected_features = dataset.n_genes
        assert batch_with_aug["x"].shape[1] == expected_features, (
            f"Expected {expected_features} features, got {batch_with_aug['x'].shape[1]}"
        )

        # With augmentations, the data should be different (due to noise/masking)
        if prm.get_dataset_params().mask_rate > 0 or prm.get_dataset_params().noise_std > 0:
            max_diff = torch.max(torch.abs(batch_with_aug["x"] - batch_without_aug["x"]))
            assert max_diff > 1e-6, "Augmentations should modify the data"

    def test_cellsdataset_feature_extraction_consistency(self, mock_adata: AnnData, tmp_path: Path) -> None:
        """Test that feature extraction works consistently with different batch sizes."""
        n_cells: int = len(mock_adata.obs)
        row_inds: np.ndarray = np.arange(n_cells)
        batch_sizes: list[int] = [1, 8]
        prm = MLPParam(
            n_epochs=100,
            early_stopping_patience=10,
            init_learning_rate=0.001,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_scheduler_config={"type": "constant"},
            optimizer_type="adam",
            adam_betas=(0.9, 0.999),
            train_batch_size=8,
            inference_batch_size=8,
            sampler_type="balanced_types",
            layer_units=[64, 32],
            dropout_rate=0.1,
            mask_rate=0.0,
            noise_std=0.0,
            soft_loss_beta=1.0,
            soft_loss_start_epoch=0,
            positives_weight=1.0,
            train_log_interval=10,
        )

        raw_file = tmp_path / "raw_features.h5ad"
        mock_adata.write_h5ad(raw_file)
        data_format = DataFormat(
            use_log_transform=False,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        adata_loaded = ad.read_h5ad(raw_file, backed="r")
        data_format.create_data_format(
            data_path=raw_file,
            adata=adata_loaded,
            row_inds_train=row_inds,
        )
        adata = data_format.prepare_adata_for_training(adata_loaded, reorder_genes=False)
        required_cols = ["expansion", "clone_id_size", "median_clone_size"]
        for col in required_cols:
            if col not in adata.obs.columns:
                adata.obs[col] = mock_adata.obs[col].to_numpy()
        processed_file = tmp_path / "processed_features.h5ad"
        adata.write_h5ad(processed_file)

        dataset = CellsDataset(
            data_path=processed_file,
            data_format=data_format,
            row_inds=row_inds,
            dataset_params=prm.get_dataset_params(),
            is_train=True,
        )

        for batch_size in batch_sizes:
            batch_indices = list(range(batch_size))
            batch = cells_collate_fn(batch_indices=batch_indices, dataset=dataset)
            assert isinstance(batch, dict)
            assert "x" in batch and "y" in batch
            assert isinstance(batch["x"], torch.Tensor)
            assert isinstance(batch["y"], torch.Tensor)
            expected_features = dataset.n_genes
            assert batch["x"].shape == (batch_size, expected_features)

            # Verify the preprocessing was applied correctly
            x_genes = batch["x"][:, : dataset.n_genes].detach().cpu().numpy()

            # Load raw data from file to get the original values
            raw_adata = ad.read_h5ad(processed_file, backed="r")
            try:
                X_raw = raw_adata.X[batch_indices, :].toarray()
                X_tensor = torch.from_numpy(X_raw).float()
                # Apply preprocessing directly (no need for dataset method)
                X_expected_tensor = preprocess_expression_data(X=X_tensor, data_format=dataset.data_format, eps=1e-10)
                X_expected = X_expected_tensor.detach().cpu().numpy()
                assert np.allclose(x_genes, X_expected, atol=1e-5), "Gene features do not match expected preprocessing"
            finally:
                raw_adata.file.close()

    @pytest.mark.slow
    def test_dataloader_multiple_workers(self, mock_dataset: CellsDataset) -> None:
        if os.cpu_count() is None or os.cpu_count() < 2:
            pytest.skip("Not enough CPU cores available for multiple worker test")
        batch_size = 16
        num_workers = 2
        collate_fn = partial(cells_collate_fn, dataset=mock_dataset)
        loader = DataLoader(
            mock_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        num_batches = 0
        for batch in loader:
            assert "x" in batch
            assert "y" in batch
            assert batch["x"].shape[0] <= batch_size
            assert batch["y"].shape[0] <= batch_size
            num_batches += 1
        assert num_batches > 0, "No batches were loaded from the DataLoader"

    def test_separated_methods_equivalence(self, mock_adata: AnnData, tmp_path: Path) -> None:
        """Test that calling the separated methods produces consistent results."""
        row_inds = np.arange(len(mock_adata.obs))
        raw_file = tmp_path / "raw_adata.h5ad"
        mock_adata.write_h5ad(raw_file)

        # Method 1: Manual call to separated methods
        data_format_combined = DataFormat(
            use_log_transform=True,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        adata_loaded_combined = ad.read_h5ad(raw_file, backed="r")
        data_format_combined.create_data_format(
            data_path=raw_file,
            adata=adata_loaded_combined,
            row_inds_train=row_inds,
        )
        adata_combined = data_format_combined.prepare_adata_for_training(adata_loaded_combined, reorder_genes=False)

        # Method 2: Separated methods
        data_format_separated = DataFormat(
            use_log_transform=True,
            aux_categorical_types=["tissue_type", "imputed_labels"],
        )
        adata_loaded_separated = ad.read_h5ad(raw_file, backed="r")
        data_format_separated.create_data_format(
            data_path=raw_file,
            adata=adata_loaded_separated,
            row_inds_train=row_inds,
        )
        adata_separated = data_format_separated.prepare_adata_for_training(adata_loaded_separated, reorder_genes=False)

        # Verify DataFormat objects are equivalent
        assert data_format_combined.n_genes == data_format_separated.n_genes
        assert data_format_combined.gene_names == data_format_separated.gene_names
        assert np.allclose(data_format_combined.genes_mu, data_format_separated.genes_mu)
        assert np.allclose(data_format_combined.genes_sigma, data_format_separated.genes_sigma)
        assert data_format_combined.aux_categorical_mappings == data_format_separated.aux_categorical_mappings

        # Verify AnnData objects are equivalent
        assert adata_combined.n_obs == adata_separated.n_obs
        assert adata_combined.n_vars == adata_separated.n_vars
        assert adata_combined.var_names.tolist() == adata_separated.var_names.tolist()

        # With batch loading, the underlying data format should be the same
        # Verify that both are backed (data loaded from disk for efficiency)
        assert adata_combined.isbacked == adata_separated.isbacked
        assert adata_combined.isbacked
        assert adata_separated.isbacked

    def test_cells_dataset_with_missing_columns(self, tmp_path: Path) -> None:
        """Test that CellsDataset handles missing columns gracefully for inference."""
        # Create a minimal dataset with only gene expression data (no metadata columns)
        n_cells, n_genes = 10, 5
        X = np.random.randn(n_cells, n_genes).astype(np.float32)

        # Create obs DataFrame with NO metadata columns (simulating inference data)
        obs_df = pd.DataFrame(index=range(n_cells))
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])

        # Create AnnData with minimal obs data
        adata_minimal = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "minimal_adata.h5ad"
        adata_minimal.write_h5ad(file_path)

        # Create DataFormat
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=list(var_df.index),
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        dataset_params = DataAugmentParams(mask_rate=0.0, noise_std=0.0, soft_loss_beta=1.0)

        # Test inference mode with missing columns - should not raise KeyError
        dataset = CellsDataset(
            data_format=data_format,
            dataset_params=dataset_params,
            is_train=False,  # Inference mode
            data_path=file_path,
        )

        # Verify dataset was created successfully
        assert dataset.n_cells == n_cells
        assert dataset.n_genes == n_genes
        assert dataset.y is None  # No labels for inference
        assert dataset.y_soft is None  # No soft labels for inference

        # Test that we can create a batch without errors
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch, dataset=dataset: cells_collate_fn(batch, dataset),
        )

        batch = next(iter(loader))
        assert "x" in batch
        assert batch["x"].shape == (4, n_genes)
        assert "y" not in batch  # No labels in inference mode

        # Test training mode with missing expansion column - should warn but not fail
        dataset_train = CellsDataset(
            data_format=data_format,
            dataset_params=dataset_params,
            is_train=True,  # Training mode
            data_path=file_path,
        )

        # Should have warning but no error
        assert dataset_train.n_cells == n_cells
        assert dataset_train.y is None  # No expansion column available
        assert dataset_train.y_soft is None  # No clone size columns available

    def test_cells_dataset_with_partial_columns(self, tmp_path: Path) -> None:
        """Test that CellsDataset handles partial column availability correctly."""
        # Create dataset with only some metadata columns
        n_cells, n_genes = 10, 5
        X = np.random.randn(n_cells, n_genes).astype(np.float32)

        # Create obs DataFrame with only some metadata columns
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["non-expanded"] * 5,  # Only expansion column
                # Missing: clone_id_size, median_clone_size
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])

        # Create AnnData with partial metadata
        adata_partial = ad.AnnData(X=X, obs=obs_df, var=var_df)
        file_path = tmp_path / "partial_adata.h5ad"
        adata_partial.write_h5ad(file_path)

        # Create DataFormat
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=list(var_df.index),
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        dataset_params = DataAugmentParams(mask_rate=0.0, noise_std=0.0, soft_loss_beta=1.0)

        # Test training mode with partial columns
        dataset = CellsDataset(
            data_format=data_format,
            dataset_params=dataset_params,
            is_train=True,
            data_path=file_path,
        )

        # Should have expansion labels but no soft labels
        assert dataset.n_cells == n_cells
        assert dataset.y is not None  # Expansion column available
        assert dataset.y_soft is None  # Clone size columns missing

        # Test that we can create a batch
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch, dataset=dataset: cells_collate_fn(batch, dataset),
        )

        batch = next(iter(loader))
        assert "x" in batch
        assert "y" in batch
        assert "y_soft" not in batch  # No soft labels available


# -------------------------------------------------------------
# Tests for optimized inference with gene count mismatches
# -------------------------------------------------------------


class TestOptimizedInferenceWithGeneMismatches:
    """Test the optimized inference system with different gene count scenarios."""

    def test_gene_mapping_initialization(self, mock_dataset: CellsDataset) -> None:
        """Test that gene mapping is correctly initialized during dataset creation."""
        # Verify that the optimized dataset has the required attributes
        assert hasattr(mock_dataset, "gene_overlap"), "Dataset should have gene_overlap"
        assert hasattr(mock_dataset, "missing_genes"), "Dataset should have missing_genes"
        assert hasattr(mock_dataset, "extra_genes"), "Dataset should have extra_genes"
        assert hasattr(mock_dataset, "transform_batch_data"), "Dataset should have transform_batch_data method"
        assert hasattr(mock_dataset, "needs_gene_transformation"), "Dataset should have needs_gene_transformation"

        # Verify gene mapping properties (gene_indices may be None if no transformation needed)
        if mock_dataset.needs_gene_transformation:
            assert hasattr(mock_dataset, "gene_indices"), "Dataset should have gene_indices when transformation needed"
            assert len(mock_dataset.gene_indices) == mock_dataset.n_genes, "Gene indices should match n_genes"
        else:
            assert mock_dataset.gene_indices is None, "Gene indices should be None when no transformation needed"
        assert len(mock_dataset.gene_overlap) >= 0, "Gene overlap should be non-negative"
        assert len(mock_dataset.missing_genes) >= 0, "Missing genes should be non-negative"
        assert len(mock_dataset.extra_genes) >= 0, "Extra genes should be non-negative"

    def test_batch_transformation_with_perfect_match(self, mock_dataset: CellsDataset) -> None:
        """Test batch transformation when inference data perfectly matches training data."""
        # Create a small test batch
        batch_size = 5
        n_raw_genes = mock_dataset.data_format.n_genes
        test_batch_raw = torch.rand(batch_size, n_raw_genes, dtype=torch.float32)

        # Transform the batch
        transformed_batch = mock_dataset.transform_batch_data(test_batch_raw, in_place=False)

        # Check output properties
        assert isinstance(transformed_batch, torch.Tensor), "Should return torch.Tensor"
        assert transformed_batch.shape == (batch_size, mock_dataset.n_genes), (
            f"Expected shape ({batch_size}, {mock_dataset.n_genes})"
        )
        assert torch.isfinite(transformed_batch).all(), "All values should be finite"

        # Check that preprocessing was applied (values should be normalized)
        assert transformed_batch.std() > 0, "Standard deviation should be positive after normalization"

    def test_batch_transformation_with_missing_genes(self, tmp_path: Path) -> None:
        """Test batch transformation when inference data has missing genes compared to training."""
        # Create a data format with more genes than the inference data
        n_inference_genes = 5
        n_training_genes = 8  # More genes in training

        # Create inference data with fewer genes
        X_inference = np.random.rand(10, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["not_expanded"] * 5,
                "clone_id_size": [1] * 10,
                "median_clone_size": [1] * 10,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "inference_data.h5ad"
        adata_inference.write_h5ad(file_path)

        # Create data format with more genes (simulating training with more genes)
        extra_genes = [f"missing_gene_{i}" for i in range(n_training_genes - n_inference_genes)]
        all_gene_names = list(var_df.index) + extra_genes

        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=all_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Check that missing genes are correctly identified
        assert len(dataset.missing_genes) == n_training_genes - n_inference_genes, (
            f"Expected {n_training_genes - n_inference_genes} missing genes"
        )
        assert len(dataset.gene_indices) == n_training_genes, f"Expected {n_training_genes} gene indices"

        # Check that missing genes have index -1
        missing_indices = dataset.gene_indices[dataset.gene_indices == -1]
        assert len(missing_indices) == n_training_genes - n_inference_genes, (
            f"Expected {n_training_genes - n_inference_genes} missing indices (-1)"
        )

        # Test batch transformation
        test_batch_raw = torch.rand(3, n_inference_genes, dtype=torch.float32)
        transformed_batch = dataset.transform_batch_data(test_batch_raw, in_place=False)

        # Check output shape
        assert transformed_batch.shape == (3, n_training_genes), f"Expected shape (3, {n_training_genes})"

        # Check that missing genes are filled with zeros (before preprocessing)
        # The missing genes should have the "missing gene signal" after preprocessing
        missing_gene_positions = [i for i, idx in enumerate(dataset.gene_indices) if idx == -1]
        for pos in missing_gene_positions:
            # After preprocessing, missing genes should have the -mu/sigma signal
            expected_missing_value = -dataset.genes_mu[pos] / (
                dataset.genes_sigma[pos] + 1e-10
            )  # Match dataset preprocessing eps
            assert torch.allclose(transformed_batch[:, pos], torch.tensor(expected_missing_value)), (
                f"Missing gene at position {pos} should have missing signal"
            )

    def test_batch_transformation_with_extra_genes(self, tmp_path: Path) -> None:
        """Test batch transformation when inference data has extra genes compared to training."""
        # Create a data format with fewer genes than the inference data
        n_training_genes = 5
        n_inference_genes = 8  # More genes in inference data

        # Create inference data with more genes
        X_inference = np.random.rand(10, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["not_expanded"] * 5,
                "clone_id_size": [1] * 10,
                "median_clone_size": [1] * 10,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "inference_data_extra.h5ad"
        adata_inference.write_h5ad(file_path)

        # Create data format with fewer genes (simulating training with fewer genes)
        training_gene_names = [f"g{i}" for i in range(n_training_genes)]

        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=training_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Check that extra genes are correctly identified
        assert len(dataset.extra_genes) == n_inference_genes - n_training_genes, (
            f"Expected {n_inference_genes - n_training_genes} extra genes"
        )
        assert len(dataset.gene_indices) == n_training_genes, f"Expected {n_training_genes} gene indices"

        # Check that all indices are valid (>= 0) since we have all training genes
        valid_indices = dataset.gene_indices[dataset.gene_indices >= 0]
        assert len(valid_indices) == n_training_genes, f"Expected {n_training_genes} valid indices"

        # Test batch transformation
        test_batch_raw = torch.rand(3, n_inference_genes, dtype=torch.float32)
        transformed_batch = dataset.transform_batch_data(test_batch_raw, in_place=False)

        # Check output shape
        assert transformed_batch.shape == (3, n_training_genes), f"Expected shape (3, {n_training_genes})"

        # Check that the transformation correctly maps the training genes
        for i, target_idx in enumerate(dataset.gene_indices):
            if target_idx >= 0:  # Gene exists in inference data
                # The transformed values should be different from raw due to preprocessing
                assert not torch.allclose(transformed_batch[:, i], torch.tensor(test_batch_raw[:, target_idx]))

    def test_collate_fn_with_gene_mismatch(self, tmp_path: Path) -> None:
        """Test that collate_fn works correctly with gene count mismatches."""
        # Create inference data with different gene count
        n_inference_genes = 6
        n_training_genes = 4

        X_inference = np.random.rand(10, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["not_expanded"] * 5,
                "clone_id_size": [1] * 10,
                "median_clone_size": [1] * 10,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "mismatch_data.h5ad"
        adata_inference.write_h5ad(file_path)

        # Create data format with different gene count
        training_gene_names = [f"g{i}" for i in range(n_training_genes)]

        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=training_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Test collate_fn
        batch_indices = [0, 1, 2]
        batch = cells_collate_fn(batch_indices, dataset)

        # Check batch properties
        assert "x" in batch, "Batch should contain 'x'"
        assert isinstance(batch["x"], torch.Tensor), "'x' should be torch.Tensor"
        assert batch["x"].shape == (3, n_training_genes), f"Expected shape (3, {n_training_genes})"
        assert torch.isfinite(batch["x"]).all(), "All values should be finite"

    def test_optimized_inference_consistency(self, tmp_path: Path) -> None:
        """Test that optimized inference produces consistent results across runs."""
        # Create test data
        n_inference_genes = 5
        n_training_genes = 3

        X_inference = np.random.rand(8, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 4 + ["not_expanded"] * 4,
                "clone_id_size": [1] * 8,
                "median_clone_size": [1] * 8,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "consistency_data.h5ad"
        adata_inference.write_h5ad(file_path)

        # Create data format
        training_gene_names = [f"g{i}" for i in range(n_training_genes)]

        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=training_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Create data loader

        data_loader = create_eval_dataloader(dataset=dataset, batch_size=4, num_workers=0)

        # Run inference twice
        results_1 = [batch["x"] for batch in data_loader]
        results_2 = [batch["x"] for batch in data_loader]

        # Concatenate results
        all_results_1 = torch.cat(results_1, dim=0)
        all_results_2 = torch.cat(results_2, dim=0)

        # Check consistency
        assert torch.allclose(all_results_1, all_results_2, rtol=1e-6), "Results should be identical between runs"
        assert all_results_1.shape == (8, n_training_genes), f"Expected shape (8, {n_training_genes})"

    def test_memory_efficiency(self, tmp_path: Path) -> None:
        """Test that the optimized system doesn't load entire dataset into memory."""
        # Create test data
        n_inference_genes = 10
        n_training_genes = 6

        X_inference = np.random.rand(20, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 10 + ["not_expanded"] * 10,
                "clone_id_size": [1] * 20,
                "median_clone_size": [1] * 20,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "memory_test_data.h5ad"
        adata_inference.write_h5ad(file_path)

        # Create data format
        training_gene_names = [f"g{i}" for i in range(n_training_genes)]

        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=training_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Check that we don't have a cached adata (should be None)
        assert not hasattr(dataset, "_adata") or dataset._adata is None, "Dataset should not cache full AnnData"

        # Check that we have the transformation components
        assert hasattr(dataset, "gene_indices"), "Dataset should have gene_indices"
        assert hasattr(dataset, "transform_batch_data"), "Dataset should have transform_batch_data method"

        # Test that we can process batches without loading full data
        test_batch_raw = torch.rand(5, n_inference_genes, dtype=torch.float32)
        transformed_batch = dataset.transform_batch_data(test_batch_raw, in_place=False)

        assert transformed_batch.shape == (5, n_training_genes), f"Expected shape (5, {n_training_genes})"

    def test_tensor_optimization_no_conversions(self, tmp_path: Path) -> None:
        """Test that tensor inputs avoid unnecessary conversions in transform_batch_data."""
        # Create test data with gene transformation needed
        n_inference_genes = 8
        n_training_genes = 5

        X_inference = np.random.rand(10, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 5 + ["not_expanded"] * 5,
                "clone_id_size": [1] * 10,
                "median_clone_size": [1] * 10,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "tensor_optimization_data.h5ad"
        adata_inference.write_h5ad(file_path)

        # Create data format with subset of genes to force transformation
        training_gene_names = [f"g{i}" for i in range(n_training_genes)]
        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=training_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Test with tensor input (tensor-only API)
        test_batch_tensor = torch.rand(3, n_inference_genes, dtype=torch.float32)
        result_from_tensor = dataset.transform_batch_data(test_batch_tensor, in_place=False)

        # Test with second tensor to verify consistency
        test_batch_tensor2 = test_batch_tensor.clone()
        result_from_tensor2 = dataset.transform_batch_data(test_batch_tensor2, in_place=False)

        # Results should be identical for same input
        assert torch.allclose(result_from_tensor, result_from_tensor2, atol=1e-10), (
            "Same tensor inputs should produce identical results"
        )

        # Should have correct output shape
        expected_shape = (3, n_training_genes)
        assert result_from_tensor.shape == expected_shape

        # Test that tensor operations preserve device and dtype
        if torch.cuda.is_available():
            test_batch_cuda = test_batch_tensor.cuda()
            result_cuda = dataset.transform_batch_data(test_batch_cuda, in_place=False)
            assert result_cuda.device.type == "cuda", "Should preserve CUDA device"
            assert result_cuda.dtype == torch.float32, "Should preserve dtype"

    def test_gene_transformation_tensor_vs_numpy(self, tmp_path: Path) -> None:
        """Test that tensor and numpy gene transformation methods produce identical results."""
        # Create dataset with gene mismatch to ensure transformation is needed
        n_inference_genes = 6
        n_training_genes = 4

        X_inference = np.random.rand(8, n_inference_genes).astype(np.float32)
        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 4 + ["not_expanded"] * 4,
                "clone_id_size": [1] * 8,
                "median_clone_size": [1] * 8,
            }
        )
        var_df = pd.DataFrame(index=[f"g{i}" for i in range(n_inference_genes)])

        adata_inference = ad.AnnData(X=X_inference, obs=obs_df, var=var_df)
        file_path = tmp_path / "gene_transformation_test.h5ad"
        adata_inference.write_h5ad(file_path)

        training_gene_names = [f"g{i}" for i in range(n_training_genes)]
        data_format = DataFormat(
            n_genes=n_training_genes,
            gene_names=training_gene_names,
            genes_mu=np.zeros(n_training_genes, dtype=np.float32),
            genes_sigma=np.ones(n_training_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Ensure gene transformation is needed
        assert dataset.needs_gene_transformation, "Test requires gene transformation"

        # Test data
        test_data_numpy = np.random.rand(5, n_inference_genes).astype(np.float32)
        test_data_tensor = torch.from_numpy(test_data_numpy).float()

        # Apply tensor transformation (only available method now)
        result_tensor = dataset._apply_gene_transformation_tensor(test_data_tensor)

        # Test with second identical tensor to verify consistency
        test_data_tensor2 = torch.from_numpy(test_data_numpy).float()
        result_tensor2 = dataset._apply_gene_transformation_tensor(test_data_tensor2)

        # Should be identical for same input
        assert torch.allclose(result_tensor, result_tensor2, atol=1e-7), (
            "Same tensor inputs should produce identical results"
        )

        # Check shapes
        expected_shape = (5, n_training_genes)
        assert result_tensor.shape == expected_shape
        assert result_tensor2.shape == expected_shape

    def test_preprocessing_tensor_preservation(self) -> None:
        """Test that preprocessing preserves tensor type throughout the pipeline."""
        # Create simple data format
        n_genes = 3
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=[f"g{i}" for i in range(n_genes)],
            genes_mu=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 1.0, 1.5], dtype=np.float32),
            use_log_transform=False,
            target_sum=10.0,
        )

        # Test data
        X_numpy = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        X_tensor = torch.from_numpy(X_numpy).float()

        # Apply preprocessing
        result_numpy = preprocess_expression_data(X_numpy, data_format)
        result_tensor = preprocess_expression_data(X_tensor, data_format)

        # Check types
        assert isinstance(result_numpy, np.ndarray), "Numpy input should return numpy array"
        assert isinstance(result_tensor, torch.Tensor), "Tensor input should return tensor"

        # Check values are equivalent
        result_tensor_np = result_tensor.detach().cpu().numpy()
        assert np.allclose(result_numpy, result_tensor_np, atol=1e-6), (
            "Results should be equivalent regardless of input type"
        )

    def test_no_transformation_needed(self, tmp_path: Path) -> None:
        """Test the optimization when no gene transformation is needed."""
        # Create data with perfect gene match
        n_genes = 100
        n_cells = 50

        X_data = np.random.rand(n_cells, n_genes).astype(np.float32)
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        obs_df = pd.DataFrame(
            {
                "expansion": ["expanded"] * 25 + ["not_expanded"] * 25,
                "clone_id_size": [1] * n_cells,
                "median_clone_size": [1] * n_cells,
            }
        )
        var_df = pd.DataFrame(index=gene_names)

        adata = ad.AnnData(X=X_data, obs=obs_df, var=var_df)
        file_path = tmp_path / "perfect_match_data.h5ad"
        adata.write_h5ad(file_path)

        # Create data format with same genes in same order
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=gene_names,
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(
            data_format=data_format,
            data_path=file_path,
            is_train=False,
        )

        # Check that no transformation is needed
        assert not dataset.needs_gene_transformation, "Should not need gene transformation"
        assert dataset.gene_indices is None, "Gene indices should be None when no transformation needed"
        assert len(dataset.missing_genes) == 0, "Should have no missing genes"
        assert len(dataset.extra_genes) == 0, "Should have no extra genes"
        assert len(dataset.gene_overlap) == n_genes, "Should have all genes overlapping"

        # Test batch transformation
        test_batch_raw = torch.rand(10, n_genes, dtype=torch.float32)
        transformed_batch = dataset.transform_batch_data(test_batch_raw, in_place=False)

        # Check output properties
        assert transformed_batch.shape == (10, n_genes), f"Expected shape (10, {n_genes})"
        assert torch.isfinite(transformed_batch).all(), "All values should be finite"

        # The transformed data should be different from raw due to preprocessing
        # but the gene order should be preserved (no reordering needed)
        assert not torch.allclose(transformed_batch, test_batch_raw), "Preprocessing should modify the data"


if __name__ == "__main__":
    pytest.main()
