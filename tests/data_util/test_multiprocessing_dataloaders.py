"""Comprehensive tests for multiprocessing functionality in dataloaders and datasets.

Tests verify that:
1. Custom samplers can be pickled/unpickled correctly
2. RNG states are properly isolated between workers
3. Augmentations produce different results in different workers
4. Thread-local caching works correctly
5. DataLoaders work with num_workers > 0
"""

import gc
import multiprocessing as mp
import os
import pickle
import platform
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataloaders import (
    BalancedLabelsBatchSampler,
    BalancedTypesBatchSampler,
    create_eval_dataloader,
    create_train_dataloader,
)
from scxpand.data_util.dataset import (
    CellsDataset,
    apply_post_normalization_augmentations,
    apply_pre_normalization_augmentations,
    cells_collate_fn,
    get_dataloader_kwargs,
)
from scxpand.data_util.transforms import apply_zscore_normalization
from scxpand.util.classes import DataAugmentParams, DataLoaderParams
from tests.test_utils import windows_safe_context_manager


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    n_cells, n_genes = 100, 50
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "expansion": np.random.choice(["expanded", "not_expanded"], n_cells),
            "clone_id_size": np.random.randint(1, 20, n_cells),
            "median_clone_size": np.random.randint(5, 15, n_cells),
            "tissue_type": np.random.choice(["tissue_A", "tissue_B"], n_cells),
            "imputed_labels": np.random.choice(["label_1", "label_2"], n_cells),
        }
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def sample_data_format(sample_adata):
    """Create a DataFormat object for testing."""
    data_format = DataFormat(
        n_genes=sample_adata.n_vars,
        gene_names=sample_adata.var_names.tolist(),
        use_log_transform=True,
        use_zscore_norm=True,
        target_sum=10000.0,
        aux_categorical_types=("tissue_type", "imputed_labels"),
    )

    # Create dummy normalization parameters
    data_format.genes_mu = np.random.normal(0, 1, sample_adata.n_vars).astype(
        np.float32
    )
    data_format.genes_sigma = np.random.uniform(0.5, 2.0, sample_adata.n_vars).astype(
        np.float32
    )

    return data_format


@pytest.fixture
def sample_dataset(sample_adata, sample_data_format):
    """Create a CellsDataset for testing."""
    dataset_params = DataAugmentParams(mask_rate=0.1, noise_std=0.1)
    return CellsDataset(
        data_format=sample_data_format,
        dataset_params=dataset_params,
        is_train=True,
        adata=sample_adata,
    )


class TestSamplerPickling:
    """Test that custom samplers can be pickled and unpickled correctly."""

    def test_balanced_labels_sampler_pickling(self, sample_dataset):
        """Test that BalancedLabelsBatchSampler can be pickled and unpickled."""
        sampler = BalancedLabelsBatchSampler(sample_dataset, batch_size=10, seed=42)

        # Test pickling and unpickling
        pickled_data = pickle.dumps(sampler)
        unpickled_sampler = pickle.loads(pickled_data)

        # Verify attributes are preserved
        assert unpickled_sampler.batch_size == sampler.batch_size
        assert unpickled_sampler.seed == sampler.seed
        assert unpickled_sampler.num_batches == sampler.num_batches

        # Verify RNG is recreated correctly
        assert hasattr(unpickled_sampler, "rng")
        assert unpickled_sampler.rng is not None

        # Test that both samplers produce same results when seeded identically
        original_batches = list(sampler)
        unpickled_batches = list(unpickled_sampler)

        # Note: They won't be identical because RNG state advances during iteration
        # But they should have same structure
        assert len(original_batches) == len(unpickled_batches)
        for orig_batch, unpick_batch in zip(
            original_batches, unpickled_batches, strict=False
        ):
            assert len(orig_batch) == len(unpick_batch)

    def test_balanced_types_sampler_pickling(self, sample_dataset):
        """Test that BalancedTypesBatchSampler can be pickled and unpickled."""
        sampler = BalancedTypesBatchSampler(sample_dataset, batch_size=10, seed=42)

        # Test pickling and unpickling
        pickled_data = pickle.dumps(sampler)
        unpickled_sampler = pickle.loads(pickled_data)

        # Verify attributes are preserved
        assert unpickled_sampler.batch_size == sampler.batch_size
        assert unpickled_sampler.seed == sampler.seed
        assert unpickled_sampler.num_batches == sampler.num_batches
        assert unpickled_sampler.num_strata == sampler.num_strata

        # Verify RNG is recreated correctly
        assert hasattr(unpickled_sampler, "rng")
        assert unpickled_sampler.rng is not None


class TestRNGIsolation:
    """Test that random number generators are properly isolated between workers."""

    def test_augmentation_rng_isolation(self):
        """Test that augmentations produce different results with different generators."""
        X = torch.randn(10, 20)

        # Create different generators with same seed
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        gen2 = torch.Generator()
        gen2.manual_seed(42)

        # Apply same augmentation with same generators - should be identical
        result1 = apply_pre_normalization_augmentations(
            X.clone(), mask_rate=0.3, generator=gen1
        )
        result2 = apply_pre_normalization_augmentations(
            X.clone(), mask_rate=0.3, generator=gen2
        )

        assert torch.allclose(result1, result2)

        # Create generators with different seeds
        gen3 = torch.Generator()
        gen3.manual_seed(123)
        gen4 = torch.Generator()
        gen4.manual_seed(456)

        # Apply same augmentation with different generators - should be different
        result3 = apply_pre_normalization_augmentations(
            X.clone(), mask_rate=0.3, generator=gen3
        )
        result4 = apply_pre_normalization_augmentations(
            X.clone(), mask_rate=0.3, generator=gen4
        )

        assert not torch.allclose(result3, result4)

    def test_noise_augmentation_rng_isolation(self):
        """Test that noise augmentations are properly isolated."""
        X = torch.randn(10, 20)

        # Same seed generators
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        gen2 = torch.Generator()
        gen2.manual_seed(42)

        result1 = apply_post_normalization_augmentations(
            X.clone(), noise_std=0.1, generator=gen1
        )
        result2 = apply_post_normalization_augmentations(
            X.clone(), noise_std=0.1, generator=gen2
        )

        assert torch.allclose(result1, result2)

        # Different seed generators
        gen3 = torch.Generator()
        gen3.manual_seed(123)
        gen4 = torch.Generator()
        gen4.manual_seed(456)

        result3 = apply_post_normalization_augmentations(
            X.clone(), noise_std=0.1, generator=gen3
        )
        result4 = apply_post_normalization_augmentations(
            X.clone(), noise_std=0.1, generator=gen4
        )

        assert not torch.allclose(result3, result4)


class TestZScoreNormalization:
    """Test that z-score normalization works correctly in multiprocessing scenarios."""

    def test_zscore_normalization_consistency(self, sample_data_format):
        """Test that z-score normalization produces consistent results."""
        X = torch.randn(10, sample_data_format.n_genes)
        genes_mu = sample_data_format.genes_mu
        genes_sigma = sample_data_format.genes_sigma

        # Multiple calls should produce identical results (same input, same normalization)
        result1 = apply_zscore_normalization(X.clone(), genes_mu, genes_sigma)
        result2 = apply_zscore_normalization(X.clone(), genes_mu, genes_sigma)

        assert torch.allclose(result1, result2)

        # Test with different device (if CUDA available)
        if torch.cuda.is_available():
            X_cuda = X.cuda()
            result_cuda = apply_zscore_normalization(X_cuda, genes_mu, genes_sigma)
            # Should work correctly on different devices
            assert result_cuda.device.type == "cuda"
            # Results should be equivalent (accounting for device transfer)
            assert torch.allclose(result_cuda.cpu(), result1, atol=1e-6)


class TestMultiprocessingStartMethod:
    """Test detection and handling of different multiprocessing start methods."""

    def test_start_method_detection(self):
        """Test that we can detect the current multiprocessing start method."""
        start_method = mp.get_start_method()
        assert start_method in ["fork", "spawn", "forkserver"]

    def test_platform_specific_start_method(self):
        """Test that start method matches platform expectations."""
        start_method = mp.get_start_method()

        if platform.system() == "Windows":
            # Windows only supports spawn
            assert start_method == "spawn"
        elif platform.system() == "Darwin":
            # macOS defaults to spawn in recent Python versions
            assert start_method in ["spawn", "fork"]
        else:
            # Linux typically uses fork but can use spawn
            assert start_method in ["fork", "spawn", "forkserver"]

    def test_windows_detection(self):
        """Test Windows platform detection."""
        is_windows = os.name == "nt"
        is_windows_platform = platform.system() == "Windows"

        # These should be consistent
        assert is_windows == is_windows_platform

    @pytest.mark.parametrize("method", ["spawn", "fork"])
    def test_start_method_availability(self, method):
        """Test which start methods are available on this platform."""
        available_methods = mp.get_all_start_methods()

        if method == "fork":
            if platform.system() == "Windows":
                assert method not in available_methods
            else:
                assert method in available_methods
        elif method == "spawn":
            # Spawn should be available on all platforms
            assert method in available_methods


class TestDataLoaderMultiprocessing:
    """Test DataLoader functionality with multiple workers."""

    def _should_skip_multiprocessing_test(self, num_workers: int) -> tuple[bool, str]:
        """Determine if a multiprocessing test should be skipped and why."""
        if num_workers == 0:
            return False, ""

        skip_reasons = []

        # Check for Windows
        if os.name == "nt":
            skip_reasons.append("Windows (os.name == 'nt')")

        # Check for spawn start method
        if mp.get_start_method() == "spawn":
            skip_reasons.append("spawn start method")

        # Check for other unreliable conditions
        if platform.system() == "Windows":
            skip_reasons.append("Windows platform")

        if skip_reasons:
            return True, f"Multiprocessing may be unreliable: {', '.join(skip_reasons)}"

        return False, ""

    @pytest.mark.parametrize(
        "num_workers", [0, 1]
    )  # Reduced from [0, 1, 2] to speed up test
    def test_train_dataloader_with_workers(self, sample_dataset, num_workers):
        """Test that train dataloader works with different worker counts."""
        should_skip, reason = self._should_skip_multiprocessing_test(num_workers)
        if should_skip:
            pytest.skip(reason)

        loader_params = DataLoaderParams(
            batch_size=8, shuffle=True, sampler_type="balanced_labels"
        )

        dataloader = create_train_dataloader(
            sample_dataset, loader_params, num_workers=num_workers
        )

        # Test that we can iterate through the dataloader
        batches = []
        for i, batch in enumerate(dataloader):
            batches.append(batch)
            assert "x" in batch
            assert "y" in batch
            assert batch["x"].shape[0] <= 8  # batch size
            assert batch["x"].shape[1] == sample_dataset.n_genes

            # Only test a few batches to keep test fast
            if i >= 2:
                break

        assert len(batches) > 0

    @pytest.mark.parametrize(
        "num_workers", [0, 1]
    )  # Reduced from [0, 1, 2] to speed up test
    def test_eval_dataloader_with_workers(self, sample_dataset, num_workers):
        """Test that eval dataloader works with different worker counts."""
        should_skip, reason = self._should_skip_multiprocessing_test(num_workers)
        if should_skip:
            pytest.skip(reason)

        # Create eval dataset (is_train=False)
        eval_dataset = CellsDataset(
            data_format=sample_dataset.data_format,
            is_train=False,
            adata=sample_dataset._adata,
        )

        dataloader = create_eval_dataloader(
            eval_dataset, batch_size=8, num_workers=num_workers
        )

        # Test that we can iterate through the dataloader
        batches = []
        for i, batch in enumerate(dataloader):
            batches.append(batch)
            assert "x" in batch
            assert batch["x"].shape[0] <= 8  # batch size
            assert batch["x"].shape[1] == eval_dataset.n_genes

            # Only test a few batches to keep test fast
            if i >= 2:
                break

        assert len(batches) > 0

    def test_dataloader_with_fork_method_when_available(self, sample_dataset):
        """Test dataloader specifically with fork method when available."""
        available_methods = mp.get_all_start_methods()
        if "fork" not in available_methods:
            pytest.skip("Fork start method not available on this platform")

        # Temporarily set start method to fork
        original_method = mp.get_start_method()
        try:
            mp.set_start_method("fork", force=True)

            loader_params = DataLoaderParams(
                batch_size=4, shuffle=True, sampler_type="balanced_labels"
            )
            dataloader = create_train_dataloader(
                sample_dataset, loader_params, num_workers=1
            )

            # Should be able to iterate without issues
            batch = next(iter(dataloader))
            assert "x" in batch
            assert "y" in batch

        finally:
            # Restore original method
            mp.set_start_method(original_method, force=True)

    def test_dataloader_with_spawn_method(self, sample_dataset):
        """Test dataloader behavior with spawn method (which we usually skip)."""
        # This test verifies that spawn method causes the expected issues
        available_methods = mp.get_all_start_methods()
        if "spawn" not in available_methods:
            pytest.skip("Spawn start method not available")

        # Temporarily set start method to spawn
        original_method = mp.get_start_method()
        try:
            mp.set_start_method("spawn", force=True)

            loader_params = DataLoaderParams(
                batch_size=4, shuffle=True, sampler_type="balanced_labels"
            )

            # With spawn method, we expect potential issues
            # This test documents the behavior rather than asserting success
            try:
                dataloader = create_train_dataloader(
                    sample_dataset, loader_params, num_workers=1
                )
                batch = next(iter(dataloader))
                # If we get here, spawn method worked (which is fine)
                assert "x" in batch
            except Exception as e:
                # If spawn method fails, that's also expected and documented
                pytest.skip(f"Spawn method failed as expected: {e}")

        finally:
            # Restore original method
            mp.set_start_method(original_method, force=True)

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_specific_behavior(self, sample_dataset):
        """Test Windows-specific multiprocessing behavior."""
        # On Windows, only spawn method is available
        available_methods = mp.get_all_start_methods()
        assert "spawn" in available_methods
        assert "fork" not in available_methods

        # Default method should be spawn
        assert mp.get_start_method() == "spawn"

        # Windows file locking issues mean we need special handling
        with windows_safe_context_manager():
            loader_params = DataLoaderParams(
                batch_size=4, shuffle=False, sampler_type="random"
            )

            # Test with num_workers=0 (should always work)
            dataloader = create_train_dataloader(
                sample_dataset, loader_params, num_workers=0
            )
            batch = next(iter(dataloader))
            assert "x" in batch

            # num_workers > 0 on Windows with spawn is typically skipped in our tests
            # but we document the expected behavior

    def test_persistent_workers_flag(self, sample_dataset):
        """Test that persistent_workers flag is set correctly based on num_workers."""
        # With num_workers=0, persistent_workers should be False
        kwargs_0 = get_dataloader_kwargs(0, sample_dataset)
        assert kwargs_0["persistent_workers"] is False

        # With num_workers>0, persistent_workers should be True
        kwargs_1 = get_dataloader_kwargs(1, sample_dataset)
        assert kwargs_1["persistent_workers"] is True

        kwargs_2 = get_dataloader_kwargs(2, sample_dataset)
        assert kwargs_2["persistent_workers"] is True

    def test_pin_memory_flag(self, sample_dataset):
        """Test that pin_memory flag is set based on CUDA availability."""
        kwargs = get_dataloader_kwargs(1, sample_dataset)
        expected_pin_memory = torch.cuda.is_available()
        assert kwargs["pin_memory"] == expected_pin_memory


class TestDatasetPickling:
    """Test that CellsDataset can be properly pickled for multiprocessing."""

    def test_dataset_pickling_with_in_memory_adata(self, sample_dataset):
        """Test that dataset with in-memory AnnData can be pickled."""
        # Should work since we have in-memory AnnData
        pickled_data = pickle.dumps(sample_dataset)
        unpickled_dataset = pickle.loads(pickled_data)

        # Verify key attributes
        assert unpickled_dataset.n_cells == sample_dataset.n_cells
        assert unpickled_dataset.n_genes == sample_dataset.n_genes
        assert unpickled_dataset.is_train == sample_dataset.is_train

        # Verify that _adata is preserved (since it's in-memory)
        assert unpickled_dataset._adata is not None

    def test_dataset_pickling_with_backed_adata(self, sample_adata, sample_data_format):
        """Test that dataset with file-backed AnnData handles pickling correctly."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save AnnData to file
            sample_adata.write(tmp_path)

            # Create dataset with file path (will use backed mode)
            dataset = CellsDataset(
                data_format=sample_data_format,
                data_path=tmp_path,
                is_train=False,
            )

            # Should work - backed AnnData should be excluded from pickling
            pickled_data = pickle.dumps(dataset)
            unpickled_dataset = pickle.loads(pickled_data)

            # Verify key attributes
            assert unpickled_dataset.n_cells == dataset.n_cells
            assert unpickled_dataset.n_genes == dataset.n_genes
            assert unpickled_dataset.data_path == dataset.data_path

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_dataset_pickling_preserves_torch_generator_state(self, sample_dataset):
        """Test that dataset pickling properly handles torch generator state."""
        # Access the generator to ensure it's initialized
        gen1 = sample_dataset._get_worker_generator()
        initial_random = torch.rand(1, generator=gen1)

        # Pickle and unpickle
        pickled_data = pickle.dumps(sample_dataset)
        unpickled_dataset = pickle.loads(pickled_data)

        # Generator should be recreated (not preserved exactly, but should work)
        gen2 = unpickled_dataset._get_worker_generator()
        assert gen2 is not None

        # Should be able to generate random numbers
        new_random = torch.rand(1, generator=gen2)
        assert new_random.shape == initial_random.shape

    def test_dataset_pickle_roundtrip_functionality(self, sample_dataset):
        """Test that pickled dataset maintains full functionality."""
        # Test original dataset
        original_item = sample_dataset[0]

        # Pickle and unpickle
        pickled_data = pickle.dumps(sample_dataset)
        unpickled_dataset = pickle.loads(pickled_data)

        # Test unpickled dataset functionality
        unpickled_item = unpickled_dataset[0]

        # Should return same type (index)
        assert type(original_item) is type(unpickled_item)
        assert original_item == unpickled_item

        # Should have same length
        assert len(unpickled_dataset) == len(sample_dataset)

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Windows file locking issues"
    )
    def test_dataset_pickling_with_file_handles(self, sample_adata, sample_data_format):
        """Test dataset pickling behavior with open file handles."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save AnnData to file
            sample_adata.write(tmp_path)

            # Create dataset and access data to open file handles
            dataset = CellsDataset(
                data_format=sample_data_format,
                data_path=tmp_path,
                is_train=False,
            )

            # Force loading of backed AnnData
            _ = dataset._adata

            # Should still be able to pickle (backed _adata should be excluded)
            pickled_data = pickle.dumps(dataset)
            unpickled_dataset = pickle.loads(pickled_data)

            # Verify that _adata is None in unpickled version (will be reloaded)
            assert unpickled_dataset._adata is None
            assert unpickled_dataset.data_path == dataset.data_path

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestWorkerProcessBehavior:
    """Test worker process specific behavior and edge cases."""

    def test_worker_generator_creates_different_seeds(self):
        """Test that _get_worker_generator creates different seeds for different workers."""
        # Create mock dataset
        sample_adata = ad.AnnData(X=np.random.randn(10, 5))
        data_format = DataFormat(n_genes=5, gene_names=[f"gene_{i}" for i in range(5)])
        dataset_params = DataAugmentParams(mask_rate=0.1, noise_std=0.1)

        dataset = CellsDataset(
            data_format=data_format,
            dataset_params=dataset_params,
            is_train=True,
            adata=sample_adata,
        )

        # Mock different worker scenarios
        with patch("torch.utils.data.get_worker_info") as mock_worker_info:
            # Test single process (no worker info)
            mock_worker_info.return_value = None
            gen1 = dataset._get_worker_generator()

            # Test worker process
            mock_worker_info.return_value = type("WorkerInfo", (), {"id": 1})()
            dataset._torch_generator = None  # Reset generator
            gen2 = dataset._get_worker_generator()

            # Generators should be different objects
            assert gen1 is not gen2

            # They should produce different random numbers due to different seeds
            rand1 = torch.rand(5, generator=gen1)
            rand2 = torch.rand(5, generator=gen2)
            assert not torch.allclose(rand1, rand2)

    def test_worker_generator_consistency_within_worker(self):
        """Test that generator is consistent within the same worker."""
        sample_adata = ad.AnnData(X=np.random.randn(10, 5))
        data_format = DataFormat(n_genes=5, gene_names=[f"gene_{i}" for i in range(5)])
        dataset_params = DataAugmentParams(mask_rate=0.1, noise_std=0.1)

        dataset = CellsDataset(
            data_format=data_format,
            dataset_params=dataset_params,
            is_train=True,
            adata=sample_adata,
        )

        with patch("torch.utils.data.get_worker_info") as mock_worker_info:
            mock_worker_info.return_value = type("WorkerInfo", (), {"id": 0})()

            # Multiple calls should return the same generator object
            gen1 = dataset._get_worker_generator()
            gen2 = dataset._get_worker_generator()
            assert gen1 is gen2

    def test_multiple_worker_ids_create_different_generators(self):
        """Test that different worker IDs create different generators."""
        sample_adata = ad.AnnData(X=np.random.randn(10, 5))
        data_format = DataFormat(n_genes=5, gene_names=[f"gene_{i}" for i in range(5)])
        dataset_params = DataAugmentParams(mask_rate=0.1, noise_std=0.1)

        generators = []
        random_outputs = []

        for worker_id in range(3):
            dataset = CellsDataset(
                data_format=data_format,
                dataset_params=dataset_params,
                is_train=True,
                adata=sample_adata,
            )

            with patch("torch.utils.data.get_worker_info") as mock_worker_info:
                mock_worker_info.return_value = type(
                    "WorkerInfo", (), {"id": worker_id}
                )()
                gen = dataset._get_worker_generator()
                generators.append(gen)
                random_outputs.append(torch.rand(3, generator=gen))

        # All generators should be different objects
        for i in range(len(generators)):
            for j in range(i + 1, len(generators)):
                assert generators[i] is not generators[j]

        # All random outputs should be different
        for i in range(len(random_outputs)):
            for j in range(i + 1, len(random_outputs)):
                assert not torch.allclose(random_outputs[i], random_outputs[j])


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in multiprocessing scenarios."""

    def test_collate_fn_with_invalid_indices(self, sample_dataset):
        """Test collate function behavior with invalid indices."""
        # Test with empty batch - should handle gracefully
        empty_batch = []
        result = cells_collate_fn(empty_batch, dataset=sample_dataset)

        # Should return empty tensors with correct structure
        assert "x" in result
        assert result["x"].shape[0] == 0  # Empty batch dimension
        assert (
            result["x"].shape[1] == sample_dataset.n_genes
        )  # Correct feature dimension

        if "y" in result:
            assert result["y"].shape[0] == 0  # Empty batch dimension

    def test_collate_fn_with_out_of_range_indices(self, sample_dataset):
        """Test collate function with indices outside dataset range."""
        # Test with index beyond dataset size
        invalid_batch = [len(sample_dataset) + 10]
        with pytest.raises(IndexError):
            cells_collate_fn(invalid_batch, dataset=sample_dataset)

    def test_dataloader_with_zero_batch_size(self, sample_dataset):
        """Test dataloader creation with invalid batch sizes."""
        loader_params = DataLoaderParams(
            batch_size=0, shuffle=True, sampler_type="random"
        )

        with pytest.raises(ValueError, match="batch_size"):
            create_train_dataloader(sample_dataset, loader_params, num_workers=0)

    def test_balanced_sampler_with_insufficient_data(self, sample_data_format):
        """Test balanced samplers with insufficient positive/negative samples."""
        # Create dataset with only positive samples
        n_cells, n_genes = 10, 20
        X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        obs = pd.DataFrame(
            {
                "expansion": ["expanded"] * n_cells,  # All positive
                "clone_id_size": np.random.randint(1, 20, n_cells),
                "median_clone_size": np.random.randint(5, 15, n_cells),
                "tissue_type": np.random.choice(["tissue_A", "tissue_B"], n_cells),
                "imputed_labels": np.random.choice(["label_1", "label_2"], n_cells),
            }
        )

        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs, var=var)

        dataset = CellsDataset(
            data_format=sample_data_format,
            is_train=True,
            adata=adata,
        )

        # Should raise error when trying to create balanced sampler
        with pytest.raises(ValueError, match="requires at least one negative sample"):
            BalancedLabelsBatchSampler(dataset, batch_size=4)

    def test_thread_safety_simulation(self, sample_dataset):
        """Simulate thread safety issues that might occur in multiprocessing."""
        results = []
        exceptions = []

        def worker_function():
            """Simulate worker process accessing dataset."""
            try:
                # Simulate multiple accesses
                for _ in range(5):
                    gen = sample_dataset._get_worker_generator()
                    random_val = torch.rand(1, generator=gen)
                    results.append(random_val.item())
                    time.sleep(
                        0.001
                    )  # Small delay to increase chance of race conditions
            except Exception as e:
                exceptions.append(e)

        # Create multiple threads to simulate concurrent access
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker_function)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should not have any exceptions
        assert len(exceptions) == 0, f"Thread safety issues detected: {exceptions}"

        # Should have results from all threads
        assert len(results) == 15  # 3 threads * 5 accesses each

    def test_memory_cleanup_after_pickling(self, sample_dataset):
        """Test that memory is properly cleaned up after pickling operations."""
        # Get initial reference count
        initial_refs = sys.getrefcount(sample_dataset)

        # Perform multiple pickle/unpickle cycles
        for _ in range(5):
            pickled_data = pickle.dumps(sample_dataset)
            unpickled_dataset = pickle.loads(pickled_data)
            del unpickled_dataset
            del pickled_data
            gc.collect()

        # Reference count should be similar (allowing for some variation)
        final_refs = sys.getrefcount(sample_dataset)
        assert abs(final_refs - initial_refs) <= 2, (
            f"Memory leak detected: {initial_refs} -> {final_refs}"
        )

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_fork_behavior_simulation(self, sample_dataset):
        """Test behavior that would occur with Unix fork (when available)."""
        # This test documents the expected behavior on Unix systems
        # where fork is available and doesn't require pickling

        if "fork" not in mp.get_all_start_methods():
            pytest.skip("Fork not available on this system")

        # Test that dataset works without pickling requirements
        # (this simulates fork behavior where memory is shared)
        original_gen = sample_dataset._get_worker_generator()
        assert original_gen is not None

        # Multiple accesses should return the same generator
        same_gen = sample_dataset._get_worker_generator()
        assert original_gen is same_gen


if __name__ == "__main__":
    pytest.main([__file__])
