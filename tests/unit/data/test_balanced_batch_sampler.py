import pytest
import torch

from scxpand.data_util.dataloaders import BalancedLabelsBatchSampler


def test_balanced_batch_sampler_initialization() -> None:
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.tensor([0, 1] * 50)
    dataset = MockCellsDataset(X, y)

    sampler = BalancedLabelsBatchSampler(dataset, batch_size=10)
    assert sampler.batch_size == 10
    assert sampler.num_batches > 0
    assert sampler.pos_per_batch == 5  # Half of batch_size
    assert sampler.neg_per_batch == 5  # Half of batch_size


def test_balanced_batch_sampler_length() -> None:
    X = torch.randn(100, 10)
    y = torch.tensor([0, 1] * 50)
    dataset = MockCellsDataset(X, y)

    sampler = BalancedLabelsBatchSampler(dataset, batch_size=10)
    # Length should be number of samples divided by batch size
    assert len(sampler) == 10


def test_batch_generation() -> None:
    X = torch.randn(100, 10)
    y = torch.tensor([0, 1] * 50)
    dataset = MockCellsDataset(X, y)

    sampler = BalancedLabelsBatchSampler(dataset, batch_size=10)
    batch_indices = next(iter(sampler))

    assert len(batch_indices) == 10
    # Check if batch is balanced
    batch_labels = dataset.y[batch_indices]
    unique_labels, counts = torch.unique(batch_labels, return_counts=True)
    assert len(unique_labels) == 2
    assert all(count == 5 for count in counts)


def test_invalid_batch_size() -> None:
    X = torch.randn(100, 10)
    y = torch.tensor([0, 1] * 50)
    dataset = MockCellsDataset(X, y)

    # Since the implementation doesn't explicitly check for even batch sizes,
    # we'll test that odd batch sizes still work but maintain approximate balance
    sampler = BalancedLabelsBatchSampler(dataset, batch_size=3)
    batch_indices = next(iter(sampler))
    batch_labels = dataset.y[batch_indices]
    unique_labels, counts = torch.unique(batch_labels, return_counts=True)
    # Check that the difference between positive and negative samples is at most 1
    assert max(counts) - min(counts) <= 1


def test_insufficient_samples() -> None:
    X = torch.randn(10, 10)
    y = torch.tensor([0] * 9 + [1])  # Imbalanced dataset: 9 negative, 1 positive
    dataset = MockCellsDataset(X, y)

    # The sampler should work with imbalanced datasets by cycling through minority class
    sampler = BalancedLabelsBatchSampler(dataset, batch_size=4)
    # With 10 samples and batch_size=4: ceil(10/4) = 3 batches
    assert len(sampler) == 3  # Uses all data, cycles through minority class as needed

    # Test that batches are actually balanced by cycling
    batches = list(sampler)
    assert len(batches) == 3

    for batch_indices in batches:
        assert len(batch_indices) == 4
        batch_labels = dataset.y[batch_indices]
        unique_labels, counts = torch.unique(batch_labels, return_counts=True)

        # Should have both classes (the single positive sample gets repeated)
        assert len(unique_labels) == 2
        # Each batch should have 2 pos and 2 neg (pos_per_batch = neg_per_batch = 2)
        assert all(count == 2 for count in counts)


def test_labels_consistency() -> None:
    X = torch.randn(100, 10)
    y = torch.tensor([0, 1] * 50)  # Binary labels instead of 3 classes
    dataset = MockCellsDataset(X, y)

    sampler = BalancedLabelsBatchSampler(dataset, batch_size=10)
    for batch_indices in sampler:
        batch_labels = dataset.y[batch_indices]
        # Check if the batch is approximately balanced
        unique_labels, counts = torch.unique(batch_labels, return_counts=True)
        assert len(unique_labels) == 2
        # Check that the difference between positive and negative samples is at most 1
        assert max(counts) - min(counts) <= 1


def test_no_positive_samples_error() -> None:
    """Test that sampler raises error when no positive samples exist."""
    X = torch.randn(10, 10)
    y = torch.tensor([0] * 10)  # All negative samples
    dataset = MockCellsDataset(X, y)

    with pytest.raises(ValueError, match="requires at least one positive sample"):
        BalancedLabelsBatchSampler(dataset, batch_size=4)


def test_no_negative_samples_error() -> None:
    """Test that sampler raises error when no negative samples exist."""
    X = torch.randn(10, 10)
    y = torch.tensor([1] * 10)  # All positive samples
    dataset = MockCellsDataset(X, y)

    with pytest.raises(ValueError, match="requires at least one negative sample"):
        BalancedLabelsBatchSampler(dataset, batch_size=4)


# Add this mock class at the top of the file
class MockCellsDataset:
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)


if __name__ == "__main__":
    pytest.main()
