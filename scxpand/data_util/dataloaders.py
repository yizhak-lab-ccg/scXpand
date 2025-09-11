import math

from collections.abc import Iterator

import numpy as np

from torch.utils.data import DataLoader, Sampler

from scxpand.data_util.dataset import CellsDataset, get_dataloader_kwargs, logger
from scxpand.util.classes import DataLoaderParams
from scxpand.util.general_util import metrics_dict_to_table


# Constants
RNG_STATE_KEY = "rng"


class BalancedLabelsBatchSampler(Sampler):
    def __init__(self, dataset: CellsDataset, batch_size: int, seed: int = 1) -> None:
        """Balanced batch sampler that ensures each batch has roughly equal number of positive and negative examples.

        Args:
            dataset: CellsDataset instance.
            batch_size: int, batch size.
            seed: int, random seed for reproducibility.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Get indices for positive and negative samples.
        labels = dataset.y.numpy()
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        # Validate that we have both positive and negative samples
        if len(self.pos_indices) == 0:
            raise ValueError("BalancedLabelsBatchSampler requires at least one positive sample (label=1)")
        if len(self.neg_indices) == 0:
            raise ValueError("BalancedLabelsBatchSampler requires at least one negative sample (label=0)")

        # Calculate number of positive/negative samples per batch.
        self.pos_per_batch = self.batch_size // 2
        self.neg_per_batch = self.batch_size - self.pos_per_batch

        # Use the same number of batches as random sampler: ceil(dataset_size / batch_size)
        # This ensures similar epoch length and sample usage as the random baseline
        self.num_batches = math.ceil(len(self.dataset) / batch_size)

        logger.info(
            f"BalancedLabelsBatchSampler: {len(self.pos_indices)} positive, {len(self.neg_indices)} negative samples. "
            f"Using {self.pos_per_batch} positive and {self.neg_per_batch} negative samples per batch. "
            f"Number of batches per epoch: {self.num_batches}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        # Create copies and shuffle indices using the instance's RNG.
        pos_indices = self.rng.permutation(self.pos_indices.copy())
        neg_indices = self.rng.permutation(self.neg_indices.copy())

        # Keep track of current position in each class
        pos_position = 0
        neg_position = 0

        # Generate batches.
        for _ in range(self.num_batches):
            batch_pos = []
            batch_neg = []

            # Get positive samples (cycling through if needed)
            remaining_pos = self.pos_per_batch
            while remaining_pos > 0:
                available_pos = len(pos_indices) - pos_position
                to_take_pos = min(remaining_pos, available_pos)

                batch_pos.extend(pos_indices[pos_position : pos_position + to_take_pos])
                remaining_pos -= to_take_pos
                pos_position += to_take_pos

                # If we reached the end, shuffle and start over
                if pos_position >= len(pos_indices):
                    pos_position = 0
                    pos_indices = self.rng.permutation(self.pos_indices.copy())

            # Get negative samples (cycling through if needed)
            remaining_neg = self.neg_per_batch
            while remaining_neg > 0:
                available_neg = len(neg_indices) - neg_position
                to_take_neg = min(remaining_neg, available_neg)

                batch_neg.extend(neg_indices[neg_position : neg_position + to_take_neg])
                remaining_neg -= to_take_neg
                neg_position += to_take_neg

                # If we reached the end, shuffle and start over
                if neg_position >= len(neg_indices):
                    neg_position = 0
                    neg_indices = self.rng.permutation(self.neg_indices.copy())

            # Combine and shuffle the batch indices.
            batch = np.concatenate([batch_pos, batch_neg])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches

    def __getstate__(self) -> dict:
        """Prepare object for pickling - preserve seed for RNG recreation in worker process."""
        state = self.__dict__.copy()
        # Remove the RNG object - it will be recreated in __setstate__
        del state[RNG_STATE_KEY]
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore object after unpickling - recreate RNG with preserved seed."""
        self.__dict__.update(state)
        # Recreate RNG with the original seed
        self.rng = np.random.default_rng(self.seed)


class BalancedTypesBatchSampler(Sampler):
    def __init__(self, dataset: CellsDataset, batch_size: int, seed: int = 1) -> None:
        """Balanced batch sampler that equalizes the proportions of each stratum defined by
        the combinations of categorical features ["tissue_type", "imputed_labels"]. For composite groups
        with both positive and negative labels, the group is split into two strata. Each batch contains an
        equal (or nearly equal) number of samples from each stratum.

        Args:
            dataset: CellsDataset instance
            batch_size: int, must be at least as large as the number of strata.
            seed: int, random seed for reproducibility.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = dataset.y.numpy()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Create stratification groups for categorical features.
        self.feature_strata = self._create_feature_strata()
        # Omit strata with no items.
        self.feature_strata = {k: v for k, v in self.feature_strata.items() if len(v) > 0}

        n_items_in_dataset_per_stratum = {k: len(v) for k, v in self.feature_strata.items()}

        self.num_strata = len(self.feature_strata)

        if batch_size < self.num_strata:
            raise ValueError("Batch size must be at least as large as the number of strata")

        base_allocation = batch_size // self.num_strata
        remainder = batch_size % self.num_strata
        # Sort strata keys by size (number of items in the dataset for each stratum)
        sorted_keys = sorted(
            self.feature_strata.keys(),
            key=lambda k: n_items_in_dataset_per_stratum[k],
            reverse=True,
        )
        self.n_items_in_batch_per_stratum = {
            key: base_allocation + (1 if i < remainder else 0) for i, key in enumerate(sorted_keys)
        }

        # Display stratum allocation as a table
        stratum_table_data = {}
        for stratum, count in self.n_items_in_batch_per_stratum.items():
            stratum_table_data[stratum] = {"Items_per_Batch": count}
        stratum_table = metrics_dict_to_table(stratum_table_data, title=f"Batch Size {batch_size} - Stratum Allocation")
        logger.info(stratum_table)

        # Calculate number of complete batches possible for each stratum.
        n_max_batch_per_stratum: dict[str, int] = {}
        for stratum in self.feature_strata:
            tot_items_for_stratum = n_items_in_dataset_per_stratum[stratum]
            desired_n_items_in_batch = self.n_items_in_batch_per_stratum[stratum]
            if tot_items_for_stratum < desired_n_items_in_batch:
                # If we have fewer samples than needed, we can still use this stratum, by repeating samples,
                # so we set it to 1 batch.
                n_max_batch_per_stratum[stratum] = 1
            else:
                n_max_batch_per_stratum[stratum] = tot_items_for_stratum // desired_n_items_in_batch

        # Use the same number of batches as random sampler: ceil(dataset_size / batch_size)
        # This ensures similar epoch length and sample usage as the random baseline
        self.num_batches = math.ceil(len(self.dataset) / batch_size)
        logger.info(f"Number of batches per epoch: {self.num_batches}")

    def _create_feature_strata(self) -> dict[str, np.ndarray]:
        """Creates groups of indices for each composite category defined by "tissue_type" and "imputed_labels".

        For groups that contain both positive and negative labels, splits them into two strata (one for each label).
        """
        strata: dict[str, np.ndarray] = {}
        stratification = (
            self.dataset.obs_df["tissue_type"].astype(str) + "_" + self.dataset.obs_df["imputed_labels"].astype(str)
        )
        for comp in stratification.unique():
            mask = stratification == comp
            indices = np.where(mask)[0]
            labels_in_comp = self.labels[indices]
            unique_labels = np.unique(labels_in_comp)
            if len(unique_labels) == 2:
                pos_indices = indices[labels_in_comp == 1]
                neg_indices = indices[labels_in_comp == 0]
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    strata[f"{comp}_pos"] = pos_indices
                    strata[f"{comp}_neg"] = neg_indices
                else:
                    strata[comp] = indices
            else:
                strata[comp] = indices
        return strata

    def __iter__(self) -> Iterator[list[int]]:
        # Create copies and shuffle indices in each stratum
        stratum_indices: dict[str, np.ndarray] = {
            k: self.rng.permutation(indices.copy()) for k, indices in self.feature_strata.items()
        }

        # Keep track of current position in each stratum
        stratum_positions = dict.fromkeys(self.feature_strata, 0)

        for _ in range(self.num_batches):
            batch: list[int] = []
            for stratum, n_samples in self.n_items_in_batch_per_stratum.items():
                indices = stratum_indices[stratum]
                current_pos = stratum_positions[stratum]

                # Calculate how many samples we need from this stratum
                remaining = n_samples
                selected_indices = []

                while remaining > 0:
                    # Calculate how many samples we can take before reaching the end
                    available = len(indices) - current_pos
                    to_take = min(remaining, available)

                    # Take samples
                    selected_indices.extend(indices[current_pos : current_pos + to_take])
                    remaining -= to_take
                    current_pos += to_take

                    # If we reached the end, shuffle and start over
                    if current_pos >= len(indices):
                        current_pos = 0
                        indices = self.rng.permutation(indices.copy())
                        stratum_indices[stratum] = indices

                stratum_positions[stratum] = current_pos
                batch.extend(selected_indices)

            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    def __getstate__(self) -> dict:
        """Prepare object for pickling - preserve seed for RNG recreation in worker process."""
        state = self.__dict__.copy()
        # Remove the RNG object - it will be recreated in __setstate__
        del state[RNG_STATE_KEY]
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore object after unpickling - recreate RNG with preserved seed."""
        self.__dict__.update(state)
        # Recreate RNG with the original seed
        self.rng = np.random.default_rng(self.seed)


def create_train_dataloader(
    train_dataset: CellsDataset,
    loader_params: DataLoaderParams,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for training.

    Args:
        train_dataset: Dataset for training
        loader_params: Parameters for the loader
        num_workers: Number of worker processes for parallel data loading.

    Returns:
        DataLoader for training
    """
    # Get common DataLoader arguments
    loader_kwargs = get_dataloader_kwargs(num_workers, train_dataset)

    # Create with appropriate sampler
    if loader_params.sampler_type in ["balanced_labels", "balanced_types"]:
        # Create balanced sampler
        if loader_params.sampler_type == "balanced_labels":
            sampler = BalancedLabelsBatchSampler(train_dataset, batch_size=loader_params.batch_size)
        else:
            sampler = BalancedTypesBatchSampler(train_dataset, batch_size=loader_params.batch_size)

        # Use batch sampler
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            **loader_kwargs,
        )
    elif loader_params.sampler_type == "random":
        # Use random sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=loader_params.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    else:
        raise ValueError(f"Invalid sampler type: {loader_params.sampler_type}")

    logger.info(
        f"Created train data loader with sampler: {loader_params.sampler_type}, batch size: {loader_params.batch_size}, num_workers: {num_workers}"
    )
    return train_loader


def create_eval_dataloader(dataset: CellsDataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader optimized for evaluation and inference.

    Sets up a DataLoader with deterministic behavior (no shuffling) suitable
    for inference tasks. Automatically configures worker processes and
    memory settings based on the dataset.

    Args:
        dataset: CellsDataset configured for evaluation (is_train=False).
        batch_size: Number of cells per batch during inference.
        num_workers: Number of parallel data loading processes.

    Returns:
        DataLoader ready for inference with consistent ordering.


    """
    # Get common DataLoader arguments
    loader_kwargs = get_dataloader_kwargs(num_workers, dataset)

    # Create evaluation loader
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    logger.info(f"Created eval data loader with batch size: {batch_size}, num_workers: {num_workers}")
    return eval_loader
