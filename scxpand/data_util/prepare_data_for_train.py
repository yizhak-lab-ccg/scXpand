from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from anndata import AnnData

from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.data_util.data_splitter import split_data
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import DATA_FORMAT_FILE, DATA_SPLITS_FILE


logger = get_logger()


@dataclass
class TrainingDataBundle:
    """Container for all data components needed for model training.

    Attributes:
        adata: Preprocessed AnnData object ready for training.
        row_inds_train: Sorted array of row indices for the training set.
        row_inds_dev: Sorted array of row indices for the validation set.
        data_format: DataFormat object containing preprocessing parameters.
        save_path: Path to the directory where training artifacts are saved.
    """

    adata: AnnData
    row_inds_train: np.ndarray
    row_inds_dev: np.ndarray
    data_format: DataFormat
    save_path: Path


def prepare_data_for_training(
    data_path: str | Path,
    aux_categorical_types: tuple[str, ...] = (),
    use_log_transform: bool = True,
    use_zscore_norm: bool = True,
    save_dir: str | Path = "results/temp",
    dev_ratio: float = 0.2,
    rand_seed: int = 42,
    resume: bool = False,
    batch_size: int = 500_000,
) -> TrainingDataBundle:
    """Prepare single-cell RNA sequencing data for model training.

    Args:
        data_path: Path to the input AnnData file (.h5ad format).
        aux_categorical_types: Tuple of categorical feature names to include.
        use_log_transform: Whether to apply log1p transformation.
        use_zscore_norm: Whether to apply z-score normalization per gene.
        save_dir: Directory to save data format files and split information.
        dev_ratio: Proportion of data to use for validation (0.0 to 1.0).
        rand_seed: Random seed for reproducible data splitting.
        resume: If True, load existing data format and splits from save_dir.
        batch_size: Batch size for computing gene statistics.

    Returns:
        TrainingDataBundle containing all data components needed for training.

    Raises:
        FileNotFoundError: If data_path doesn't exist.
        ValueError: If dev_ratio is not between 0.0 and 1.0.
        KeyError: If required metadata columns are missing from AnnData.obs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    data_format_path = save_dir / DATA_FORMAT_FILE
    data_splits_path = save_dir / DATA_SPLITS_FILE

    if not resume:
        logger.info(f"Reading data from: {data_path}")
        adata = ad.read_h5ad(data_path, backed="r")

        row_inds_train, row_inds_dev = split_data(
            adata=adata,
            dev_ratio=dev_ratio,
            random_seed=rand_seed,
            save_path=save_dir,
        )
        np.savez(data_splits_path, train=row_inds_train, dev=row_inds_dev)

        data_format = DataFormat(
            use_log_transform=use_log_transform,
            use_zscore_norm=use_zscore_norm,
            aux_categorical_types=aux_categorical_types,
        )

        data_format.create_data_format(
            data_path=data_path,
            adata=adata,
            row_inds_train=row_inds_train,
            batch_size=batch_size,
        )
        adata = data_format.prepare_adata_for_training(adata, reorder_genes=False)
        logger.info(f"Saving data format to: {data_format_path}")
        data_format.save(data_format_path)
    else:
        data_format = load_data_format(data_format_path)
        splits = np.load(data_splits_path)
        logger.info(f"Loading data train/dev splits from: {data_splits_path}")
        logger.info(f"Reading data from: {data_path}")
        adata = ad.read_h5ad(data_path, backed="r")
        logger.info(f"Converting data format from: {data_format_path}")
        logger.info(
            f"Resuming with {data_format.n_genes} genes, normalization stats loaded (mu: {data_format.genes_mu.shape}, sigma: {data_format.genes_sigma.shape})"
        )
        adata = data_format.prepare_adata_for_training(adata)
        row_inds_train = splits["train"]
        row_inds_dev = splits["dev"]

    return TrainingDataBundle(
        adata=adata,
        row_inds_train=row_inds_train,
        row_inds_dev=row_inds_dev,
        data_format=data_format,
        save_path=save_dir,
    )
