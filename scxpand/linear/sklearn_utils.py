from pathlib import Path

import anndata as ad
import joblib
import numpy as np

from sklearn.base import BaseEstimator

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataloaders import create_eval_dataloader
from scxpand.data_util.dataset import CellsDataset
from scxpand.linear.linear_trainer import LinearBatchPredictor
from scxpand.util.classes import DataAugmentParams
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import SKLEARN_MODEL_FILE


logger = get_logger()


def run_linear_inference(
    model: BaseEstimator,
    eval_row_inds: np.ndarray | None = None,
    *,
    data_format: DataFormat | None = None,
    data_path: str | Path | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
    adata: ad.AnnData = None,
) -> np.ndarray:
    """Runs inference using a trained scikit-learn model. Accepts either data_path or adata (AnnData object)."""
    dataset_params = DataAugmentParams(mask_rate=0.0, noise_std=0.0, soft_loss_beta=0.0)
    dataset = CellsDataset(
        data_format=data_format,
        row_inds=eval_row_inds,
        dataset_params=dataset_params,
        is_train=False,
        data_path=data_path,
        adata=adata,
    )

    # Wrap in DataLoader
    dataloader = create_eval_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    # Use the same batch predictor as in training
    predictor = LinearBatchPredictor(dataset=dataset, dataloader=dataloader)
    predictions = predictor.predict_all(model=model)

    logger.info(f"Inference complete. Predictions shape: {predictions.shape}")
    return predictions


def load_sklearn_model(results_path: Path) -> BaseEstimator:
    """Loads a trained scikit-learn model."""
    logger.info(f"Loading scikit-learn model from: {results_path}")
    model_file = results_path / SKLEARN_MODEL_FILE
    model = joblib.load(model_file)
    logger.info("Scikit-learn model loaded successfully.")

    return model
