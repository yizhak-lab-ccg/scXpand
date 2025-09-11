import time

from pathlib import Path

import optuna

from scxpand.core.evaluation import evaluate_predictions_and_save
from scxpand.data_util.prepare_data_for_train import prepare_data_for_training
from scxpand.mlp.mlp_params import MLPParam
from scxpand.mlp.mlp_trainer import logger, run_mlp_inference, run_trainer
from scxpand.util.classes import ModelType
from scxpand.util.general_util import get_device, save_params, set_seed


def run_mlp_training(
    data_path: str | Path,
    base_save_dir: str | Path,
    prm: MLPParam,
    device: str | None = None,
    dev_ratio: float = 0.2,
    trial: optuna.Trial | None = None,
    score_metric: str = "harmonic_avg/AUROC",
    resume: bool = False,
    num_workers: int = 0,
) -> dict[str, dict[str, float]]:
    """Run neural network training and evaluation.

    Args:
        data_path: Path to data file (optional if adata provided)
        base_save_dir: Directory to save model and results
        prm: Neural network parameters
        device: Device to use for training (CPU/GPU)
        dev_ratio: Ratio of data to use for validation
        trial: Optuna trial object for hyperparameter optimization
        score_metric: Metric to use for scoring
        resume: Whether to resume from existing checkpoint
        adata: AnnData object (optional if data_path provided)
        num_workers: Number of workers for data loading

    Returns:
        Dictionary containing evaluation results
    """
    start_time = time.time()
    base_save_dir = Path(base_save_dir)
    set_seed(prm.random_seed)
    save_params(params=prm, save_dir=base_save_dir)

    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")

    data_bundle = prepare_data_for_training(
        data_path=data_path,
        aux_categorical_types=prm.aux_categorical_types,
        use_log_transform=prm.use_log_transform,
        save_dir=base_save_dir,
        dev_ratio=dev_ratio,
        rand_seed=prm.random_seed,
        resume=resume,
    )

    adata = data_bundle.adata
    row_inds_train = data_bundle.row_inds_train
    row_inds_dev = data_bundle.row_inds_dev
    data_format = data_bundle.data_format

    logger.info("Training the classifier...")
    model = run_trainer(
        data_path=data_path,
        data_format=data_format,
        row_inds_train=row_inds_train,
        row_inds_dev=row_inds_dev,
        save_path=base_save_dir,
        prm=prm,
        device=device,
        trial=trial,
        score_metric=score_metric,
        resume=resume,
        num_workers=num_workers,
    )

    y_dev_pred_prob = run_mlp_inference(
        data_path=data_path,
        data_format=data_format,
        eval_row_inds=row_inds_dev,
        model=model,
        device=device,
        batch_size=prm.inference_batch_size,
    )

    # Use utility function for evaluation and saving
    results = evaluate_predictions_and_save(
        y_pred_prob=y_dev_pred_prob,
        obs_df=adata.obs.iloc[row_inds_dev],
        model_type=ModelType.MLP,
        save_path=base_save_dir,
        eval_name="dev",
        score_metric=score_metric,
        trial=trial,
    )

    # Note: Model is automatically saved as best_ckpt.pt by TrainLogger during training
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    return results
