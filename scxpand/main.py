"""Single entry point for all scXpand operations.

Example usage:
# Training models:
python -m scxpand.main train --model_type autoencoder --data_path data/example_data.h5ad --n_epochs 1
python -m scxpand.main train --model_type mlp --data_path data/example_data.h5ad --n_epochs 10
python -m scxpand.main train --model_type lightgbm --data_path data/example_data.h5ad
python -m scxpand.main train --model_type linear --data_path data/example_data.h5ad
python -m scxpand.main train --model_type svm --data_path data/example_data.h5ad --config_path config/svm_config.json

# Hyperparameter optimization:
python -m scxpand.main optimize --model_type autoencoder --n_trials 100 --data_path data/example_data.h5ad
python -m scxpand.main optimize --model_type mlp --n_trials 100 --data_path data/example_data.h5ad --n_epochs 10
python -m scxpand.main optimize-all --n_trials 10 --data_path data/example_data.h5ad --num_workers 6
python -m scxpand.main optimize-all --n_trials 100 --data_path data/example_data.h5ad --model_types mlp,autoencoder

# Model prediction/inference:
python -m scxpand.main predict --model_path results/autoencoder --data_path new_data.h5ad
python -m scxpand.main predict --model_path results/mlp --data_path new_data.h5ad
"""

from pathlib import Path

import fire
import matplotlib

from scxpand.core.inference import run_inference
from scxpand.core.model_types import MODEL_TYPES
from scxpand.hyperopt.hyperopt_optimizer import HyperparameterOptimizer
from scxpand.pretrained import PRETRAINED_MODELS
from scxpand.util.classes import ModelType, ensure_model_type
from scxpand.util.general_util import get_device, get_new_version_path, load_and_override_params
from scxpand.util.io import load_eval_indices
from scxpand.util.logger import get_logger
from scxpand.util.training_utils import call_training_function, validate_and_setup_common


matplotlib.use("Agg")


logger = get_logger()


def train(
    model_type: ModelType | str,
    data_path: str = "data/example_data.h5ad",
    save_dir: str | None = None,
    config_path: str | None = None,
    resume: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> None:
    """Train a single model.

    Args:
        model_type: Type of model to train (autoencoder, mlp, lightgbm, logistic, svm)
        data_path: Path to input data file
        save_dir: Directory to save results (if None, uses default for model type)
        config_path: Path to configuration file
        resume: Whether to resume from existing checkpoint
        num_workers: Number of workers for data loading
        **kwargs: Additional parameters to override config
    """
    # Common validation and setup
    model_type_enum, model_spec = validate_and_setup_common(model_type=model_type, data_path=data_path)

    # Create save directory using registry default
    save_dir = get_new_version_path(save_dir or model_spec.default_save_dir)

    # Load parameters and apply overrides
    prm = load_and_override_params(param_class=model_spec.param_class, config_path=config_path, **kwargs)

    # Run training
    call_training_function(
        model_type=model_type_enum,
        train_fn=model_spec.runner,
        data_path=data_path,
        save_dir=save_dir,
        prm=prm,
        resume=resume,
        num_workers=num_workers,
    )


def optimize(
    model_type: ModelType | str,
    data_path: str = "data/example_data.h5ad",
    n_trials: int = 100,
    study_name: str | None = None,
    storage_path: str = "results/optuna_studies",
    score_metric: str = "harmonic_avg/AUROC",
    resume: bool = True,
    seed_base: int = 42,
    num_workers: int = 4,
    config_path: str | None = None,
    fail_fast: bool = False,
    **kwargs,
) -> None:
    """Run hyperparameter optimization for a specified model type.

    Args:
        model_type: Type of model to optimize (autoencoder, mlp, lightgbm, logistic, svm)
        data_path: Path to the input data file (h5ad format)
        n_trials: Number of optimization trials to run
        study_name: Name of the optimization study (defaults to model_type)
        storage_path: Directory to store optimization results
        score_metric: Metric to optimize (e.g., "harmonic_avg/AUROC", "AUROC", "AUPRC")
        resume: Whether to resume from existing study (False = start fresh)
        seed_base: Base seed for reproducibility across trials
        num_workers: Number of workers for parallel processing
        config_path: Path to configuration file for base parameters
        fail_fast: Whether to fail immediately on any exception (for testing)
        **kwargs: Additional parameters to override config

    Raises:
        ValueError: If model_type is not supported for optimization
        FileNotFoundError: If data_path does not exist
        ValueError: If study already exists and resume=False (with instructions to delete manually)
    """
    # Common validation and setup
    model_type_enum, _model_spec = validate_and_setup_common(model_type=model_type, data_path=data_path)

    study_name = study_name or model_type_enum.value

    # Create optimizer instance
    optimizer = HyperparameterOptimizer(
        model_type=model_type_enum,
        data_path=data_path,
        study_name=study_name,
        storage_path=storage_path,
        score_metric=score_metric,
        seed_base=seed_base,
        num_workers=num_workers,
        config_path=config_path,
        resume=resume,
        fail_fast=fail_fast,
        **kwargs,
    )

    # Run optimization and print results
    study = optimizer.run_optimization(n_trials=n_trials)
    optimizer.print_results(study)


def optimize_all(
    data_path: str = "data/example_data.h5ad",
    n_trials: int = 100,
    storage_path: str = "results/optuna_studies",
    score_metric: str = "harmonic_avg/AUROC",
    resume: bool = True,
    num_workers: int = 4,
    model_types: list[ModelType] | None = None,
    **kwargs,
) -> None:
    """Run hyperparameter optimization for all supported model types or a specified subset.

    Args:
        data_path: Path to the input data file (h5ad format)
        n_trials: Number of optimization trials per model type
        storage_path: Directory to store optimization results
        score_metric: Metric to optimize (e.g., "harmonic_avg/AUROC", "AUROC", "AUPRC")
        resume: Whether to resume existing studies (False = start fresh for all models)
        num_workers: Number of workers for parallel processing
        model_types: List of model types to optimize in order. If None, optimizes all supported models.
                    Supported types: ["autoencoder", "mlp", "lightgbm", "logistic", "svm"]
        **kwargs: Additional parameters to override config for all models
    """
    # Validate data path only
    if not data_path:
        raise ValueError("data_path cannot be empty")
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not data_file.is_file():
        raise ValueError(f"Data path is not a file: {data_path}")

    logger.info(f"Using device: {get_device()}")

    # Determine which model types to optimize
    if model_types is None:
        # Use all model types from registry
        models_to_optimize = list(MODEL_TYPES.keys())
    else:
        # Validate provided model types
        models_to_optimize = [ensure_model_type(model_type) for model_type in model_types]

    logger.info(f"Will optimize models in order: {models_to_optimize}")

    for model_type in models_to_optimize:
        logger.info(f"\nOptimizing {model_type.value}...")
        optimize(
            model_type=model_type,
            data_path=data_path,
            n_trials=n_trials,
            study_name=model_type.value,
            storage_path=storage_path,
            score_metric=score_metric,
            resume=resume,
            num_workers=num_workers,
            **kwargs,
        )


def predict(
    data_path: str,
    model_path: str | None = None,
    model_name: str | None = None,
    model_url: str | None = None,
    save_path: str | None = None,
    batch_size: int = 1024,
    num_workers: int = 4,
    eval_row_inds: str | None = None,
) -> None:
    """Unified prediction function for both local and pre-trained models.

    Args:
        data_path: Path to input data file (h5ad format)
        model_path: Path to directory containing the trained model (for local models)
        model_name: Name of pre-trained model from registry (for pre-trained models)
        model_url: Direct URL to model ZIP file (for any external model)
        save_path: Directory to save prediction results
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        eval_row_inds: Path to file containing cell indices to evaluate (one per line), or None for all cells

    Examples:
        # Local model inference
        predict --data_path my_data.h5ad --model_path results/mlp

        # Registry model inference
        predict --data_path my_data.h5ad --model_name pan_cancer_autoencoder

        # Direct URL inference (any external model)
        predict --data_path my_data.h5ad --model_url "https://your-platform.com/model.zip"
    """
    # Load evaluation indices if specified
    eval_indices = None
    if eval_row_inds:
        eval_indices = load_eval_indices(eval_row_inds)

    # Set default save path for pre-trained models if not provided
    if save_path is None and model_name is not None:
        save_path = f"results/{model_name}_predictions"

    # Use the unified inference function
    run_inference(
        data_path=data_path,
        model_path=model_path,
        model_name=model_name,
        model_url=model_url,
        save_path=save_path,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_row_inds=eval_indices,
    )


def list_pretrained_models() -> None:
    """List all available pre-trained models with their information."""

    logger.info("Available pre-trained models:")
    logger.info("=" * 50)

    for name, info in PRETRAINED_MODELS.items():
        logger.info(f"Name: {name}")
        logger.info(f"  Version: {info.version}")
        logger.info(f"  URL configured: {'Yes' if info.url else 'No'}")
        logger.info("  Model type: Auto-detected from model_type.txt")
        logger.info("")


def main():
    """Main entry point for the scxpand CLI."""
    fire.Fire(
        {
            "train": train,
            "optimize": optimize,
            "optimize-all": optimize_all,
            "predict": predict,
            "list-models": list_pretrained_models,
        }
    )


if __name__ == "__main__":
    main()
