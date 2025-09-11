"""Linear model training components - consolidated trainer with all functionality."""

from pathlib import Path

import joblib
import numpy as np
import optuna

from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader

from scxpand.core.evaluation import evaluate_predictions_and_save
from scxpand.data_util.dataloaders import create_eval_dataloader, create_train_dataloader
from scxpand.data_util.dataset import CellsDataset
from scxpand.data_util.prepare_data_for_train import (
    TrainingDataBundle,
    prepare_data_for_training,
)
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.model_manager import ModelManager
from scxpand.util.classes import DataAugmentParams, DataLoaderParams
from scxpand.util.general_util import (
    decisions_to_probabilities,
    flatten_nested_dict,
    log_nested_metrics,
    save_params,
    set_seed,
    time_seconds_to_str,
)
from scxpand.util.logger import get_logger
from scxpand.util.metrics import calculate_metrics
from scxpand.util.model_constants import SKLEARN_MODEL_FILE
from scxpand.util.train_logger import TrainLogger
from scxpand.util.train_util import report_to_optuna_and_handle_pruning


logger = get_logger()


class LinearBatchPredictor:
    """Handles batch prediction for linear models."""

    def __init__(self, dataset: CellsDataset, dataloader: DataLoader):
        self.dataset = dataset
        self.dataloader = dataloader
        # Cache model properties per model instance
        self._model_properties_cache = {}

    def _get_model_properties(self, model: SGDClassifier) -> tuple[bool, bool]:
        """Cache and return model properties to avoid repeated attribute checks."""
        # Use model id as cache key
        model_id = id(model)
        if model_id not in self._model_properties_cache:
            has_predict_proba = hasattr(model, "predict_proba") and callable(model.predict_proba)
            is_hinge_loss = getattr(model, "loss", None) == "hinge"
            self._model_properties_cache[model_id] = (has_predict_proba, is_hinge_loss)
        return self._model_properties_cache[model_id]

    def predict_batch(self, model: SGDClassifier, X_batch: np.ndarray) -> np.ndarray:
        """Predict probabilities for a single batch."""
        has_predict_proba, is_hinge_loss = self._get_model_properties(model)

        # For SVM (hinge loss) or models without predict_proba, convert decision
        # scores to probabilities using shared utility.
        if is_hinge_loss or not has_predict_proba:
            decisions = model.decision_function(X_batch)
            return decisions_to_probabilities(decisions)
        else:
            return model.predict_proba(X_batch)[:, 1]

    def predict_all(self, model: SGDClassifier) -> np.ndarray:
        """Predict probabilities for all samples in the dataset."""
        all_predictions = []

        for batch in self.dataloader:
            X_batch = batch["x"].numpy()  # Convert from torch tensor to numpy
            batch_predictions = self.predict_batch(model, X_batch)
            all_predictions.append(batch_predictions)

        return np.concatenate(all_predictions)


class LinearTrainLogger(TrainLogger):
    """Specialized logger for linear model training that extends the existing TrainLogger."""

    def __init__(self, base_save_dir: Path, trial: optuna.Trial | None = None):
        super().__init__(save_path=base_save_dir, trial=trial)

    def init_linear_training(self, n_epochs: int, n_batches_per_epoch: int) -> None:
        """Initialize training parameters for linear models."""
        # Adapt the existing init_writer method for linear models
        self.n_train_batches = n_batches_per_epoch
        self.n_epochs = n_epochs
        self.n_total_steps = n_epochs * n_batches_per_epoch
        self.best_model_score = None
        self.best_model_metrics = {}

        # Create a minimal writer setup for linear models (no tensorboard needed)
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Linear training initialized: {n_epochs} epochs, {n_batches_per_epoch} batches per epoch")
        logger.info(f"Total training steps: {self.n_total_steps}")

    def _log_scalars_simple(
        self,
        scalars: dict,
        group: str,
        global_step: int,
        epoch: int | None = None,
        i_batch: int | None = None,
    ):
        """Simplified version of add_scalars that doesn't require tensorboard writer."""
        completed_percent = 100 * global_step / self.n_total_steps
        total_seconds_elapsed = self.get_elapsed_time_in_seconds()

        if global_step > 0:
            seconds_per_step = total_seconds_elapsed / global_step
            total_expected_seconds = seconds_per_step * self.n_total_steps
            remaining_seconds = total_expected_seconds - total_seconds_elapsed
            remaining_seconds_str = time_seconds_to_str(remaining_seconds)
        else:
            remaining_seconds_str = "unknown"

        print_str = f"({completed_percent:.1f}%) ({group}),"
        if self.trial_number is not None:
            print_str = f"(trial {self.trial_number}) " + print_str
        if epoch is not None:
            print_str += f" epoch {epoch + 1}/{self.n_epochs},"
        if i_batch is not None:
            print_str += f" batch {i_batch + 1}/{self.n_train_batches},"

        # Add metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in scalars.items()])
        print_str += f" {metrics_str}"
        print_str += f", elapsed: {time_seconds_to_str(total_seconds_elapsed)}, remaining: {remaining_seconds_str}"

        logger.info(print_str)

    def log_epoch_end(self, epoch: int) -> None:
        """Log end of epoch with timing information."""
        total_seconds_elapsed = self.get_elapsed_time_in_seconds()
        completed_percent = 100 * (epoch + 1) / self.n_epochs

        if epoch > 0:
            avg_time_per_epoch = total_seconds_elapsed / (epoch + 1)
            remaining_epochs = self.n_epochs - (epoch + 1)
            estimated_remaining = avg_time_per_epoch * remaining_epochs
            remaining_str = time_seconds_to_str(estimated_remaining)
        else:
            remaining_str = "unknown"

        logger.info(
            f"Epoch {epoch + 1}/{self.n_epochs} complete ({completed_percent:.1f}%) - "
            f"elapsed: {time_seconds_to_str(total_seconds_elapsed)}, "
            f"estimated remaining: {remaining_str}"
        )

    def log_validation_metrics(self, epoch: int, dev_set_metrics: dict, score_metric: str) -> None:
        """Log validation metrics with hierarchical display."""
        log_nested_metrics(
            metrics=dev_set_metrics,
            logger_func=logger.info,
            group="validation",
            score_metric=score_metric,
            epoch=epoch,
        )

    def update_best_score(self, score: float, epoch: int, metrics: dict) -> None:
        """Update the best model score and metrics."""
        self.best_model_score = score
        self.best_model_epoch = epoch
        self.best_model_metrics = metrics

    def log_training_summary(self) -> None:
        """Log training completion summary using existing infrastructure."""
        # Ensure we have a valid best_model_score before calling save_final_summary
        if self.best_model_score is None:
            self.best_model_score = 0.0
            self.best_model_epoch = 0
        self.save_final_summary()


class TrainingSession:
    """Manages a single training session with state tracking."""

    def __init__(self, prm: LinearClassifierParam, score_metric: str):
        self.prm = prm
        self.score_metric = score_metric
        self.classes = np.array([0, 1])
        self.best_score = float("-inf")
        self.best_model_state = None
        self.patience_counter = 0

    def check_early_stopping(self, current_score: float, epoch: int) -> bool:
        """Check if early stopping should be triggered and update patience counter."""
        is_new_best = current_score > self.best_score

        if is_new_best:
            self.patience_counter = 0
            logger.info(f"New best score: {current_score:.4f} at epoch {epoch + 1}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter} evaluation intervals")

        should_stop = self.patience_counter >= self.prm.early_stopping_patience
        if should_stop:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")

        return should_stop

    def update_best_model(
        self,
        model: SGDClassifier,
        current_score: float,
        epoch: int,
        dev_set_metrics: dict,
        logger: LinearTrainLogger,
    ) -> None:
        """Update the best model state if current score is better."""
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_model_state = ModelManager.save_model_state(
                model=model,
                current_score=current_score,
                epoch=epoch,
                dev_set_metrics=dev_set_metrics,
            )
            # Update the logger's best score tracking
            logger.update_best_score(current_score, epoch, dev_set_metrics)


class LinearTrainer:
    """Consolidated linear model trainer with data preparation, training, and evaluation."""

    def __init__(self, prm: LinearClassifierParam, base_save_dir: Path):
        self.prm = prm
        self.base_save_dir = base_save_dir

    def _prepare_datasets_and_dataloaders(
        self,
        data_bundle: TrainingDataBundle,
        data_path: str | Path,
        num_workers: int = 0,
    ) -> tuple[CellsDataset, DataLoader, CellsDataset, DataLoader]:
        """Prepare datasets and dataloaders for training and validation."""
        # Create dataset params using linear model parameters
        dataset_params = DataAugmentParams(
            mask_rate=self.prm.mask_rate,
            noise_std=self.prm.noise_std,
            soft_loss_beta=self.prm.soft_loss_beta,
        )

        # Create train dataset and dataloader
        train_dataset = CellsDataset(
            data_format=data_bundle.data_format,
            row_inds=data_bundle.row_inds_train,
            dataset_params=dataset_params,
            is_train=True,
            data_path=data_path,
        )

        train_loader_params = DataLoaderParams(
            batch_size=self.prm.batch_size,
            shuffle=True,
            sampler_type=self.prm.sampler_type,
        )

        train_dataloader = create_train_dataloader(
            train_dataset=train_dataset,
            loader_params=train_loader_params,
            num_workers=num_workers,
        )

        # Create dev dataset and dataloader
        dev_dataset = CellsDataset(
            data_format=data_bundle.data_format,
            row_inds=data_bundle.row_inds_dev,
            dataset_params=dataset_params,
            is_train=False,
            data_path=data_path,
        )

        dev_dataloader = create_eval_dataloader(
            dataset=dev_dataset,
            batch_size=self.prm.batch_size,
            num_workers=num_workers,
        )

        logger.info(f"Created train dataset with {len(train_dataset)} samples")
        logger.info(f"Created dev dataset with {len(dev_dataset)} samples")

        return train_dataset, train_dataloader, dev_dataset, dev_dataloader

    def prepare_data_and_model(
        self,
        dev_ratio: float,
        data_path: str | None,
        num_workers: int = 0,
    ) -> tuple[SGDClassifier, CellsDataset, DataLoader, CellsDataset, DataLoader]:
        """Prepare data and initialize model for training."""
        set_seed(self.prm.random_seed)
        save_params(params=self.prm, save_dir=self.base_save_dir)

        logger.info(f"Running SGDClassifier with {self.prm.model_type} and parameters: {self.prm}")

        # Use prepare_data_for_training for memory efficient batch loading
        data_bundle = prepare_data_for_training(
            data_path=data_path,
            use_log_transform=self.prm.use_log_transform,
            save_dir=self.base_save_dir,
            dev_ratio=dev_ratio,
            rand_seed=self.prm.random_seed,
            resume=False,
        )

        logger.info(f"Training set size: {len(data_bundle.row_inds_train)} samples")
        logger.info(f"Validation set size: {len(data_bundle.row_inds_dev)} samples")

        # Prepare datasets and dataloaders
        train_dataset, train_dataloader, dev_dataset, dev_dataloader = self._prepare_datasets_and_dataloaders(
            data_bundle=data_bundle,
            data_path=data_path,
            num_workers=num_workers,
        )

        # Get y labels for model initialization
        y_train = train_dataset.y.numpy()

        # Initialize model
        model = ModelManager.initialize_model(prm=self.prm, y_train=y_train)

        return model, train_dataset, train_dataloader, dev_dataset, dev_dataloader

    def train_epoch(
        self,
        model: SGDClassifier,
        train_dataloader: DataLoader,
        train_logger: LinearTrainLogger,
        classes: np.ndarray,
        epoch: int,
    ) -> None:
        """Train the model for one epoch using DataLoader following scikit-learn SGD best practices."""
        n_batches = len(train_dataloader)
        logger.info(f"Training epoch {epoch + 1} with {n_batches} batches")

        batch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
            X_batch = batch["x"].numpy()
            y_batch = batch["y"].numpy()

            # Train on this batch
            model.partial_fit(X=X_batch, y=y_batch, classes=classes)
            batch_count += 1

            # Log progress
            if (batch_idx + 1) % self.prm.train_log_interval == 0 or (batch_idx + 1) == n_batches:
                train_logger._log_scalars_simple(
                    scalars={"batch_size": len(y_batch)},
                    group="train",
                    global_step=epoch * n_batches + batch_idx + 1,
                    epoch=epoch,
                    i_batch=batch_idx,
                )

        # Log only at epoch level (following scikit-learn SGD best practices)
        logger.info(f"Completed epoch {epoch + 1}: processed {batch_count} batches")
        train_logger.log_epoch_end(epoch=epoch)

    def evaluate_model(
        self,
        model: SGDClassifier,
        eval_dataset: CellsDataset,
        eval_dataloader: DataLoader,
        train_logger: LinearTrainLogger,
        score_metric: str,
        epoch: int,
    ) -> tuple[float, dict, np.ndarray]:
        """Evaluate model on validation set using DataLoader."""
        logger.info("Evaluating on validation set...")

        # Get predictions for all samples using batch processing
        predictor = LinearBatchPredictor(dataset=eval_dataset, dataloader=eval_dataloader)
        y_pred_prob = predictor.predict_all(model=model)
        y_true = eval_dataset.y.numpy()

        # Calculate metrics
        dev_set_metrics = calculate_metrics(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            obs_df=eval_dataset.obs_df,
        )
        current_score = flatten_nested_dict(dev_set_metrics)[score_metric]

        # Log validation metrics
        train_logger.log_validation_metrics(epoch=epoch, dev_set_metrics=dev_set_metrics, score_metric=score_metric)

        return current_score, dev_set_metrics, y_pred_prob

    def finalize_training(
        self,
        model: SGDClassifier,
        eval_dataset: CellsDataset,
        eval_dataloader: DataLoader,
        train_logger: LinearTrainLogger,
        trial: optuna.Trial | None,
        score_metric: str,
    ) -> dict:
        """Finalize training by evaluating and saving the final model."""
        logger.info("Predicting on validation set for final evaluation...")

        # Get final predictions using batch processing
        predictor = LinearBatchPredictor(dataset=eval_dataset, dataloader=eval_dataloader)
        final_pred_prob = predictor.predict_all(model=model)

        # Use utility function for evaluation and saving
        results = evaluate_predictions_and_save(
            y_pred_prob=final_pred_prob,
            obs_df=eval_dataset.obs_df,
            model_type=self.prm.model_type,
            save_path=self.base_save_dir,
            eval_name="dev",
            score_metric=score_metric,
            trial=trial,
        )
        logger.info("Saving model...")
        joblib.dump(value=model, filename=self.base_save_dir / SKLEARN_MODEL_FILE)

        # Log training summary
        train_logger.log_training_summary()
        logger.info("Training completed")
        return results

    def run_training(
        self,
        dev_ratio: float = 0.2,
        trial: optuna.Trial | None = None,
        score_metric: str = "harmonic_avg/AUROC",
        data_path: str | None = None,
        num_workers: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Run the complete training process."""
        # Prepare data and initialize components
        model, train_dataset, train_dataloader, dev_dataset, dev_dataloader = self.prepare_data_and_model(
            dev_ratio=dev_ratio,
            data_path=data_path,
            num_workers=num_workers,
        )

        # Initialize logging and training components
        train_logger = LinearTrainLogger(base_save_dir=self.base_save_dir, trial=trial)
        n_batches_per_epoch = len(train_dataloader)
        train_logger.init_linear_training(
            n_epochs=self.prm.n_epochs,
            n_batches_per_epoch=n_batches_per_epoch,
        )

        session = TrainingSession(prm=self.prm, score_metric=score_metric)

        logger.info(
            f"Starting training for {self.prm.n_epochs} epochs with early stopping patience of {self.prm.early_stopping_patience}"
        )

        # Training loop
        for i_epoch in range(self.prm.n_epochs):
            logger.info(f"Epoch {i_epoch + 1} of {self.prm.n_epochs}")

            # Train for one epoch
            self.train_epoch(
                model=model,
                train_dataloader=train_dataloader,
                train_logger=train_logger,
                classes=session.classes,
                epoch=i_epoch,
            )

            # Evaluate on dev set every eval_interval epochs
            if (i_epoch + 1) % self.prm.eval_interval == 0:
                current_score, dev_set_metrics, _ = self.evaluate_model(
                    model=model,
                    eval_dataset=dev_dataset,
                    eval_dataloader=dev_dataloader,
                    train_logger=train_logger,
                    score_metric=score_metric,
                    epoch=i_epoch,
                )

                logger.info(f"Epoch {i_epoch + 1}: {score_metric} = {current_score:.4f}")

                # Report to Optuna
                report_to_optuna_and_handle_pruning(
                    trial=trial,
                    current_score=current_score,
                    epoch=i_epoch,
                )

                # Check early stopping BEFORE updating best model
                if session.check_early_stopping(current_score=current_score, epoch=i_epoch):
                    break

                # Update best model after early stopping check
                session.update_best_model(
                    model=model,
                    current_score=current_score,
                    epoch=i_epoch,
                    dev_set_metrics=dev_set_metrics,
                    logger=train_logger,
                )

        # Load best model state if we found one
        if session.best_model_state is not None:
            ModelManager.load_model_state(
                model=model,
                best_model_state=session.best_model_state,
                base_save_dir=self.base_save_dir,
                score_metric=score_metric,
            )
        else:
            logger.info("No best model found, using final model state")

        # Finalize training and save results
        results = self.finalize_training(
            model=model,
            eval_dataset=dev_dataset,
            eval_dataloader=dev_dataloader,
            train_logger=train_logger,
            trial=trial,
            score_metric=score_metric,
        )

        return results


def run_linear_training(
    base_save_dir: str | Path,
    prm: LinearClassifierParam,
    data_path: str | Path,
    dev_ratio: float = 0.2,
    trial: optuna.Trial | None = None,
    score_metric: str = "harmonic_avg/AUROC",
    num_workers: int = 0,
    resume: bool = False,  # noqa: ARG001
) -> dict[str, dict[str, float]]:
    """Run SGDClassifier model training and evaluation with support for logistic regression and SVM.

    Note: Linear models don't support resuming from checkpoints like PyTorch models.
    The resume parameter is accepted for API compatibility but not implemented.
    """
    base_save_dir = Path(base_save_dir)

    trainer = LinearTrainer(prm=prm, base_save_dir=base_save_dir)
    return trainer.run_training(
        dev_ratio=dev_ratio,
        trial=trial,
        score_metric=score_metric,
        data_path=data_path,
        num_workers=num_workers,
    )
