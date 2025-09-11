import optuna
import torch

from torch.utils.data import DataLoader

from scxpand.util.classes import (
    LRSchedulerParams,
    LRSchedulerType,
    OptimizerParams,
    OptimizerType,
)
from scxpand.util.logger import get_logger
from scxpand.util.train_logger import TrainLogger


logger = get_logger()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler_params: LRSchedulerParams,
    n_epochs: int,
    train_loader: DataLoader,
    init_learning_rate: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    lr_scheduler_config = lr_scheduler_params.lr_scheduler_config
    lr_scheduler_type = lr_scheduler_config["type"]
    if lr_scheduler_type == LRSchedulerType.ONE_CYCLE_LR:
        num_training_steps = n_epochs * len(train_loader)
        num_warmup_steps = num_training_steps * lr_scheduler_config["warmup_ratio"]
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=init_learning_rate,
            total_steps=num_training_steps,
            pct_start=num_warmup_steps / num_training_steps,
            anneal_strategy="cos",
        )
    elif lr_scheduler_type == LRSchedulerType.REDUCE_LR_ON_PLATEAU:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_scheduler_config["factor"],
            patience=lr_scheduler_config["patience"],
            min_lr=lr_scheduler_config["min_lr"],
        )
    elif lr_scheduler_type == LRSchedulerType.STEP_LR:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_config["step_size"],
            gamma=lr_scheduler_config["gamma"],
        )
    elif lr_scheduler_type == LRSchedulerType.COSINE_ANNEALING_LR:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_scheduler_config["T_max"],
            eta_min=lr_scheduler_config["eta_min"],
        )
    elif lr_scheduler_type == LRSchedulerType.CONSTANT_LR:
        # ConstantLR maintains the same learning rate throughout training
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=0,  # No decay
        )
    elif lr_scheduler_type == LRSchedulerType.NO_SCHEDULER:
        lr_scheduler = None
    else:
        raise ValueError(f"Unknown learning rate scheduler: {lr_scheduler_type}")
    return lr_scheduler


def get_optimizer(
    model: torch.nn.Module,
    optimizer_params: OptimizerParams,
) -> torch.optim.Optimizer:
    """Get an optimizer that applies weight decay selectively.

    Excludes LayerNorm weights and biases from weight decay following best practices.
    """
    # Parameters that should not have weight decay
    no_decay = ["bias", "LayerNorm.weight"]

    # Split parameters into two groups
    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]

    # Create optimizer groups with different weight decay settings
    optim_groups = [
        {"params": params_decay, "weight_decay": optimizer_params.weight_decay},
        {"params": params_no_decay, "weight_decay": 0.0},
    ]

    # Create optimizer based on config
    if optimizer_params.optimizer_type == OptimizerType.ADAMW:
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=optimizer_params.init_learning_rate,
            betas=optimizer_params.adam_betas,
        )
    elif optimizer_params.optimizer_type == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(
            optim_groups,
            lr=optimizer_params.init_learning_rate,
            betas=optimizer_params.adam_betas,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_params.optimizer_type}")

    return optimizer


def update_lr_scheduler(
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    lr_scheduler_params: LRSchedulerParams,
    score: float,
):
    """Update the learning rate scheduler."""
    lr_scheduler_type = lr_scheduler_params.lr_scheduler_type
    if lr_scheduler_type == LRSchedulerType.REDUCE_LR_ON_PLATEAU:
        lr_scheduler.step(score)
    elif lr_scheduler_type in (
        LRSchedulerType.STEP_LR,
        LRSchedulerType.COSINE_ANNEALING_LR,
        LRSchedulerType.CONSTANT_LR,
    ):
        lr_scheduler.step()
    elif lr_scheduler_type == LRSchedulerType.NO_SCHEDULER:
        pass


def check_early_stopping(
    current_score: float,
    log_manager: TrainLogger,
    patience_counter: int,
    patience_limit: int,
    epoch: int,
) -> tuple[int, bool]:
    if log_manager.best_model_score is None or current_score > log_manager.best_model_score:
        patience_counter = 0
        should_stop = False
    else:
        patience_counter += 1
        should_stop = patience_counter >= patience_limit
        if should_stop:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")

    return patience_counter, should_stop


def report_to_optuna_and_handle_pruning(
    trial: optuna.Trial | None,
    current_score: float,
    epoch: int,
) -> None:
    """Report current score to Optuna trial and handle pruning, with duplicate prevention.

    This function prevents duplicate epoch reports that can occur when resuming
    from checkpoints, ensuring the Optuna dashboard shows accurate progress.

    Args:
        trial: The Optuna trial object (or None).
        current_score: The current score to report.
        epoch: The current epoch number.

    Raises:
        optuna.TrialPruned: If the trial should be pruned.
    """
    if trial is None:
        return

    # Check for duplicate reports to prevent issues when resuming
    if _is_duplicate_epoch_report(trial, epoch):
        logger.debug(f"Skipping duplicate report for epoch {epoch} in trial {trial.number}")
    else:
        _report_epoch_score(trial, current_score, epoch)

    # Handle pruning regardless of whether we reported
    _handle_trial_pruning(trial, epoch)


def _is_duplicate_epoch_report(trial: optuna.Trial, epoch: int) -> bool:
    """Check if this epoch has already been reported for this trial."""
    reported_epochs_list = trial.user_attrs.get("reported_epochs", [])
    reported_epochs = set(reported_epochs_list)
    return epoch in reported_epochs


def _report_epoch_score(trial: optuna.Trial, current_score: float, epoch: int) -> None:
    """Report the epoch score and update tracking."""
    trial.report(value=current_score, step=epoch)

    # Update the set of reported epochs
    reported_epochs_list = trial.user_attrs.get("reported_epochs", [])
    reported_epochs = set(reported_epochs_list)
    reported_epochs.add(epoch)
    trial.set_user_attr("reported_epochs", list(reported_epochs))

    logger.info(f"Reported epoch {epoch} score {current_score:.4f} to Optuna trial {trial.number}")


def _handle_trial_pruning(trial: optuna.Trial, epoch: int) -> None:
    """Handle trial pruning if the trial should be pruned."""
    if trial.should_prune():
        logger.info(f"Trial {trial.number} was pruned at epoch {epoch}")
        raise optuna.TrialPruned
