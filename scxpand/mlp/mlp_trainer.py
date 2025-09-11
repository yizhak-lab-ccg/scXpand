import time

from pathlib import Path

import anndata as ad
import numpy as np
import optuna
import torch
import torch.nn.functional as F

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataloaders import create_eval_dataloader, create_train_dataloader
from scxpand.data_util.dataset import (
    CellsDataset,
)
from scxpand.mlp.mlp_losses import compute_batch_loss, create_loss_function
from scxpand.mlp.mlp_model import MLPModel
from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.general_util import (
    compute_error_rate,
    compute_false_negative_rate,
    compute_false_positive_rate,
    flatten_nested_dict,
    log_inference_progress,
    log_nested_metrics,
    to_np,
)
from scxpand.util.logger import get_logger
from scxpand.util.metrics import calculate_metrics
from scxpand.util.train_logger import TrainLogger, has_checkpoint
from scxpand.util.train_util import (
    check_early_stopping,
    get_lr_scheduler,
    get_optimizer,
    report_to_optuna_and_handle_pruning,
    update_lr_scheduler,
)


logger = get_logger()


def run_trainer(
    data_path: str | Path,
    data_format: DataFormat,
    row_inds_train: np.ndarray,
    row_inds_dev: np.ndarray,
    save_path: Path,
    prm: MLPParam,
    device: str,
    trial: optuna.Trial | None = None,
    score_metric: str = "harmonic_avg/AUROC",
    resume: bool = False,
    num_workers: int = 0,
):
    """Runs the training loop for the multi-layer perceptron (MLP) model.

    Args:
        data_path: Path to the AnnData file
        data_format: DataFormat object containing preprocessing parameters
        row_inds_train: cell indices of the training data (in the full dataset)
        row_inds_dev: cell indices of the validation data (in the full dataset)
        save_path: path to save results
        prm: MLPParam object containing model parameters
        device: Device to train on ('cuda', 'mps', or 'cpu')
        trial: Optuna trial for hyperparameter optimization
        score_metric: Metric to use for model selection
        resume: whether to resume training from a checkpoint
        num_workers: Number of worker processes for data loading
    """
    # Extract dataset params
    dataset_params = prm.get_dataset_params()

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CellsDataset(
        data_format=data_format,
        row_inds=row_inds_train,
        dataset_params=dataset_params,
        is_train=True,
        data_path=data_path,
    )
    dev_dataset = CellsDataset(
        data_format=data_format,
        row_inds=row_inds_dev,
        dataset_params=dataset_params,
        is_train=False,
        data_path=data_path,
    )

    train_loader = create_train_dataloader(
        train_dataset=train_dataset,
        loader_params=prm.get_data_loader_params(),
        num_workers=num_workers,
    )

    n_epochs = prm.n_epochs

    model = MLPModel(
        prm=prm,
        device=device,
        data_format=data_format,
    ).to(device)

    optimizer = get_optimizer(model=model, optimizer_params=prm.get_optimizer_params())

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler_params=prm.get_lr_scheduler_params(),
        n_epochs=n_epochs,
        train_loader=train_loader,
        init_learning_rate=prm.init_learning_rate,
    )

    log_manager = TrainLogger(save_path=save_path, trial=trial)
    log_manager.init_writer(n_epochs=n_epochs, n_train_batches=len(train_loader), prm=prm)

    # Load checkpoint if resuming
    resume_epoch = 0
    best_val_score = float("-inf")
    if has_checkpoint(save_path) and resume:
        resume_epoch = log_manager.resume_from_checkpoint(
            resume_exp_path=save_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_name=device,
        )
        # Load best score from checkpoint info
        best_val_score = log_manager.best_model_score
        logger.info(f"Resumed from epoch {resume_epoch} with best score {best_val_score}")

    # Loss function
    loss_fn = create_loss_function(prm=prm, device=device)

    # Early stopping parameters
    patience_counter = 0

    logger.info(f"Training model from epoch {resume_epoch} to {n_epochs}")
    logger.info(f"Using device: {device}")
    i_step = 0
    i_epoch = 0

    for i_epoch in range(resume_epoch, n_epochs):
        logger.info(f"Epoch {i_epoch} of {n_epochs}")
        model.train()

        for i_batch, batch in enumerate(train_loader):
            x = batch["x"].to(device)  # [B, n_features]
            y_soft_gt = batch["y_soft"].to(device)  # [B] in range [0, 1]
            y_gt = batch["y"].to(device)  # [B] in {0, 1}

            # Get categorical targets if available
            categorical_targets = batch.get("categorical_targets", None)
            if categorical_targets is not None:
                categorical_targets = {k: v.to(device) for k, v in categorical_targets.items()}

            # Forward pass to get the predicted logits
            model_output = model.forward(x)
            main_logit = model_output.main_logit  # [batch_size]
            categorical_logits = model_output.categorical_logits  # dict[feature_name, logits[batch_size, n_classes]]

            prob_pred = to_np(F.sigmoid(main_logit))

            loss, bin_cls_loss, cat_loss = compute_batch_loss(
                main_logit=main_logit,
                y_soft_gt=y_soft_gt,
                y_gt=y_gt,
                categorical_logits=categorical_logits,
                categorical_targets=categorical_targets,
                loss_fn=loss_fn,
                prm=prm,
                epoch=i_epoch,
            )

            # Backward pass
            loss.backward()
            if prm.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), prm.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # Compute metrics and log
            label = to_np(batch["y"]).squeeze()  # [B]
            i_step += 1

            if (i_step + 1) % prm.train_log_interval == 0:
                train_metrics = {
                    "loss": loss.item(),
                    "bin_cls_loss": bin_cls_loss.item(),
                    "cat_loss": cat_loss.item(),
                    "err": compute_error_rate(label=label, y_pred=prob_pred),
                    "fp": compute_false_positive_rate(label=label, prob_out=prob_pred),
                    "fn": compute_false_negative_rate(label=label, prob_out=prob_pred),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": i_epoch,
                    "batch": i_batch,
                }
                log_manager.add_scalars(
                    scalars=train_metrics,
                    group="train",
                    global_step=i_step,
                    epoch=i_epoch,
                    i_batch=i_batch,
                    one_line=True,
                )

        # Evaluate on validation set
        y_dev_pred_prob = run_mlp_inference(
            data_path=data_path,
            data_format=data_format,
            eval_row_inds=row_inds_dev,
            model=model,
            device=device,
            batch_size=prm.inference_batch_size,
            num_workers=num_workers,
        )
        dev_set_metrics = calculate_metrics(
            y_true=dev_dataset.y,
            y_pred_prob=y_dev_pred_prob,
            obs_df=dev_dataset.obs_df,
        )
        current_score = flatten_nested_dict(dev_set_metrics)[score_metric]

        report_to_optuna_and_handle_pruning(
            trial=trial,
            current_score=current_score,
            epoch=i_epoch,
        )

        # Log validation metrics with hierarchical display
        log_nested_metrics(
            metrics=dev_set_metrics,
            logger_func=logger.info,
            group="validation",
            score_metric=score_metric,
            epoch=i_epoch,
        )
        # Also log to tensorboard for visualization
        log_manager.add_scalars(scalars=dev_set_metrics, group="dev", global_step=i_step, epoch=i_epoch)

        # Early stopping check
        patience_counter, should_stop = check_early_stopping(
            current_score=current_score,
            log_manager=log_manager,
            patience_counter=patience_counter,
            patience_limit=prm.early_stopping_patience,
            epoch=i_epoch,
        )
        if should_stop:
            break

        # Save latest checkpoint
        log_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=i_epoch,
            model_score=current_score,
            dev_set_metrics=dev_set_metrics,
        )

        # Update the learning rate scheduler
        update_lr_scheduler(
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=prm.get_lr_scheduler_params(),
            score=current_score,
        )

        logger.info(f"End of epoch {i_epoch}")

    # Load best model state if early stopping was triggered
    model = log_manager.load_best_model(model=model, device_name=device)

    log_manager.save_final_summary()
    logger.info("Training completed")

    return model


def run_mlp_inference(
    data_path: str | Path | None = None,
    data_format: DataFormat = None,
    eval_row_inds: np.ndarray | None = None,
    model: MLPModel = None,
    device: str | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
    adata: ad.AnnData | None = None,
) -> np.ndarray:
    """Run inference using a trained MLP model.

    Returns:
        pred_prob: np.ndarray, shape [N] (predicted probabilities of the positive class)
    Accepts either data_path or adata (AnnData object).
    """
    # Create dataset for inference (no data augmentation)
    dataset = CellsDataset(
        data_format=data_format,
        row_inds=eval_row_inds,
        dataset_params=None,
        is_train=False,
        data_path=data_path,
        adata=adata,
    )

    data_loader = create_eval_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    model.eval()
    model.to(device)

    total_samples = len(dataset)
    total_batches = len(data_loader)
    pred_prob = torch.zeros(total_samples, device="cpu")

    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Get inputs and move to device
            x = batch["x"].to(device)

            # Run forward pass
            model_output = model.forward(x)
            output = model_output.main_logit

            # Store predictions
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            pred_prob[start_idx:end_idx] = torch.sigmoid(output).cpu()

            # Print progress using utility function
            log_inference_progress(
                current_iteration=i,
                total_iterations=total_batches,
                start_time=start_time,
                log_interval=20,
                logger_instance=logger,
            )

    pred_prob = pred_prob.numpy()
    return pred_prob
