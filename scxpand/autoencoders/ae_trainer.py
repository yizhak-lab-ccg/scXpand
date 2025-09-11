import time

from pathlib import Path

import anndata as ad
import numpy as np
import optuna
import torch

from scxpand.autoencoders.ae_losses import (
    compute_total_autoencoder_loss,
    create_autoencoder_recon_loss_function,
)
from scxpand.autoencoders.ae_models import create_ae_model
from scxpand.autoencoders.ae_modules import BaseAutoencoder
from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataloaders import create_eval_dataloader, create_train_dataloader
from scxpand.data_util.dataset import (
    CellsDataset,
)
from scxpand.util.general_util import (
    get_device,
    log_inference_progress,
    log_nested_metrics,
)
from scxpand.util.logger import get_logger
from scxpand.util.metrics import calculate_metrics, get_score_from_nested_dict
from scxpand.util.train_logger import TrainLogger, has_checkpoint
from scxpand.util.train_util import (
    check_early_stopping,
    get_lr_scheduler,
    get_optimizer,
    report_to_optuna_and_handle_pruning,
    update_lr_scheduler,
)


logger = get_logger()


def calculate_training_metrics(
    prob_pred: np.ndarray,
    y_true: np.ndarray,
    loss_outputs: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer,
    prm: AutoEncoderParams,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate training metrics for logging.

    Args:
        prob_pred: Predicted probabilities from model output
        y_true: True binary labels
        loss_outputs: Tuple containing (loss, recon_loss, cls_loss, l1_loss, cat_loss)
        optimizer: Optimizer to get learning rate from
        prm: AutoEncoder parameters to check for categorical losses
        threshold: Classification threshold for binary predictions

    Returns:
        Dictionary containing calculated metrics
    """
    loss, recon_loss, cls_loss, l1_loss, cat_loss = loss_outputs

    # Calculate classification metrics
    err = ((prob_pred > threshold) != y_true).mean()
    fp = ((prob_pred > threshold) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
    fn = ((prob_pred <= threshold) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())

    metrics = {
        "loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "cls_loss": cls_loss.item(),
        "l1_loss": l1_loss.item(),
        "err": err,
        "fp": fp,
        "fn": fn,
        "lr": optimizer.param_groups[0]["lr"],
    }

    # Add categorical loss if applicable
    if prm.aux_categorical_types:
        metrics["cat_loss"] = cat_loss.item()

    return metrics


def run_ae_trainer(
    data_path: str | Path | None,
    data_format: DataFormat,
    row_inds_train: np.ndarray,
    row_inds_dev: np.ndarray,
    save_path: Path,
    prm: AutoEncoderParams,
    device: str,
    trial: optuna.Trial | None = None,
    score_metric: str = "harmonic_avg/AUROC",
    resume: bool = False,
    num_workers: int = 0,
) -> BaseAutoencoder:
    logger.info("Creating datasets...")
    train_dataset = CellsDataset(
        data_format=data_format,
        row_inds=row_inds_train,
        dataset_params=prm.get_dataset_params(),
        is_train=True,
        data_path=data_path,
        include_row_normalized_gene_counts=True,
    )
    dev_dataset = CellsDataset(
        data_format=data_format,
        row_inds=row_inds_dev,
        dataset_params=prm.get_dataset_params(),
        is_train=False,
        data_path=data_path,
        include_row_normalized_gene_counts=False,
    )

    train_loader = create_train_dataloader(
        train_dataset=train_dataset,
        loader_params=prm.get_data_loader_params(),
        num_workers=num_workers,
    )

    model = create_ae_model(data_format=data_format, prm=prm, device=device)

    optimizer = get_optimizer(
        model=model,
        optimizer_params=prm.get_optimizer_params(),
    )

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler_params=prm.get_lr_scheduler_params(),
        n_epochs=prm.n_epochs,
        train_loader=train_loader,
        init_learning_rate=prm.init_learning_rate,
    )

    recon_loss_fn = create_autoencoder_recon_loss_function(prm=prm)

    log_manager = TrainLogger(save_path=save_path, trial=trial)
    log_manager.init_writer(n_epochs=prm.n_epochs, n_train_batches=len(train_loader), prm=prm)

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
        best_val_score = log_manager.best_model_score
        logger.info(f"Resumed from epoch {resume_epoch} with best score {best_val_score}")

    bce_loss = torch.nn.BCEWithLogitsLoss().to(device)

    patience_counter = 0  # Early stopping parameter

    logger.info(f"Training model from epoch {resume_epoch} to {prm.n_epochs}")
    logger.info(f"Using device: {device}")
    logger.info(
        f"Loss weights: recon={prm.recon_loss_weight:.2f}, cls={prm.cls_loss_weight:.2f}, cat={prm.cat_loss_weight:.2f}"
    )
    i_step = 0

    for epoch in range(resume_epoch, prm.n_epochs):
        model.train()
        for i_batch, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            y = batch["y"].to(device).float()
            x_row_normalized_gene_counts = batch["x_row_normalized_gene_counts"].to(device)

            y_soft_gt = None
            if "y_soft" in batch:
                y_soft_gt = batch["y_soft"].to(device).float()

            # Forward pass
            outputs = model.forward(x)

            categorical_targets = None
            if "categorical_targets" in batch:
                categorical_targets = {k: v.to(device) for k, v in batch["categorical_targets"].items()}

            loss_outputs = compute_total_autoencoder_loss(
                x_genes_true=x_row_normalized_gene_counts,
                mu=outputs.mu,
                pi=outputs.pi,
                theta=outputs.theta,
                latent_vec=outputs.latent_vec,
                class_logit=outputs.class_logit,
                y_true=y,
                y_soft_gt=y_soft_gt,
                recon_loss_fn=recon_loss_fn,
                bce_loss=bce_loss,
                prm=prm,
                epoch=epoch,
                categorical_logits=outputs.categorical_logits,
                categorical_targets=categorical_targets,
            )

            loss, recon_loss, cls_loss, l1_loss, cat_loss = loss_outputs

            # Backward pass
            loss.backward()
            if prm.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), prm.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # Logging
            i_step += 1
            if (i_step + 1) % prm.train_log_interval == 0:
                # Use the convenience method from ModelOutput to get predictions
                prob_pred = outputs.get_binary_predictions().detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                # Calculate training metrics using extracted function
                train_metrics = calculate_training_metrics(
                    prob_pred=prob_pred,
                    y_true=y_np,
                    loss_outputs=loss_outputs,
                    optimizer=optimizer,
                    prm=prm,
                    threshold=0.5,
                )

                log_manager.add_scalars(
                    scalars=train_metrics,
                    group="train",
                    global_step=i_step,
                    epoch=epoch,
                    i_batch=i_batch,
                    one_line=True,
                )

        # Evaluate on validation set
        y_dev_pred_prob = run_ae_inference(
            model=model,
            batch_size=prm.inference_batch_size,
            data_path=data_path,
            data_format=data_format,
            eval_row_inds=row_inds_dev,
            device=device,
            num_workers=num_workers,
        )

        dev_set_metrics = calculate_metrics(
            y_true=dev_dataset.y,
            y_pred_prob=y_dev_pred_prob,
            obs_df=dev_dataset.obs_df,
        )

        current_score = get_score_from_nested_dict(nested_metrics_dict=dev_set_metrics, metric_name=score_metric)

        report_to_optuna_and_handle_pruning(
            trial=trial,
            current_score=current_score,
            epoch=epoch,
        )

        # Log validation metrics with table display
        log_nested_metrics(
            metrics=dev_set_metrics,
            logger_func=logger.info,
            group="validation",
            score_metric=score_metric,
            epoch=epoch,
        )
        # Also log to tensorboard for visualization
        log_manager.add_scalars(scalars=dev_set_metrics, group="dev", global_step=i_step, epoch=epoch)

        # Early stopping check
        patience_counter, should_stop = check_early_stopping(
            current_score=current_score,
            log_manager=log_manager,
            patience_counter=patience_counter,
            patience_limit=prm.early_stopping_patience,
            epoch=epoch,
        )
        if should_stop:
            break

        # Save checkpoint
        log_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            model_score=current_score,
            dev_set_metrics=dev_set_metrics,
        )

        update_lr_scheduler(
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=prm.get_lr_scheduler_params(),
            score=current_score,
        )
        logger.info(f"End of epoch {epoch}")

    # Load best model for final evaluation
    model = log_manager.load_best_model(model=model, device_name=device)
    log_manager.save_final_summary()
    logger.info("Training completed")

    return model


def run_ae_inference(
    model: BaseAutoencoder,
    batch_size: int,  # No default - must be specified
    data_path: str | Path | None = None,
    data_format: DataFormat = None,
    eval_row_inds: np.ndarray | None = None,
    device: str | None = None,
    num_workers: int = 0,
    adata: ad.AnnData = None,
) -> np.ndarray:
    """Runs autoencoder inference. Accepts either data_path or adata (AnnData object)."""
    device = device if device else get_device()

    dataset = CellsDataset(
        data_format=data_format,
        row_inds=eval_row_inds,
        is_train=False,
        data_path=data_path,
        adata=adata,
        include_row_normalized_gene_counts=False,
    )

    dataloader = create_eval_dataloader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    model.eval()
    model.to(device)
    total_samples = len(dataset)
    pred_prob = torch.zeros(total_samples, device="cpu")
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Get data
            x = batch["x"].to(device)

            # Forward pass
            model_output = model.forward(x)

            # Get class logits and apply sigmoid
            logits = model_output.class_logit

            # Apply sigmoid to get probabilities
            batch_probs = torch.sigmoid(logits).cpu()

            # Final check for NaNs in probabilities
            if torch.isnan(batch_probs).any() or torch.isinf(batch_probs).any():
                # Use neutral probability for invalid values
                neutral_prob = 0.5  # Neutral probability for binary classification
                logger.warning(
                    f"NaN or Inf values detected in probabilities for batch {i}, replacing with {neutral_prob}"
                )
                batch_probs = torch.nan_to_num(batch_probs, nan=neutral_prob, posinf=1.0, neginf=0.0)

            # Store predictions
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            pred_prob[start_idx:end_idx] = batch_probs

            log_inference_progress(
                current_iteration=i,
                total_iterations=len(dataloader),
                start_time=start_time,
                log_interval=20,
                logger_instance=logger,
            )

    pred_prob = pred_prob.numpy()
    return pred_prob
