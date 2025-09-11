import torch

from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.logger import get_logger


logger = get_logger()


def create_loss_function(*, prm: MLPParam, device: str) -> torch.nn.Module:
    """Create and return the binary classification loss function."""
    pos_weight = torch.tensor([prm.positives_weight], device=device)
    return torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)


def should_use_soft_loss(*, epoch: int, prm: MLPParam) -> bool:
    """Check if the soft loss should be used at the given epoch."""
    return (prm.soft_loss_start_epoch is not None) and (epoch >= prm.soft_loss_start_epoch)


def compute_batch_loss(
    *,
    main_logit: torch.Tensor,
    y_soft_gt: torch.Tensor,
    y_gt: torch.Tensor,
    categorical_logits: dict[str, torch.Tensor] | None,
    categorical_targets: dict[str, torch.Tensor] | None,
    loss_fn: torch.nn.Module,
    prm: MLPParam,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete batch loss including target selection and loss computation.

    Args:
        main_logit: Binary classification logits, shape [batch_size]
        y_soft_gt: Soft binary labels, shape [batch_size] in range [0, 1]
        y_gt: Hard binary labels, shape [batch_size] in {0, 1}
        categorical_logits: Dict of logits for categorical features, each shape [batch_size, n_classes]
        categorical_targets: Dict of target indices for categorical features, each shape [batch_size]
        loss_fn: Binary classification loss function (e.g., BCEWithLogitsLoss)
        prm: NNParam containing all loss weights
        epoch: Current training epoch

    Returns:
        total_loss: torch.Tensor, sum of binary classification and categorical losses
        bin_cls_loss: torch.Tensor, binary classification loss
        cat_loss: torch.Tensor, sum of categorical classification losses
    """
    device = main_logit.device

    # Determine target based on soft loss schedule
    target = y_soft_gt if should_use_soft_loss(epoch=epoch, prm=prm) else y_gt

    if categorical_logits is not None and categorical_targets is not None:
        # Use the comprehensive loss function that includes categorical loss
        return compute_total_nn_loss(
            main_logit=main_logit,
            y_true=target,
            main_cls_loss_fn=lambda logits, targets: torch.mean(loss_fn(input=logits, target=targets)),
            prm=prm,
            categorical_logits=categorical_logits,
            categorical_targets=categorical_targets,
        )
    else:
        # Standard loss computation
        losses = loss_fn(input=main_logit, target=target)
        loss = torch.mean(losses)
        bin_cls_loss = loss
        cat_loss = torch.tensor(0.0, device=device)
        return loss, bin_cls_loss, cat_loss


def compute_total_nn_loss(
    *,
    main_logit: torch.Tensor,
    y_true: torch.Tensor,
    main_cls_loss_fn: torch.nn.Module,
    prm: MLPParam,
    categorical_logits: dict[str, torch.Tensor] | None = None,
    categorical_targets: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the total loss as the sum of binary classification loss and categorical losses.

    Args:
        main_logit: Binary classification logits, shape [batch_size]
        y_true: True binary labels, shape [batch_size]
        main_cls_loss_fn: Binary classification loss function (e.g., BCEWithLogitsLoss)
        prm: NNParam containing all loss weights
        categorical_logits: Dict of logits for categorical features, each shape [batch_size, n_classes]
        categorical_targets: Dict of target indices for categorical features, each shape [batch_size]

    Returns:
        total_loss: torch.Tensor, sum of binary classification and categorical losses
        bin_cls_loss: torch.Tensor, binary classification loss
        cat_loss: torch.Tensor, sum of categorical classification losses
    """
    device = main_logit.device

    # --- Binary classification loss ---
    main_cls_loss = main_cls_loss_fn(main_logit, y_true)

    # --- Categorical loss ---
    cat_loss = torch.tensor(0.0, device=device)
    has_categorical = (
        categorical_logits is not None
        and categorical_targets is not None
        and len(categorical_logits) > 0
        and len(categorical_targets) > 0
    )

    if has_categorical:
        cross_entropy = torch.nn.CrossEntropyLoss()
        for feature_name, logits in categorical_logits.items():
            if feature_name in categorical_targets:
                targets = categorical_targets[feature_name].to(device)
                n_classes = logits.shape[1]

                # Handle invalid target indices
                if torch.any(targets >= n_classes) or torch.any(targets < 0):
                    invalid_indices = torch.logical_or(targets >= n_classes, targets < 0)
                    targets = torch.where(invalid_indices, torch.zeros_like(targets), targets)

                feature_loss = cross_entropy(logits, targets)
                cat_loss = cat_loss + feature_loss

    # --- Total loss ---
    total_loss = main_cls_loss + prm.cat_loss_weight * cat_loss

    return total_loss, main_cls_loss, cat_loss
