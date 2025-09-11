import torch

from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.util.logger import get_logger


logger = get_logger()


def should_use_soft_loss(*, epoch: int, prm: AutoEncoderParams) -> bool:
    """Check if the soft loss should be used at the given epoch."""
    return (prm.soft_loss_start_epoch is not None) and (epoch >= prm.soft_loss_start_epoch)


class MSELoss:
    def __init__(
        self,
        eps: float = 1e-8,
    ):
        """Mean squared error loss.

        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps

    def _nan2zero(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN values with zeros."""
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)

    def mse_loss(
        self,
        x_genes_true: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.eps
        x_genes_true = x_genes_true.float()
        x_pred = x_pred.float()

        # Mask NaN values
        masking = torch.isnan(x_genes_true)
        nelem = torch.sum(~masking).float()
        x_genes_true = self._nan2zero(x_genes_true)

        # Calculate MSE
        mse = torch.square(x_genes_true - x_pred)

        if torch.any(masking):
            mse = torch.sum(mse) / (nelem + eps)
        else:
            mse = torch.mean(mse)
        return mse

    def __call__(
        self,
        x_genes_true: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        mse_loss = self.mse_loss(x_genes_true=x_genes_true, x_pred=x_pred)

        return mse_loss


def _compute_nb_log_likelihood(
    x_genes_true: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute negative binomial log-likelihood with enhanced stability.

    This is a shared function to ensure ZINB and NB use identical computation.

    Args:
        x_genes_true: Observed counts
        mu: Expected means (μ)
        theta: Dispersion parameter
        eps: Small constant for numerical stability

    Returns:
        Log-likelihood values
    """
    # Enhanced stability: clamp parameters to reasonable ranges
    mu = torch.clamp(mu, min=eps, max=1e6)  # Prevent extreme mu values
    theta = torch.clamp(theta, min=eps, max=1e4)  # Prevent extreme theta values
    x_genes_true = torch.clamp(x_genes_true, min=0.0, max=1e6)  # Ensure non-negative

    # Clamp inputs to lgamma to prevent overflow (more conservative)
    x_plus_theta = torch.clamp(x_genes_true + theta, max=1e3)
    theta_clamped = torch.clamp(theta, max=1e3)

    # Standard negative binomial log-likelihood
    t1 = torch.lgamma(x_plus_theta + eps) - torch.lgamma(x_genes_true + 1.0) - torch.lgamma(theta_clamped + eps)

    # Use stable log-space computation for the log-probabilities
    log_theta_mu_eps = torch.log(theta + mu + eps)
    t2 = theta * (torch.log(theta + eps) - log_theta_mu_eps)
    t3 = x_genes_true * (torch.log(mu + eps) - log_theta_mu_eps)

    # Log-likelihood
    log_likelihood = t1 + t2 + t3

    return log_likelihood


class ZINBLoss:
    def __init__(
        self,
        eps: float = 1e-8,
    ):
        """Zero-inflated negative binomial loss.

        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps

    def _nan2zero(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN values with zeros."""
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)

    def nb_loss(
        self,
        x_genes_true: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Negative binomial negative log-likelihood without zero-inflation.

        Args:
            x_genes_true: Observed counts
            mu: Expected means (μ)
            theta: Dispersion parameter

        Returns:
            Mean negative log-likelihood across batch
        """
        eps = self.eps
        x_genes_true = x_genes_true.float()

        # Mask NaN values
        masking = torch.isnan(x_genes_true) | torch.isinf(x_genes_true)
        nelem = torch.sum(~masking).float()
        x_genes_true = self._nan2zero(x_genes_true)

        # Use shared computation
        log_likelihood = _compute_nb_log_likelihood(x_genes_true=x_genes_true, mu=mu, theta=theta, eps=eps)

        # Negative log-likelihood (we want to minimize, so negate)
        final = -log_likelihood

        # Handle any remaining NaNs or Infs - set to zero
        final = torch.where(torch.isnan(final) | torch.isinf(final), torch.zeros_like(final), final)

        # Mean reduction
        if torch.any(masking):
            final = torch.sum(final) / (nelem + eps)
        else:
            final = torch.mean(final)

        return final

    def zinb_loss(
        self,
        x_genes_true: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-inflated negative binomial negative log-likelihood.

        For ZINB, the probability mass function is:
        P(Y = 0) = π + (1-π) * NB(0; μ, θ)
        P(Y = k) = (1-π) * NB(k; μ, θ) for k > 0

        Args:
            x_genes_true: Observed counts
            mu: Expected means (μ)
            theta: Dispersion parameter
            pi: Zero-inflation parameter (probability of structural zero)

        Returns:
            nll: Negative log-likelihood across batch
        """
        eps = self.eps
        x_genes_true = x_genes_true.float()
        mu = mu.float()

        # Mask NaN values
        masking = torch.isnan(x_genes_true) | torch.isinf(x_genes_true)
        nelem = torch.sum(~masking).float()
        x_genes_true = self._nan2zero(x_genes_true)

        # Compute log-likelihood for all observations using shared function
        log_nb_all = _compute_nb_log_likelihood(x_genes_true=x_genes_true, mu=mu, theta=theta, eps=eps)

        # For zero observations: log[π + (1-π) * NB(0; μ, θ)]
        # We need NB(0; μ, θ), which we get by computing log_nb for x=0 and then exp
        zeros_like_true = torch.zeros_like(x_genes_true)
        log_nb_zero = _compute_nb_log_likelihood(x_genes_true=zeros_like_true, mu=mu, theta=theta, eps=eps)

        # Convert to probability: NB(0; μ, θ) = exp(log_nb_zero) with stability
        # Clamp log_nb_zero to prevent overflow in exp
        log_nb_zero_clamped = torch.clamp(log_nb_zero, min=-50, max=50)
        nb_zero = torch.exp(log_nb_zero_clamped)
        nb_zero = torch.clamp(nb_zero, min=eps, max=1.0)  # Ensure valid probability

        # Zero case: log[π + (1-π) * NB(0; μ, θ)] with stability
        pi_clamped = torch.clamp(pi, min=eps, max=1.0 - eps)  # Ensure valid probability
        zero_prob = pi_clamped + (1.0 - pi_clamped) * nb_zero
        zero_case = torch.log(zero_prob + eps)

        # Non-zero case: log[(1-π) * NB(k; μ, θ)] = log(1-π) + log_nb with stability
        nonzero_case = torch.log(1.0 - pi_clamped + eps) + log_nb_all

        # Combine zero and non-zero cases
        log_likelihood = torch.where(
            x_genes_true <= eps,  # Use eps instead of exact zero comparison
            zero_case,
            nonzero_case,
        )

        # Negative log-likelihood
        nll = -log_likelihood

        # Handle any remaining NaNs or Infs - set to zero
        nll = torch.where(torch.isnan(nll) | torch.isinf(nll), torch.zeros_like(nll), nll)

        # Mean reduction
        if torch.any(masking):
            nll = torch.sum(nll) / (nelem + eps)
        else:
            nll = torch.mean(nll)

        return nll

    def __call__(
        self,
        x_genes_true: torch.Tensor,
        mu: torch.Tensor,
        pi: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        zinb_loss = self.zinb_loss(
            x_genes_true=x_genes_true,
            mu=mu,
            theta=theta,
            pi=pi,
        )

        return zinb_loss


class NB:
    def __init__(
        self,
        use_masking: bool = False,
        eps: float = 1e-8,
    ):
        """Negative binomial loss.

        Args:
            use_masking: Whether to use masking for NaN values
            eps: Small constant for numerical stability
        """
        self.use_masking = use_masking
        self.eps = eps

    def _nan2zero(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN values with zeros."""
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)

    def loss(self, x_genes_true: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Negative binomial negative log-likelihood loss.

        Args:
            x_genes_true: Observed counts
            mu: Expected means (μ)
            theta: Dispersion parameter

        Returns:
            Mean negative log-likelihood across batch
        """
        eps = self.eps
        x_genes_true = x_genes_true.float()

        # Mask NaN values if requested
        if self.use_masking:
            masking = torch.isnan(x_genes_true) | torch.isinf(x_genes_true)
            nelem = torch.sum(~masking).float()
            x_genes_true = self._nan2zero(x_genes_true)
        else:
            masking = None
            nelem = torch.numel(x_genes_true)

        # Use shared computation for consistency with ZINB
        log_likelihood = _compute_nb_log_likelihood(x_genes_true=x_genes_true, mu=mu, theta=theta, eps=eps)

        # Negative log-likelihood
        nll = -log_likelihood

        # Handle any remaining NaNs or Infs
        nll = torch.where(torch.isnan(nll) | torch.isinf(nll), torch.zeros_like(nll), nll)

        # Mean reduction
        if self.use_masking and masking is not None and torch.any(masking):
            nll = torch.sum(nll) / (nelem + eps)
        else:
            nll = torch.mean(nll)

        return nll

    def __call__(
        self,
        x_genes_true: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        nb_loss = self.loss(
            x_genes_true=x_genes_true,
            mu=mu,
            theta=theta,
        )

        return nb_loss


def create_autoencoder_recon_loss_function(prm: AutoEncoderParams) -> torch.nn.Module:
    """Create reconstruction loss function based on parameters."""
    loss_type = prm.loss_type.lower()

    if loss_type == "zinb":
        return ZINBLoss()
    elif loss_type == "nb":
        return NB()
    elif loss_type == "mse":
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_total_autoencoder_loss(
    *,
    x_genes_true: torch.Tensor,
    mu: torch.Tensor,
    pi: torch.Tensor | None,
    theta: torch.Tensor | None,
    latent_vec: torch.Tensor,
    class_logit: torch.Tensor,
    y_true: torch.Tensor,
    y_soft_gt: torch.Tensor | None,
    recon_loss_fn: ZINBLoss | NB | MSELoss,
    bce_loss: torch.nn.Module,
    prm: AutoEncoderParams,
    epoch: int,
    categorical_logits: dict[str, torch.Tensor] | None = None,
    categorical_targets: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the total loss as the sum of reconstruction loss, classification loss, and categorical losses.

    Args:
        x_genes_true: Observed counts, shape [batch_size, n_genes]
        mu: Predicted means, shape [batch_size, n_genes]
        pi: Zero-inflation probabilities, shape [batch_size, n_genes] (None for MSE/NB)
        theta: Dispersion parameters, shape [batch_size, n_genes] (None for MSE)
        latent_vec: Latent vector from encoder, shape [batch_size, latent_dim]
        class_logit: Classification logits, shape [batch_size]
        y_true: True binary labels, shape [batch_size]
        y_soft_gt: Soft binary labels, shape [batch_size] in range [0, 1] (None if not using soft loss)
        recon_loss_fn: Loss function instance (ZINBLoss, NB, or MSELoss)
        bce_loss: Binary cross-entropy loss module
        prm: AutoEncoderParams containing all loss weights and regularization parameters
        epoch: Current training epoch (used for soft loss scheduling)
        categorical_logits: Dict of logits for categorical features, each shape [batch_size, n_classes]
        categorical_targets: Dict of target indices for categorical features, each shape [batch_size]

    Returns:
        total_loss: torch.Tensor, sum of reconstruction, classification, and categorical losses
        recon_loss: torch.Tensor, reconstruction loss
        cls_loss: torch.Tensor, classification loss
        l1_loss: torch.Tensor, L1 regularization loss
        cat_loss: torch.Tensor, sum of categorical classification losses
    """
    # Validate inputs
    if x_genes_true.device != mu.device:
        raise ValueError(f"Device mismatch: x_genes_true on {x_genes_true.device}, m on {mu.device}")

    device = x_genes_true.device

    # --- Reconstruction loss ---
    if isinstance(recon_loss_fn, MSELoss):
        recon_loss = recon_loss_fn(
            x_genes_true=x_genes_true,
            x_pred=mu,
        )
    elif isinstance(recon_loss_fn, NB):
        if theta is None:
            raise ValueError("NB loss requires theta parameter, but theta is None")
        recon_loss = recon_loss_fn(
            x_genes_true=x_genes_true,
            mu=mu,
            theta=theta,
        )
    elif isinstance(recon_loss_fn, ZINBLoss):
        if theta is None:
            raise ValueError("ZINB loss requires theta parameter, but theta is None")
        if pi is None:
            raise ValueError("ZINB loss requires pi parameter, but pi is None")
        recon_loss = recon_loss_fn.zinb_loss(
            x_genes_true=x_genes_true,
            mu=mu,
            theta=theta,
            pi=pi,
        )
    else:
        raise ValueError("Unknown reconstruction loss function type.")

    # --- Regularization losses ---
    # L2 regularization on pi (ridge regularization)
    l2_loss = torch.tensor(0.0, device=device)
    if pi is not None and prm.ridge_lambda > 0:
        l2_loss = prm.ridge_lambda * torch.mean(pi**2)

    # L1 regularization on latent vector
    l1_loss = torch.tensor(0.0, device=device)
    if latent_vec is not None and prm.l1_lambda > 0:
        l1_loss = prm.l1_lambda * torch.mean(torch.abs(latent_vec))

    # --- Binary classification loss ---
    # Determine target based on soft loss schedule
    if y_soft_gt is not None and should_use_soft_loss(epoch=epoch, prm=prm):
        target = y_soft_gt
    else:
        target = y_true

    cls_loss = bce_loss(class_logit, target)

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
                if torch.any(targets >= n_classes) or torch.any(targets < 0):
                    invalid_indices = torch.logical_or(targets >= n_classes, targets < 0)
                    targets = torch.where(invalid_indices, torch.zeros_like(targets), targets)
                feature_loss = cross_entropy(logits, targets)
                cat_loss = cat_loss + feature_loss

    # --- Total loss ---
    total_loss = prm.recon_loss_weight * (recon_loss + l1_loss + l2_loss) + prm.cls_loss_weight * cls_loss
    if has_categorical:
        total_loss = total_loss + prm.cat_loss_weight * cat_loss

    return total_loss, recon_loss, cls_loss, l1_loss, cat_loss
