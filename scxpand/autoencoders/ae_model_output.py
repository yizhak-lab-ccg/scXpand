from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class DecoderOutput:
    """Container for autoencoder decoder outputs.

    This class standardizes the decoder outputs, making the interface cleaner
    and more explicit about which outputs are available.

    The decoder handles all transformations internally and returns processed values.

    Attributes:
        mu: Mean parameter after inverse transforms to match row-normalized target scale [batch_size, n_genes]
        pi: Dropout probability [batch_size, n_genes] (optional, only for ZINB)
        theta: Dispersion parameter [batch_size, n_genes] (optional, only for NB/ZINB, positive values)
    """

    mu: torch.Tensor
    pi: torch.Tensor | None = None
    theta: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        """Get the device of the tensors."""
        return self.mu.device

    def to(self, device: torch.device | str) -> "DecoderOutput":
        """Move all tensors to the specified device."""
        mu = self.mu.to(device)
        pi = self.pi.to(device) if self.pi is not None else None
        theta = self.theta.to(device) if self.theta is not None else None
        return DecoderOutput(mu=mu, pi=pi, theta=theta)

    def detach(self) -> "DecoderOutput":
        """Detach all tensors from the computation graph."""
        return DecoderOutput(
            mu=self.mu.detach(),
            pi=self.pi.detach() if self.pi is not None else None,
            theta=self.theta.detach() if self.theta is not None else None,
        )


@dataclass
class ModelOutput:
    """Container for autoencoder model outputs.

    This class standardizes outputs from all autoencoder model variants, making code more maintainable
    and error handling more robust. It replaces the tuple return type with a proper object
    that has named attributes.

    Attributes:
        latent_vec: Encoded latent vector from encoder [batch_size, latent_dim]
        mu: Mean parameter after inverse transforms to match row-normalized target scale [batch_size, n_genes]
        class_logit: Classification logit [batch_size]
        pi: Dropout probability [batch_size, n_genes] (optional, only for ZINB)
        theta: Dispersion parameter [batch_size, n_genes] (optional, only for NB/ZINB, positive values)
        categorical_logits: Dict of categorical feature logits (optional) {feature_name: [batch_size, n_classes]}
    """

    latent_vec: torch.Tensor
    mu: torch.Tensor
    class_logit: torch.Tensor
    pi: torch.Tensor | None = None
    theta: torch.Tensor | None = None
    categorical_logits: Dict[str, torch.Tensor] | None = None

    def __post_init__(self):
        """Ensure categorical_logits is initialized as a dict."""
        # Ensure categorical_logits is a dictionary
        if self.categorical_logits is None:
            self.categorical_logits = {}

    @property
    def device(self) -> torch.device:
        """Get the device of the tensors."""
        return self.latent_vec.device

    def to(self, device: torch.device | str) -> "ModelOutput":
        """Move all tensors to the specified device."""
        return ModelOutput(
            latent_vec=self.latent_vec.to(device),
            mu=self.mu.to(device),
            class_logit=self.class_logit.to(device),
            pi=self.pi.to(device) if self.pi is not None else None,
            theta=self.theta.to(device) if self.theta is not None else None,
            categorical_logits={k: v.to(device) for k, v in self.categorical_logits.items()}
            if self.categorical_logits
            else {},
        )

    def detach(self) -> "ModelOutput":
        """Detach all tensors from the computation graph."""
        return ModelOutput(
            latent_vec=self.latent_vec.detach(),
            mu=self.mu.detach(),
            class_logit=self.class_logit.detach(),
            pi=(self.pi.detach() if self.pi is not None else None),
            theta=(self.theta.detach() if self.theta is not None else None),
            categorical_logits=(
                {k: v.detach() for k, v in self.categorical_logits.items()} if self.categorical_logits else {}
            ),
        )

    def cpu(self) -> "ModelOutput":
        return self.to("cpu")

    def get_binary_predictions(self) -> torch.Tensor:
        return torch.sigmoid(self.class_logit)
