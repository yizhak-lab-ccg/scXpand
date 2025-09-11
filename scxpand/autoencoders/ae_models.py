import json

from pathlib import Path
from typing import Tuple

import torch

from torch import nn

from scxpand.autoencoders.ae_model_output import DecoderOutput
from scxpand.autoencoders.ae_modules import (
    BaseAutoencoder,
    Decoder,
    Encoder,
    apply_inverse_transforms_to_mean,
    dropout_activation,
    theta_activation,
)
from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import DATA_FORMAT_FILE, PARAMETERS_FILE
from scxpand.util.model_loading import (
    ModelLoadingError,
    load_and_validate_model,
    load_model_parameters,
    validate_model_directory,
)


logger = get_logger()


class AutoencoderModel(BaseAutoencoder):
    def __init__(
        self,
        data_format: DataFormat,
        latent_dim: int,
        encoder_hidden_dims: Tuple[int, ...],
        decoder_hidden_dims: Tuple[int, ...],
        classifier_hidden_dims: Tuple[int, ...],
        dropout_rate: float,
        needs_pi: bool = True,
        needs_theta: bool = True,
    ):
        super().__init__(
            data_format=data_format,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            classifier_hidden_dims=classifier_hidden_dims,
            dropout_rate=dropout_rate,
        )

        self.encoder = Encoder(
            in_dim=self.input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            dropout_rate=dropout_rate,
            n_genes=self.n_genes,
            needs_pi=needs_pi,
            needs_theta=needs_theta,
            data_format=data_format,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent_vec = self.encoder(x)
        return latent_vec

    def decode(self, latent_vec: torch.Tensor) -> DecoderOutput:
        """Decode latent representation.

        Args:
            latent_vec: Latent vector, shape [batch_size, latent_dim]

        Returns:
            DecoderOutput containing:
                - mu: Mean parameter after inverse transforms to match row-normalized target scale, shape [batch_size, n_genes], non-negative
                - pi: Zero-inflation parameter, shape [batch_size, n_genes] (None if not needed, in [0, 1])
                - theta: Dispersion parameter, shape [batch_size, n_genes] (None if not needed, positive values)
        """
        return self.decoder(latent_vec)


class ForkAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        data_format: DataFormat,
        latent_dim: int,
        encoder_hidden_dims: Tuple[int, ...],
        decoder_hidden_dims: Tuple[int, ...],
        classifier_hidden_dims: Tuple[int, ...],
        dropout_rate: float,
        needs_pi: bool = True,
        needs_theta: bool = True,
    ):
        """The fork autoencoder has separate decoder paths for mean, dispersion, and dropout probability."""
        super().__init__(
            data_format=data_format,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            classifier_hidden_dims=classifier_hidden_dims,
            dropout_rate=dropout_rate,
        )

        self.needs_pi = needs_pi
        self.needs_theta = needs_theta

        # Encoder (shared)
        layers = []
        prev_dim = self.input_dim
        for h_dim in encoder_hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.latent_layer = nn.Linear(prev_dim, latent_dim)

        # Always create mean decoder
        self.mean_decoder = self._build_decoder_path(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            dropout_rate=dropout_rate,
        )
        self.m_head = nn.Linear(decoder_hidden_dims[-1] if decoder_hidden_dims else latent_dim, self.n_genes)

        # Conditionally create dispersion decoder
        if self.needs_theta:
            self.disp_decoder = self._build_decoder_path(
                latent_dim=latent_dim,
                hidden_dims=decoder_hidden_dims,
                dropout_rate=dropout_rate,
            )
            self.theta_head = nn.Linear(decoder_hidden_dims[-1] if decoder_hidden_dims else latent_dim, self.n_genes)
        else:
            self.disp_decoder = None
            self.theta_head = None

        # Conditionally create pi decoder
        if self.needs_pi:
            self.pi_decoder = self._build_decoder_path(
                latent_dim=latent_dim,
                hidden_dims=decoder_hidden_dims,
                dropout_rate=dropout_rate,
            )
            self.pi_head = nn.Linear(decoder_hidden_dims[-1] if decoder_hidden_dims else latent_dim, self.n_genes)
        else:
            self.pi_decoder = None
            self.pi_head = None

    def _build_decoder_path(self, latent_dim: int, hidden_dims: Tuple[int, ...], dropout_rate: float) -> nn.Sequential:
        """Build a decoder path with proper handling of empty hidden_dims."""
        if not hidden_dims:
            # If no hidden layers, return identity
            return nn.Sequential(nn.Identity())

        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        latent_vec = self.latent_layer(h)
        return latent_vec

    def decode(
        self,
        latent_vec: torch.Tensor,
    ) -> DecoderOutput:
        """Decode latent representation.

        Returns:
        -------
        DecoderOutput
            Contains:
            - mu: Mean parameter after inverse normalization to match row-normalized target scale
            - pi: Dropout probability (None if not needed)
            - theta: Dispersion parameter (None if not needed, positive values)
        """
        # Always decode mean
        mean_h = self.mean_decoder(latent_vec)
        raw_mu = self.m_head(mean_h)  # raw mean logits (no activation)

        # Apply inverse normalization to get mean in row-normalized scale for reconstruction loss
        mu = apply_inverse_transforms_to_mean(raw_mu=raw_mu, data_format=self.data_format)

        # Conditionally decode dispersion
        theta = None
        if self.needs_theta and self.disp_decoder is not None and self.theta_head is not None:
            disp_h = self.disp_decoder(latent_vec)
            theta = theta_activation(self.theta_head(disp_h))  # dispersion parameter (positive)

        # Conditionally decode pi
        pi = None
        if self.needs_pi and self.pi_decoder is not None and self.pi_head is not None:
            pi_h = self.pi_decoder(latent_vec)
            pi = dropout_activation(self.pi_head(pi_h))  # dropout probability

        return DecoderOutput(mu=mu, pi=pi, theta=theta)


def create_ae_model(data_format: DataFormat, prm: AutoEncoderParams, device: str | torch.device) -> BaseAutoencoder:
    model_type = prm.model_type
    logger.info(f"Model input dimensions: genes={data_format.n_genes}")
    logger.info(f"Building {model_type} model with latent_dim={prm.latent_dim}")

    if model_type == "standard":
        model = AutoencoderModel(
            data_format=data_format,
            latent_dim=prm.latent_dim,
            encoder_hidden_dims=prm.encoder_hidden_dims,
            decoder_hidden_dims=prm.decoder_hidden_dims,
            classifier_hidden_dims=prm.classifier_hidden_dims,
            dropout_rate=prm.dropout_rate,
            needs_pi=prm.needs_pi_head(),
            needs_theta=prm.needs_theta_head(),
        ).to(device)
    elif model_type == "fork":
        model = ForkAutoencoder(
            data_format=data_format,
            latent_dim=prm.latent_dim,
            encoder_hidden_dims=prm.encoder_hidden_dims,
            decoder_hidden_dims=prm.decoder_hidden_dims,
            classifier_hidden_dims=prm.classifier_hidden_dims,
            dropout_rate=prm.dropout_rate,
            needs_pi=prm.needs_pi_head(),
            needs_theta=prm.needs_theta_head(),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def load_ae_model(
    model_path: Path,
    device: str | torch.device,
) -> BaseAutoencoder:
    """Load a trained autoencoder model using unified loading utilities.

    Args:
        model_path: Path to directory containing model files
        device: Device to load model on ('cuda', 'mps', or 'cpu')

    Returns:
        Loaded autoencoder model ready for inference or embedding generation

    Raises:
        ModelLoadingError: If model loading fails
        FileNotFoundError: If required files are missing
    """
    model_path = Path(model_path)
    assert model_path.exists(), f"Model path does not exist: {model_path}"

    try:
        # Validate model directory structure
        required_files = [PARAMETERS_FILE, DATA_FORMAT_FILE]
        validate_model_directory(model_path, required_files)

        # Load parameters using unified utility
        prm = load_model_parameters(model_path, AutoEncoderParams)

        # Load data format
        data_format = load_data_format(model_path / "data_format.json")

        # Create model architecture
        model = create_ae_model(data_format=data_format, prm=prm, device=device)

        # Load and validate model using unified utility
        model = load_and_validate_model(model, model_path, device, "Autoencoder")

        return model

    except (ModelLoadingError, FileNotFoundError, json.JSONDecodeError, TypeError):
        # Re-raise expected exceptions as-is
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        raise ModelLoadingError(f"Unexpected error loading Autoencoder model from {model_path}: {e}") from e
