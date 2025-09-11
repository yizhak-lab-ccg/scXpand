import torch

from torch import nn
from torch.nn import GELU, Dropout, LayerNorm, Linear
from torch.nn.functional import sigmoid, softplus

from scxpand.autoencoders.ae_model_output import DecoderOutput, ModelOutput
from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.transforms import (
    apply_inverse_log_transform,
    apply_inverse_zscore_normalization,
    apply_row_normalization,
)
from scxpand.util.logger import get_logger


logger = get_logger()


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64,),
        dropout_rate: float = 0.3,
    ):
        """Initialize encoder network.

        Args:
            in_dim: Input dimension
            latent_dim: Dimension of the latent space
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(Linear(prev_dim, h_dim))
            layers.append(LayerNorm(h_dim))
            layers.append(GELU())
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            prev_dim = h_dim
        self.shared_net = nn.Sequential(*layers)
        self.latent_vec = nn.Linear(prev_dim, latent_dim)

        # Initialize weights properly
        self._initialize_encoder_weights()

    def _initialize_encoder_weights(self):
        """Initialize encoder weights with PyTorch default Xavier initialization."""
        for module in self.shared_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)  # PyTorch default gain=1.0
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize latent layer with default Xavier initialization
        nn.init.xavier_normal_(self.latent_vec.weight)
        nn.init.zeros_(self.latent_vec.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared_net(x)
        latent_vec = self.latent_vec(h)
        return latent_vec


def apply_inverse_transforms_to_mean(raw_mu: torch.Tensor, data_format: DataFormat) -> torch.Tensor:
    """Apply inverse transforms to decoder mean output to match row-normalized target scale.

    Applies inverse transforms in reverse order:
    1. Inverse z-score normalization
    2. Inverse log transform (if used)

    Args:
        raw_mu: Decoder mean output (raw logits from the model)
        data_format: Data format containing normalization parameters

    Returns:
        Transformed mean that matches row-normalized target scale
    """
    # Apply inverse z-score normalization
    mu = apply_inverse_zscore_normalization(
        X=raw_mu,
        genes_mu=data_format.genes_mu,
        genes_sigma=data_format.genes_sigma,
    )

    # Apply inverse log transform if it was used during preprocessing.
    # The `apply_inverse_log_transform` function has its own internal clamping for stability.
    if data_format.use_log_transform:
        mu = apply_inverse_log_transform(mu)

    # Ensure non-negative values (counts should be non-negative).
    mu = softplus(mu)

    # Apply row normalization - to match the target scale
    mu = apply_row_normalization(
        X=mu,
        target_sum=data_format.target_sum,
    )

    return mu


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64,),
        dropout_rate: float = 0.3,
        n_genes: int | None = None,
        needs_pi: bool = True,
        needs_theta: bool = True,
        data_format: DataFormat | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_genes = n_genes
        self.needs_pi = needs_pi
        self.needs_theta = needs_theta
        self.data_format = data_format
        input_dim = latent_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(Linear(prev_dim, h_dim))
            layers.append(LayerNorm(h_dim))
            layers.append(GELU())
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            prev_dim = h_dim
        self.shared_net = nn.Sequential(*layers)

        # Output heads - only create what's needed
        self.m_head = nn.Linear(in_features=prev_dim, out_features=self.n_genes)

        if self.needs_pi:
            self.pi_head = nn.Linear(in_features=prev_dim, out_features=self.n_genes)
        else:
            self.pi_head = None

        if self.needs_theta:
            self.theta_head = nn.Linear(in_features=prev_dim, out_features=self.n_genes)
        else:
            self.theta_head = None

    def forward(
        self,
        latent_vec: torch.Tensor,
    ) -> DecoderOutput:
        """Forward pass through the decoder.

        Args:
            latent_vec: Latent vector from encoder

        Returns:
            DecoderOutput containing:
                - mu: Mean parameter after inverse transforms to match row-normalized target scale
                - pi: Dropout probability (None if not needed), in [0, 1]
                - theta: Dispersion parameter (None if not needed)
        """
        h = self.shared_net(latent_vec)
        raw_mu = self.m_head(h)  # raw mean logits (no activation)

        # Apply inverse transforms to get mean in row-normalized scale for reconstruction loss
        mu = apply_inverse_transforms_to_mean(raw_mu=raw_mu, data_format=self.data_format)

        pi = None
        if self.needs_pi and self.pi_head is not None:
            pi = dropout_activation(self.pi_head(h))  # dropout probability

        theta = None
        if self.needs_theta and self.theta_head is not None:
            theta = theta_activation(self.theta_head(h))  # dispersion parameter (positive values)

        return DecoderOutput(mu=mu, pi=pi, theta=theta)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (16,),
        dropout_rate: float = 0.3,
    ):
        """Initialize classification head network.

        Args:
            latent_dim: Input dimension (latent vector size)
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout probability
        """
        super().__init__()
        # Use hidden layers instead of a single linear layer
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(Linear(prev_dim, h_dim))
            layers.append(LayerNorm(h_dim))
            layers.append(GELU())
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            prev_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latent_vec: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classification head.

        Args:
            latent_vec: Latent vector from encoder

        Returns:
            Classification logit
        """
        result = self.net(latent_vec).squeeze(-1)
        return result


class CategoricalHead(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (16,),
        dropout_rate: float = 0.3,
        n_classes: int = 2,
        feature_name: str | None = None,
    ):
        """Initialize a classification head for a categorical feature.

        Args:
            latent_dim: Input dimension (latent vector size)
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout probability
            n_classes: Number of classes for this categorical feature
            feature_name: Name of the feature this head predicts (for debugging)
        """
        super().__init__()
        self.feature_name = feature_name
        self.n_classes = n_classes

        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(Linear(prev_dim, h_dim))
            layers.append(LayerNorm(h_dim))
            layers.append(GELU())
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            prev_dim = h_dim

        # Final output layer
        layers.append(Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, latent_vec: torch.Tensor) -> torch.Tensor:
        """Forward pass through the categorical head.

        Args:
            latent_vec: Latent vector from encoder

        Returns:
            Categorical logits with shape [batch_size, n_classes]
        """
        return self.net(latent_vec)


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(
        self,
        data_format: DataFormat,
        latent_dim: int = 32,
        encoder_hidden_dims: tuple[int, ...] = (64,),
        decoder_hidden_dims: tuple[int, ...] = (64,),
        classifier_hidden_dims: tuple[int, ...] = (16,),
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.data_format = data_format
        self.n_genes = data_format.n_genes
        self.input_dim = self.n_genes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.classifier_hidden_dims = classifier_hidden_dims

        # Classifier for binary prediction
        self.classifier = ClassificationHead(
            latent_dim=latent_dim,
            hidden_dims=classifier_hidden_dims,
            dropout_rate=dropout_rate,
        )

        # Initialize categorical feature prediction heads for auxiliary categorical targets
        self.categorical_heads = nn.ModuleDict()
        if data_format.aux_categorical_mappings:
            for feature_name, mapping in data_format.aux_categorical_mappings.items():
                n_classes = len(mapping)
                self.categorical_heads[feature_name] = CategoricalHead(
                    latent_dim=latent_dim,
                    hidden_dims=classifier_hidden_dims,
                    dropout_rate=dropout_rate,
                    n_classes=n_classes,
                    feature_name=feature_name,
                )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input data to latent representation.

        Args:
            x: Input data tensor, shape [batch_size, n_genes]

        Returns:
            latent_vec: Latent representation, shape [batch_size, latent_dim]
        """
        raise NotImplementedError("Subclasses must implement the encode method")

    def decode(self, latent_vec: torch.Tensor) -> DecoderOutput:
        """Decode latent representation to output parameters.

        Args:
            latent_vec: Latent vector, shape [batch_size, latent_dim]

        Returns:
            DecoderOutput containing:
                - mu: Mean parameter after inverse transforms to match row-normalized target scale, shape [batch_size, n_genes], non-negative
                - pi: Zero-inflation parameter, shape [batch_size, n_genes] (None if not needed, in [0, 1])
                - theta: Dispersion parameter, shape [batch_size, n_genes] (None if not needed, positive values)

        Note:
            The decoder handles activation and inverse transforms internally.
        """
        raise NotImplementedError("Subclasses must implement the decode method")

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through the autoencoder model."""
        latent_vec = self.encode(x)
        decoding_results = self.decode(latent_vec)

        # Predict binary target
        class_logit = self.classifier(latent_vec)

        # Predict auxiliary categorical targets
        categorical_logits = {}
        for feature_name, head in self.categorical_heads.items():
            categorical_logits[feature_name] = head(latent_vec)

        return ModelOutput(
            latent_vec=latent_vec,
            mu=decoding_results.mu,
            class_logit=class_logit,
            pi=decoding_results.pi,
            theta=decoding_results.theta,
            categorical_logits=categorical_logits or {},
        )


def theta_activation(x: torch.Tensor) -> torch.Tensor:
    """Activation function for theta dispersion parameter in negative binomial regression.

    In negative binomial regression, theta (r) is the dispersion parameter that must be positive.
    """
    return torch.clamp(softplus(x), min=1e-4, max=1e3)


def dropout_activation(x: torch.Tensor) -> torch.Tensor:
    """Activation function for dropout probability parameter.

    Range: (0, 1) as appropriate for probabilities.
    """
    return sigmoid(x)
