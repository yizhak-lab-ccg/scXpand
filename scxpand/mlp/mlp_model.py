import json

from dataclasses import dataclass
from pathlib import Path

import torch

from torch import nn
from torch.nn import GELU, Dropout, LayerNorm, Linear

from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.mlp.mlp_params import MLPParam
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import DATA_FORMAT_FILE, PARAMETERS_FILE
from scxpand.util.model_loading import (
    ModelLoadingError,
    load_and_validate_model,
    load_model_parameters,
    validate_model_directory,
)


logger = get_logger()


@dataclass
class ModelOutput:
    main_logit: torch.Tensor  # Main prediction logits with shape [batch_size]
    categorical_logits: dict[str, torch.Tensor] | None = (
        None  # Auxiliary categorical predictions, dict[feature_name, logits]
    )


class FC_Net(torch.nn.Module):
    """Fully connected neural network with layer normalization and dropout.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hid_layers: Tuple of hidden layer dimensions
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_layers: tuple,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hid_layers
        self.dims = [self.in_dim, *self.hidden_dims]
        self.input_ln = LayerNorm(self.in_dim)

        self.layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            block = nn.Sequential(
                Linear(self.dims[i - 1], self.dims[i]),
                LayerNorm(self.dims[i]),
                GELU(),
                Dropout(dropout_rate),
            )
            self.layers.append(block)

        self.output = nn.Sequential(
            Linear(self.dims[-1], self.dims[-1] // 2),
            LayerNorm(self.dims[-1] // 2),
            GELU(),
            Dropout(dropout_rate),
            Linear(self.dims[-1] // 2, out_dim),
        )

    def forward(self, x):
        x = self.input_ln(x)
        for layer in self.layers:
            x = layer(x)

        x = self.output(x)
        return x  # [batch_size, out_dim]


class CategoricalHead(nn.Module):
    """Neural network head for categorical feature prediction.

    Args:
        in_dim: Input dimension
        n_classes: Number of output classes
        hidden_dims: Tuple of hidden layer dimensions
        dropout_rate: Dropout probability
        feature_name: Name of the feature being predicted
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dims: tuple = (16,),
        dropout_rate: float = 0.3,
        feature_name: str | None = None,
    ):
        super().__init__()
        self.feature_name = feature_name
        self.n_classes = n_classes

        layers = []
        prev_dim = in_dim
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)  #  [batch_size, n_classes]
        return logits


class MLPModel(nn.Module):
    """Multi-layer perceptron model for single-cell expansion prediction.

    Args:
        prm: MLP parameters including layer dimensions and training config
        data_format: Data format specification with gene information
        device: Device to place the model on (cuda/cpu)
    """

    def __init__(
        self,
        prm: MLPParam,
        data_format: DataFormat,
        device: str | None = None,
    ):
        super().__init__()
        self.data_format = data_format
        self.n_input_features = data_format.n_genes
        self.fc_net = FC_Net(in_dim=self.n_input_features, out_dim=1, hid_layers=prm.layer_units)
        self.device = device

        # Initialize categorical feature prediction heads for auxiliary categorical targets
        self.categorical_heads = nn.ModuleDict()
        if data_format.aux_categorical_mappings:
            # Use the last hidden layer size as input dimension for categorical heads
            hidden_dim = prm.layer_units[-1] if prm.layer_units else 64
            for feature_name, mapping in data_format.aux_categorical_mappings.items():
                n_classes = len(mapping)
                self.categorical_heads[feature_name] = CategoricalHead(
                    in_dim=hidden_dim,
                    n_classes=n_classes,
                    hidden_dims=(16,),
                    dropout_rate=prm.dropout_rate,
                    feature_name=feature_name,
                )

        self.float()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass of the model.

        Args:
            x: torch.Tensor with shape [batch_size, n_features], input data
        Returns:
            ModelOutput containing logits and optional categorical_logits
        """
        # Get features from the main network (before final output layer)
        z = self.fc_net.input_ln(x)
        for layer in self.fc_net.layers:
            z = layer(z)

        # Get main task prediction logit
        pred_logit = self.fc_net.output(z).squeeze()

        # Get auxiliary categorical predictions if we have categorical heads
        if self.categorical_heads:
            categorical_logits = {}
            for feature_name, head in self.categorical_heads.items():
                categorical_logits[feature_name] = head(z)
            return ModelOutput(main_logit=pred_logit, categorical_logits=categorical_logits)
        else:
            return ModelOutput(main_logit=pred_logit)


def load_nn_model(results_path: Path, device: torch.device | str) -> MLPModel:
    """Load a trained MLP model using unified loading utilities.

    Args:
        results_path: Path to directory containing model files
        device: Device to load model on

    Returns:
        Loaded MLP model ready for inference

    Raises:
        ModelLoadingError: If model loading fails
        FileNotFoundError: If required files are missing
    """
    results_path = Path(results_path)

    try:
        # Validate model directory structure
        required_files = [PARAMETERS_FILE, DATA_FORMAT_FILE]
        validate_model_directory(results_path, required_files)

        # Load parameters using unified utility
        prm = load_model_parameters(results_path, MLPParam)

        # Load data format
        data_format = load_data_format(results_path / "data_format.json")

        # Create model architecture
        model = MLPModel(prm=prm, device=device, data_format=data_format)

        # Load and validate model using unified utility
        model = load_and_validate_model(model, results_path, device, "MLP")

        return model

    except (ModelLoadingError, FileNotFoundError, json.JSONDecodeError, TypeError):
        # Re-raise expected exceptions as-is
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        raise ModelLoadingError(f"Unexpected error loading MLP model from {results_path}: {e}") from e
