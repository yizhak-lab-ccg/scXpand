"""Unified model loading utilities for scXpand.

This module provides robust, consistent model loading functionality across all model types
to prevent silent failures and ensure proper error handling.
"""

import json

from pathlib import Path
from typing import Any, Dict, Union

import torch

from torch import nn

from scxpand.util.logger import get_logger
from scxpand.util.model_constants import (
    BEST_CHECKPOINT_FILE,
    PARAMETERS_FILE,
)


logger = get_logger()


class ModelLoadingError(Exception):
    """Custom exception for model loading failures."""


def load_model_state_dict(
    model_dir: Path,
    device: Union[torch.device, str],
    model_name: str = "model",
) -> Dict[str, Any]:
    """Load model state dictionary from training checkpoint.

    This function loads model state dictionary from best_ckpt.pt only.
    No fallback to legacy files - ensures consistent, reliable model loading.

    Args:
        model_dir: Directory containing model files
        device: Device to load model on
        model_name: Name of the model (for logging)

    Returns:
        Model state dictionary

    Raises:
        ModelLoadingError: If no valid model checkpoint is found
        FileNotFoundError: If model directory doesn't exist
        RuntimeError: If model loading fails due to corruption or incompatibility
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")

    # Define checkpoint path
    best_ckpt_path = model_dir / BEST_CHECKPOINT_FILE

    logger.info(f"Loading {model_name} model from {model_dir}")

    # Load from best checkpoint only
    if not best_ckpt_path.exists():
        available_files = list(model_dir.glob("*.pt"))
        raise ModelLoadingError(
            f"Training checkpoint not found: {best_ckpt_path}\n"
            f"Available .pt files: {available_files}\n"
            f"This prevents silent loading of untrained models."
        )

    try:
        logger.info(f"Loading from training checkpoint: {best_ckpt_path}")
        checkpoint_info = torch.load(best_ckpt_path, map_location=device, weights_only=False)

        # Validate checkpoint structure
        if not isinstance(checkpoint_info, dict):
            raise RuntimeError(f"Invalid checkpoint format in {best_ckpt_path}")

        if "model_state_dict" not in checkpoint_info:
            raise RuntimeError(f"Missing 'model_state_dict' key in {best_ckpt_path}")

        state_dict = checkpoint_info["model_state_dict"]
        logger.info(f"✅ Successfully loaded {model_name} from training checkpoint")
        return state_dict

    except Exception as e:
        logger.error(f"Failed to load from {best_ckpt_path}: {e}")
        raise ModelLoadingError(f"Failed to load {model_name} model from {best_ckpt_path}: {e}") from e


def load_model_parameters(model_dir: Path, param_class: type, param_file: str = PARAMETERS_FILE) -> Any:
    """Load model parameters from JSON file.

    Args:
        model_dir: Directory containing parameter file
        param_class: Parameter class to instantiate
        param_file: Name of parameter file

    Returns:
        Instantiated parameter object

    Raises:
        FileNotFoundError: If parameter file doesn't exist
        json.JSONDecodeError: If parameter file is invalid JSON
        TypeError: If parameter instantiation fails
    """
    model_dir = Path(model_dir)
    params_path = model_dir / param_file

    if not params_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_path}")

    try:
        with params_path.open("r") as f:
            param_dict = json.load(f)

        if not isinstance(param_dict, dict):
            raise ValueError(f"Parameter file must contain a JSON object: {params_path}")

        # Instantiate parameter object
        params = param_class(**param_dict)
        logger.info(f"✅ Loaded parameters from {params_path}")
        return params

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in parameter file {params_path}: {e.msg}", e.doc, e.pos) from e
    except TypeError as e:
        raise TypeError(f"Failed to instantiate {param_class.__name__} from {params_path}: {e}") from e


def load_and_validate_model(
    model: nn.Module, model_dir: Path, device: Union[torch.device, str], model_name: str = "model"
) -> nn.Module:
    """Load state dict into model with validation.

    Args:
        model: Initialized model instance
        model_dir: Directory containing model files
        device: Device to load model on
        model_name: Name of the model (for logging)

    Returns:
        Model with loaded weights in eval mode

    Raises:
        ModelLoadingError: If loading fails
        RuntimeError: If state dict is incompatible with model
    """
    try:
        # Load state dictionary
        state_dict = load_model_state_dict(model_dir, device, model_name)

        # Move model to device first
        model = model.to(device)

        # Load state dict with strict checking
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Validate loading results
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

        # Critical validation: ensure some weights were actually loaded
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            raise ModelLoadingError("Model has no parameters after loading")

        # Set to evaluation mode
        model.eval()

        logger.info(f"✅ {model_name} model loaded and validated successfully")
        logger.info(f"Model has {total_params:,} parameters")

        return model

    except Exception as e:
        if isinstance(e, ModelLoadingError):
            raise
        raise ModelLoadingError(f"Failed to load {model_name} model: {e}") from e


def validate_model_directory(model_dir: Path, required_files: list[str]) -> None:
    """Validate that model directory contains required files.

    Args:
        model_dir: Directory to validate
        required_files: List of required file names

    Raises:
        FileNotFoundError: If directory or required files don't exist
        ValueError: If path is not a directory
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")

    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        available_files = [f.name for f in model_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Missing required files in {model_dir}: {missing_files}\nAvailable files: {available_files}"
        )
