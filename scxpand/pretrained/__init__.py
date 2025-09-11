"""Pre-trained model management for scXpand.

This module provides functionality to download and manage pre-trained models
from Google Drive, including automatic model loading for inference.
"""

from .download_manager import download_pretrained_model
from .inference_api import fetch_model_and_run_inference
from .model_registry import PRETRAINED_MODELS, get_pretrained_model_info


__all__ = [
    "PRETRAINED_MODELS",
    "download_pretrained_model",
    "fetch_model_and_run_inference",
    "get_pretrained_model_info",
]
