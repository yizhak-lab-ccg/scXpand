"""Registry for pre-trained models and their metadata.

This module defines the available pre-trained models, their Google Drive links,
and associated metadata for automatic download and inference.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PretrainedModelInfo:
    """Information about a pre-trained model."""

    name: str
    url: str  # Direct download URL for the model ZIP file
    version: str


# Registry of available pre-trained models
# Key: model_name (registry identifier - can be any descriptive string)
# Note: model_type is auto-detected from model_type.txt file in each model archive
PRETRAINED_MODELS: Dict[str, PretrainedModelInfo] = {
    "pan_cancer_autoencoder": PretrainedModelInfo(
        name="pan_cancer_autoencoder",
        url="https://figshare.com/ndownloader/articles/30067666/versions/1?folder_path=autoencoder",
        version="1.0.0",
    ),
    "pan_cancer_mlp": PretrainedModelInfo(
        name="pan_cancer_mlp",
        url="https://figshare.com/ndownloader/articles/30067666/versions/1?folder_path=mlp",
        version="1.0.0",
    ),
    "pan_cancer_lightgbm": PretrainedModelInfo(
        name="pan_cancer_lightgbm",
        url="https://figshare.com/ndownloader/articles/30067666/versions/1?folder_path=lightgbm",
        version="1.0.0",
    ),
    "pan_cancer_logistic": PretrainedModelInfo(
        name="pan_cancer_logistic",
        url="https://figshare.com/ndownloader/articles/30067666/versions/1?folder_path=logistic",
        version="1.0.0",
    ),
    "pan_cancer_svm": PretrainedModelInfo(
        name="pan_cancer_svm",
        url="https://figshare.com/ndownloader/articles/30067666/versions/1?folder_path=svm",
        version="1.0.0",
    ),
}


def get_pretrained_model_info(model_name: str) -> PretrainedModelInfo:
    """Get information about a pre-trained model.

    Args:
        model_name: Name of the pre-trained model

    Returns:
        PretrainedModelInfo object containing model metadata

    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in PRETRAINED_MODELS:
        available_models = ", ".join(PRETRAINED_MODELS.keys())
        raise ValueError(f"Pre-trained model '{model_name}' not found. Available models: {available_models}")

    return PRETRAINED_MODELS[model_name]
