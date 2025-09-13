"""scXpand: Pan-cancer detection of T-cell clonal expansion from single-cell RNA sequencing.

A framework for predicting T-cell clonal expansion from single-cell RNA sequencing data
using multiple machine learning approaches including autoencoders, MLPs, LightGBM, and linear models.
"""

from importlib.metadata import version

from scxpand.core.inference import run_inference
from scxpand.core.prediction import run_prediction_pipeline
from scxpand.pretrained import (
    PRETRAINED_MODELS,
    download_pretrained_model,
    get_pretrained_model_info,
)
from scxpand.util.classes import ModelType
from scxpand.util.model_registry import list_pretrained_models


def _get_version():
    """Get the version of the installed scxpand package."""
    # Try different package names in order of preference
    package_names = ["scxpand-cuda", "scxpand"]

    for package_name in package_names:
        try:
            return version(package_name)
        except Exception:
            continue

    # If no package is found, return fallback version
    return "0.0.0"


__version__ = _get_version()
_version_ = __version__  # Alias for compatibility

__all__ = [
    "PRETRAINED_MODELS",
    "ModelType",
    "_version_",
    "download_pretrained_model",
    "get_pretrained_model_info",
    "list_pretrained_models",
    "run_inference",
    "run_prediction_pipeline",
]
