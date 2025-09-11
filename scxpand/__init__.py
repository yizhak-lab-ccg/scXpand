"""scXpand: Pan-cancer detection of T-cell clonal expansion from single-cell RNA sequencing.

A framework for predicting T-cell clonal expansion from single-cell RNA sequencing data
using multiple machine learning approaches including autoencoders, MLPs, LightGBM, and linear models.
"""

__version__ = "0.1.1"
__author__ = "Ron Amit, Ofir Shorer"
__email__ = "ron2amit@gmail.com, ofirshorer@campus.technion.ac.il"

# Import core functionality
from scxpand.core.inference import run_inference
from scxpand.core.prediction import run_prediction_pipeline

# Import CLI functions (lazy import to avoid module loading conflicts)
# from scxpand.main import list_pretrained_models
# Import pre-trained model functionality
from scxpand.pretrained import (
    PRETRAINED_MODELS,
    download_pretrained_model,
    get_pretrained_model_info,
)
from scxpand.util.classes import ModelType


__all__ = [
    "PRETRAINED_MODELS",
    "ModelType",
    "download_pretrained_model",
    "get_pretrained_model_info",
    "run_inference",
    "run_prediction_pipeline",
]
