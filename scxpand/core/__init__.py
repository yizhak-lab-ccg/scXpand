"""Core domain logic for scXpand.

This module contains the core business logic that doesn't depend on
external frameworks or utilities. It follows clean architecture principles.
"""

from scxpand.core.inference import run_inference
from scxpand.core.prediction import run_prediction_pipeline


__all__ = [
    "run_inference",
    "run_prediction_pipeline",
]
