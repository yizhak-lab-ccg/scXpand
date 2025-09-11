"""Inference results data structure for scXpand models.

This module defines the InferenceResults dataclass that provides a structured
way to return inference results instead of using dictionaries.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ModelInfo:
    """Model metadata information.

    Attributes:
        model_name: Name of the model (for registry models)
        model_type: Type of model (e.g., "mlp", "autoencoder")
        version: Model version (for registry models)
        source: Source of the model ("local", "registry", "external_url")
        url: URL for external models
    """

    model_name: str | None = None
    model_type: str | None = None
    version: str | None = None
    source: str | None = None
    url: str | None = None


@dataclass
class InferenceResults:
    """Structured results from scXpand model inference.

    This dataclass provides a clean, typed interface for accessing inference
    results instead of using dictionaries. It automatically handles cases where
    ground truth labels are not available (metrics will be None).

    Attributes:
        predictions: Array of prediction probabilities [0, 1] for each cell
        metrics: Dictionary containing evaluation metrics (None if ground truth unavailable)
        model_info: Model metadata information (None for local models)

    Examples:
        >>> results = scxpand.run_inference(
        ...     data_path="data.h5ad", model_path="model/"
        ... )
        >>> print(f"Generated {len(results.predictions)} predictions")
        >>> if results.metrics:
        ...     print(f"AUROC: {results.metrics['AUROC']:.3f}")
        >>> if results.model_info:
        ...     print(f"Model type: {results.model_info.model_type}")
    """

    predictions: np.ndarray
    metrics: Dict[str, Any] | None = None
    model_info: ModelInfo | None = None

    def __post_init__(self):
        """Validate the inference results after initialization."""
        if not isinstance(self.predictions, np.ndarray):
            raise TypeError("predictions must be a numpy array")

        if len(self.predictions) == 0:
            raise ValueError("predictions array cannot be empty")

        if not np.all((self.predictions >= 0) & (self.predictions <= 1)):
            raise ValueError("predictions must be probabilities in range [0, 1]")

    @property
    def n_predictions(self) -> int:
        """Number of predictions generated."""
        return len(self.predictions)

    @property
    def has_metrics(self) -> bool:
        """Whether evaluation metrics are available."""
        return self.metrics is not None

    @property
    def has_model_info(self) -> bool:
        """Whether model metadata is available."""
        return self.model_info is not None

    def get_auroc(self) -> float | None:
        """Get the overall AUROC score if available."""
        if not self.has_metrics:
            return None
        return self.metrics.get("AUROC")

    def get_harmonic_avg_auroc(self) -> float | None:
        """Get the harmonic average AUROC across strata if available."""
        if not self.has_metrics:
            return None
        harmonic_avg = self.metrics.get("harmonic_avg")
        if harmonic_avg and isinstance(harmonic_avg, dict):
            return harmonic_avg.get("AUROC")
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {"predictions": self.predictions}

        if self.metrics is not None:
            result["metrics"] = self.metrics

        if self.model_info is not None:
            result["model_info"] = {
                "model_name": self.model_info.model_name,
                "model_type": self.model_info.model_type,
                "version": self.model_info.version,
                "source": self.model_info.source,
                "url": self.model_info.url,
            }

        return result
