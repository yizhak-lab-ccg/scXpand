"""Utility functions for managing the model registry."""

from scxpand.pretrained import PRETRAINED_MODELS
from scxpand.util.logger import get_logger


logger = get_logger()


def list_pretrained_models() -> None:
    """List all available pre-trained models with their information."""
    logger.info("Available pre-trained models:")
    logger.info("=" * 50)

    for name, info in PRETRAINED_MODELS.items():
        logger.info(f"Name: {name}")
        logger.info(f"  Version: {info.version}")
        logger.info(f"  URL configured: {'Yes' if info.url else 'No'}")
        logger.info("  Model type: Auto-detected from model_type.txt")
        logger.info("")
