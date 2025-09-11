"""Simple model type detection.

This module provides utilities to:
1. Save model type to a separate file during training
2. Load model type from saved file
3. Provide helpful error messages with available options
"""

from pathlib import Path

from scxpand.util.classes import ModelType
from scxpand.util.logger import get_logger


logger = get_logger()

MODEL_TYPE_FILE = "model_type.txt"


def save_model_type(model_type: str, save_dir: Path | str) -> None:
    """Save model type to a text file.

    Args:
        model_type: The model type string (e.g., "autoencoder", "mlp", etc.)
        save_dir: Directory where to save the model type
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_type_file = save_dir / MODEL_TYPE_FILE
    with open(model_type_file, "w") as f:
        f.write(model_type)

    logger.info(f"Saved model type '{model_type}' to {model_type_file}")


def load_model_type(results_path: Path | str) -> str | None:
    """Load model type from saved file.

    Args:
        results_path: Path to the directory containing model type file

    Returns:
        Model type string if found, None if file doesn't exist
    """
    results_path = Path(results_path)
    model_type_file = results_path / MODEL_TYPE_FILE

    if not model_type_file.exists():
        return None

    try:
        with open(model_type_file) as f:
            content = f.read().strip()

        if not content:
            return None

        # Handle various user input formats
        model_type = _clean_model_type_input(content)
        return model_type if model_type else None
    except Exception as e:
        logger.warning(f"Could not read model type from {model_type_file}: {e}")
        return None


def _clean_model_type_input(content: str) -> str:
    """Clean and normalize model type input from various formats.

    Handles cases like:
    - Plain text: mlp
    - With quotes: "mlp" or 'mlp'
    - With braces: {mlp}
    - Extra whitespace, newlines, etc.
    """
    # Strip whitespace and newlines
    content = content.strip()

    # Remove quotes if present
    if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
        content = content[1:-1]

    # Final cleanup
    content = content.strip()

    return content


def infer_model_type_from_parameters(results_path: Path | str) -> str:
    """Infer the model type from saved information.

    Args:
        results_path: Path to the directory containing model files

    Returns:
        Model type string

    Raises:
        ValueError: If model type cannot be determined with helpful message
    """
    results_path = Path(results_path)

    # Try to load from saved model type file
    model_type = load_model_type(results_path)
    if model_type:
        logger.info(f"Loaded model type: {model_type}")
        return model_type

    # If not found, show helpful error message
    available_types = get_available_model_types()

    raise ValueError(
        f"Model type file 'model_type.txt' not found in {results_path}\n\n"
        f"To fix this:\n"
        f"1. Create a file named 'model_type.txt' in: {results_path}\n"
        f"2. Add just the model type name (e.g., 'mlp' or 'autoencoder')\n\n"
        f"Available model types: {', '.join(available_types)}\n\n"
        f"Example: echo 'mlp' > {results_path}/model_type.txt"
    )


def get_available_model_types() -> list[str]:
    """Get list of available model types from the ModelType enum."""
    return [model_type.value for model_type in ModelType]
