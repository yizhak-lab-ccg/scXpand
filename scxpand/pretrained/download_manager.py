"""Download manager for pre-trained models using Pooch.

This module handles downloading pre-trained models using the Pooch library,
which provides robust caching, integrity checking, and progress tracking.
"""

import re
import shutil

from pathlib import Path

import pooch

from scxpand.util.logger import get_logger
from scxpand.util.model_constants import (
    BEST_CHECKPOINT_FILE,
    DATA_FORMAT_FILE,
    DATA_FORMAT_NPZ_FILE,
    MODEL_TYPE_FILE,
    PARAMETERS_FILE,
    SKLEARN_MODEL_FILE,
)

from .model_registry import get_pretrained_model_info


logger = get_logger()


def _normalize_model_filenames(model_dir: Path) -> None:
    """Normalize model filenames by removing numeric prefixes.

    Some model archives contain files with numeric prefixes (e.g., '57702349_data_format.json').
    This function removes such prefixes to match the expected filenames.

    Pooch handles version updates automatically through hash checking of the ZIP file.
    When a new version is downloaded, it gets a new cache directory, so we can
    safely rename files without conflicts.

    Args:
        model_dir: Directory containing model files to normalize
    """
    # Expected filenames for model files
    expected_files = {
        DATA_FORMAT_FILE,
        DATA_FORMAT_NPZ_FILE,
        PARAMETERS_FILE,
        MODEL_TYPE_FILE,
        SKLEARN_MODEL_FILE,  # For sklearn models
        BEST_CHECKPOINT_FILE,  # For PyTorch models
    }

    # Pattern to match numeric prefixes (e.g., "57702349_")
    prefix_pattern = re.compile(r"^\d+_(.+)$")

    for file_path in model_dir.iterdir():
        if file_path.is_file():
            match = prefix_pattern.match(file_path.name)
            if match:
                original_filename = match.group(1)
                if original_filename in expected_files:
                    new_path = model_dir / original_filename
                    # Simply rename - Pooch ensures we're in a fresh directory for new versions
                    logger.info(f"Normalizing filename: {file_path.name} -> {original_filename}")
                    shutil.move(str(file_path), str(new_path))


def download_pretrained_model(
    model_name: str | None = None,
    model_url: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a pre-trained model and return the path to the extracted model.

    Uses Pooch for robust caching, automatic hash verification, and extraction.
    Pooch automatically computes SHA256 hashes on first download and verifies
    them on subsequent accesses for integrity checking. When a model is updated
    (different hash), Pooch automatically downloads the new version to a fresh
    cache directory, ensuring version updates work seamlessly.
    Supports both registry models and direct URLs, including DOI URLs.

    By default, downloads to a `.scxpand_cache` directory in the current working
    directory, making it easy for users to manage and clean up downloaded models.

    Args:
        model_name: Name of pre-trained model from registry (alternative to model_url)
        model_url: Direct URL to model file (alternative to model_name)
                  Supports HTTP/HTTPS URLs for direct downloads
        cache_dir: Custom cache directory (uses `.scxpand_cache` in current dir if None)

    Returns:
        Path to the extracted model directory or file

    Raises:
        ValueError: If neither model_name nor model_url is provided, or if both are provided

    Examples:
        >>> # Registry model (downloads to ./.scxpand_cache/)
        >>> model_path = download_pretrained_model(
        ...     model_name="pan_cancer_autoencoder"
        ... )
        >>>
        >>> # Direct URL (downloads to ./.scxpand_cache/)
        >>> model_path = download_pretrained_model(
        ...     model_url="https://your-platform.com/model.zip"
        ... )
        >>>
        >>> # Custom cache directory
        >>> model_path = download_pretrained_model(
        ...     model_url="https://figshare.com/ndownloader/files/model.zip",
        ...     cache_dir=Path("/my/custom/cache"),
        ... )
    """
    # Validate inputs
    if model_name is None and model_url is None:
        raise ValueError("Either model_name or model_url must be provided")

    if model_name is not None and model_url is not None:
        raise ValueError("Cannot specify both model_name and model_url. Use one or the other.")

    # Set up cache directory
    if cache_dir is not None:
        cache_path = str(cache_dir)
    else:
        # Create cache directory in current working directory
        current_dir = Path.cwd()
        cache_dir_path = current_dir / ".scxpand_cache"
        cache_dir_path.mkdir(exist_ok=True)
        cache_path = str(cache_dir_path)
        logger.info(f"Using cache directory: {cache_dir_path}")

    if model_name is not None:
        # Use registry model
        model_info = get_pretrained_model_info(model_name)

        if not model_info.url:
            raise ValueError(
                f"Download URL not configured for model '{model_name}'. "
                "Please contact the maintainers or use a local model."
            )

        download_url = model_info.url

        logger.info(f"Downloading registry model '{model_name}' from: {download_url}")
    else:
        # Use direct URL
        download_url = model_url

        logger.info(f"Downloading model from URL: {model_url}")

    # Let Pooch handle everything - it will:
    # 1. Download the file if not cached
    # 2. Compute and verify SHA256 hash automatically
    # 3. Auto-detect and extract archives
    # 4. Return path to extracted content
    try:
        model_path = pooch.retrieve(
            url=download_url,
            known_hash=None,  # Let Pooch compute hash automatically on first download
            path=cache_path,
            progressbar=True,
            # Let Pooch auto-detect the processor based on file extension
            processor=pooch.Untar() if download_url.endswith((".tar", ".tar.gz", ".tgz")) else pooch.Unzip(),
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to download model from {download_url}: {e}") from e

    # Handle case where Pooch returns a list of extracted files
    if isinstance(model_path, list):
        # Get the directory containing the extracted files
        result_path = Path(model_path[0]).parent
    else:
        result_path = Path(model_path)
        # If Pooch returned a file path, return its parent directory
        # (since we want the extracted directory, not the archive file)
        if result_path.is_file():
            result_path = result_path.parent

    # Normalize filenames to remove any numeric prefixes
    _normalize_model_filenames(result_path)

    logger.info(f"Model successfully downloaded and cached at: {result_path}")
    return result_path


def download_model(model_name: str, cache_dir: Path | None = None) -> Path:
    """Download a pre-trained model by name from the registry.

    This is a convenience function that wraps the download_pretrained_model functionality
    to make it easier to download models using just the model name.

    Args:
        model_name: Name of the pre-trained model to download
        cache_dir: Custom cache directory (uses `.scxpand_cache` in current dir if None)

    Returns:
        Path to the downloaded model directory

    Raises:
        ValueError: If model_name is not found in registry

    Examples:
        >>> # Download autoencoder model
        >>> model_path = download_model("pan_cancer_autoencoder")
        >>>
        >>> # Download with custom cache directory
        >>> model_path = download_model(
        ...     "pan_cancer_mlp", cache_dir=Path("/my/cache")
        ... )
    """
    # Validate model exists in registry
    get_pretrained_model_info(model_name)

    # Download the model
    return download_pretrained_model(model_name=model_name, cache_dir=cache_dir)
