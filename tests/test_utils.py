import gc
import os
import platform
import time

from pathlib import Path

import pytest


def robust_remove(file_path, max_retries=10, delay=0.2):
    """Try to remove a file with retries to avoid Windows file lock issues."""
    for _ in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                gc.collect()  # Extra GC after removal
            return
        except PermissionError:
            gc.collect()  # Try to force release before retry
            time.sleep(delay)
    # Final attempt: wait and gc, then try one last time
    gc.collect()
    time.sleep(1)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            gc.collect()
        return
    except PermissionError:
        pass
    pytest.skip(f"Could not remove file after retries: {file_path}")


def create_temp_h5ad_file(adata, temp_dir: str) -> str:
    """Create a temporary H5AD file with proper error handling and Windows-safe cleanup."""
    temp_path = Path(temp_dir)
    test_file_path = temp_path / "test_data.h5ad"

    try:
        temp_path.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(test_file_path)

        if platform.system() == "Windows":
            # On Windows, we may need to wait for the file to be released
            del adata  # Explicitly delete AnnData object
            gc.collect()
            for _ in range(10):
                try:
                    with open(test_file_path, "rb"):
                        pass
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.2)

        return str(test_file_path)
    except Exception as e:
        pytest.skip(f"Could not create H5AD file: {e}")


def close_adata_safely(adata):
    """Safely close AnnData file handle to avoid Windows file locks."""
    try:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        # Force garbage collection
        del adata
        gc.collect()
    except (AttributeError, OSError):
        # File might already be closed or not backed
        pass


def ensure_h5ad_handles_closed(*adatas):
    """Ensure all H5AD file handles are closed for Windows compatibility."""
    for adata in adatas:
        if adata is not None:
            close_adata_safely(adata)
    gc.collect()  # Force cleanup


def windows_safe_tempfile_cleanup(temp_dir, *file_patterns):
    """Clean up temporary files in a Windows-safe manner."""
    if platform.system() == "Windows":
        # Extra wait for Windows
        time.sleep(0.5)
        gc.collect()

    # Remove specific files if patterns provided
    temp_path = Path(temp_dir)
    for pattern in file_patterns:
        for file_path in temp_path.glob(pattern):
            if file_path.exists():
                robust_remove(str(file_path))


def windows_safe_context_manager():
    """Context manager for Windows-safe test execution."""
    return WindowsSafeTestContext()


class WindowsSafeTestContext:
    """Context manager to handle Windows file lock issues in tests."""

    def __init__(self):
        self.adatas_to_close = []
        self.files_to_cleanup = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close all AnnData handles
        for adata in self.adatas_to_close:
            close_adata_safely(adata)

        # Force garbage collection
        gc.collect()

        # Wait a bit on Windows
        if platform.system() == "Windows":
            time.sleep(0.5)

        # Clean up files
        for file_path in self.files_to_cleanup:
            robust_remove(file_path)

    def register_adata(self, adata):
        """Register an AnnData object for cleanup."""
        if adata is not None:
            self.adatas_to_close.append(adata)
        return adata

    def register_file(self, file_path):
        """Register a file for cleanup."""
        self.files_to_cleanup.append(str(file_path))
        return file_path
