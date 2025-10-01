from __future__ import annotations

import gc
import os
import platform
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from optuna.storages import RDBStorage


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


@contextmanager
def safe_context_manager():
    """
    A context manager that creates and safely cleans up a temporary directory,
    handling file-locking issues on Windows with AnnData and Optuna.
    """
    temp_dir = tempfile.mkdtemp()
    # Track objects that need to be manually closed to avoid file locking issues on Windows
    studies_to_close = []
    adatas_to_close = []
    storage_urls = set()

    class Context:
        def __init__(self, temp_dir_path):
            self.temp_dir = temp_dir_path

        def register_study(self, study):
            """Register a study to be closed."""
            if study is not None:
                studies_to_close.append(study)
                try:
                    if hasattr(study._storage, "url"):
                        storage_urls.add(study._storage.url)
                except (AttributeError, Exception):
                    pass

        def register_adata(self, adata):
            """Register an AnnData object to be closed."""
            if adata is not None:
                adatas_to_close.append(adata)

    try:
        yield Context(temp_dir)
    finally:
        # --- Cleanup Phase ---
        for adata in adatas_to_close:
            close_adata_safely(adata)
        for study in studies_to_close:
            close_optuna_storage(study)

        if platform.system() == "Windows":
            for url in storage_urls:
                try:
                    RDBStorage.clear_instance_cache(url)
                except Exception:
                    pass

        studies_to_close.clear()
        adatas_to_close.clear()
        storage_urls.clear()

        gc.collect()
        gc.collect()
        time.sleep(0.2)

        # Now, safely remove the temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=platform.system() != "Windows")
        except PermissionError:
            # Fallback to robust remove for stubborn files on Windows
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    robust_remove(os.path.join(root, name))
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
        except Exception:
            pass


def close_adata_safely(adata):
    """Safely close an AnnData file to avoid Windows file lock issues."""
    if adata is None or not hasattr(adata, "file") or not hasattr(adata.file, "close"):
        return
    try:
        adata.file.close()
    except Exception:
        pass  # Ignore errors during cleanup
    del adata


def close_optuna_storage(study):
    """Safely close Optuna storage to avoid Windows file lock issues."""
    if study is None:
        return

    # Nullify references to help the garbage collector
    try:
        study._storage = None
        study._study_id = -1  # Invalidate study
    except (AttributeError, Exception):
        pass  # Ignore all errors

    # Aggressive garbage collection and a delay
    gc.collect()
    gc.collect()
    time.sleep(0.1)
    del study
    gc.collect()
