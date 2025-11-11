#!/usr/bin/env python3
"""
Verify PyTorch installation and print backend information.

This script checks what PyTorch backend is available after uv sync has completed.
It does NOT attempt to install torch; installation is handled via pyproject.toml
configuration and `uv sync` in the main install script.
"""

import sys


def check_torch_backend() -> bool:
    """Check if torch is installed and what backend is available."""
    try:
        import torch  # type: ignore

        print(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✓ PyTorch successfully installed with CUDA backend.")
            return True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✓ PyTorch successfully installed with MPS backend.")
            return True
        else:
            print("⚠ PyTorch installed but CUDA/MPS is not available.")
            print("This may be due to:")
            print("  - NVIDIA drivers not installed or outdated")
            print("  - CUDA toolkit not installed")
            print("  - GPU not compatible with current CUDA version")
            print("  - Running on CPU-only system")
            return True  # Still successful, just CPU-only
    except ImportError:
        print("❌ Error: Failed to import torch after installation.")
        return False


def print_torch_info() -> None:
    """Print detailed PyTorch environment information."""
    try:
        import torch
    except ImportError:
        print(
            "Error: torch is not installed. Please run 'uv sync' to install dependencies."
        )
        return

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA version in torch: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Python executable: {sys.executable}")


if __name__ == "__main__":
    print("Verifying PyTorch installation...\n")

    if check_torch_backend():
        print_torch_info()
        print("\n✓ PyTorch verification complete.")
        sys.exit(0)
    else:
        print("\n❌ PyTorch verification failed.")
        sys.exit(1)
