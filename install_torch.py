import os
import subprocess
import sys


def is_in_virtual_environment():
    """Check if we're running in a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix")
        and sys.base_prefix != sys.prefix
        and os.path.exists(os.path.join(sys.prefix, "pyvenv.cfg"))
    )


def install_torch_with_optimal_backend():
    """Install torch with optimal backend using uv's auto backend selection."""
    print("Installing PyTorch with auto backend selection...")

    try:
        # Use uv's auto backend selection with reinstall to override existing installation
        # Set environment variable as documented in the uv PyTorch guide
        env = os.environ.copy()
        env["UV_TORCH_BACKEND"] = "auto"

        subprocess.check_call(
            ["uv", "pip", "install", "torch", "--torch-backend=auto", "--reinstall-package", "torch"], env=env
        )
        print("✓ PyTorch installation completed with auto backend selection.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch with auto backend: {e}")
        print("Trying CPU-only installation as fallback...")
        try:
            # Fallback to CPU-only torch
            subprocess.check_call(["uv", "pip", "install", "torch", "--reinstall-package", "torch"])
            print("✓ PyTorch installed with CPU backend (fallback).")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed to install PyTorch with fallback method: {e2}")
            return False

    # Update lock file to reflect the installation
    print("Updating lock file to include torch installation...")
    try:
        subprocess.check_call(["uv", "lock"])
        print("✓ Lock file updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Failed to update lock file: {e}")

    return True


def check_torch_backend():
    """Check if torch is installed and what backend is available."""
    try:
        import torch  # type: ignore # noqa: PLC0415

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


def print_torch_backend():
    """Print information about the PyTorch backend in use."""
    try:
        import torch  # type: ignore # noqa: PLC0415
    except ImportError:
        print("Error: torch is not installed. Please run 'uv sync' to install dependencies.")
        return

    print(f"PyTorch version: {torch.__version__}")

    backend = getattr(torch, "backend", None)
    if backend is not None:
        print(f"Torch backend in use: {backend}")

    # Check for various backends in order of preference
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        print(f"Torch is using CUDA (GPU): {device_name}")
        print(f"Number of CUDA devices available: {device_count}")
        print(f"CUDA version: {cuda_version}")

        # Check for CUDA 12.8+ support
        if cuda_version and "12." in cuda_version:
            print("✓ CUDA 12.x detected - compatible with latest PyTorch features")
        else:
            print("⚠ Older CUDA version detected - consider upgrading for optimal performance")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Torch is using MPS (Metal Performance Shaders) backend")
        print("Running on Apple Silicon with GPU acceleration")

    else:
        print("Torch is using CPU backend")

    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")


if __name__ == "__main__":
    # Install torch with optimal backend
    if install_torch_with_optimal_backend():
        # Check if installation was successful
        if check_torch_backend():
            # Print detailed backend info
            print_torch_backend()
        else:
            print("❌ Torch installation failed. Please check the error messages above.")
    else:
        print("❌ Failed to install torch. Please check the error messages above.")
