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


def has_cuda_support():
    """Check if CUDA is available on the system."""
    try:
        # Check for NVIDIA drivers and CUDA toolkit
        if sys.platform.startswith("linux"):
            # On Linux, check for nvidia-smi and CUDA
            result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ NVIDIA GPU detected with nvidia-smi")
                return True
        elif sys.platform == "win32":
            # On Windows, check for NVIDIA drivers
            result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print("✓ NVIDIA GPU detected with nvidia-smi")
                return True

        print("i No CUDA support detected (no NVIDIA GPU or drivers)")
        return False
    except (subprocess.SubprocessError, FileNotFoundError):
        print("i No CUDA support detected (nvidia-smi not found)")
        return False


def install_torch_with_optimal_backend():
    """Install PyTorch with optimal backend using uv's project interface."""
    print("Configuring PyTorch backend for optimal performance...")

    try:
        # Check if CUDA is available on the system
        if has_cuda_support():
            print("CUDA support detected. Installing CUDA-enabled PyTorch...")
            # Install with CUDA extra and preserve dev/docs extras
            subprocess.check_call(["uv", "sync", "--extra", "cuda", "--extra", "dev", "--extra", "docs"])
            print("✓ PyTorch installation completed with CUDA support.")
        else:
            print("No CUDA support detected. Installing CPU-only PyTorch...")
            # Install with CPU extra and preserve dev/docs extras
            subprocess.check_call(["uv", "sync", "--extra", "cpu", "--extra", "dev", "--extra", "docs"])
            print("✓ PyTorch installation completed with CPU/MPS support.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch with backend selection: {e}")
        print("Falling back to default PyTorch installation...")
        try:
            # Fallback to default sync with dev/docs extras preserved
            subprocess.check_call(["uv", "sync", "--extra", "dev", "--extra", "docs"])
            print("✓ PyTorch installed with default backend (fallback).")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed to install PyTorch with fallback method: {e2}")
            return False

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
