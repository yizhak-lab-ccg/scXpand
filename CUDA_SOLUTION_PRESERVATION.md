# CUDA PyTorch Installation Solution Preservation

This document describes the CUDA PyTorch installation solution that has been preserved during the scverse migration.

## Overview

scXpand uses a sophisticated dual-package system to provide both CPU and CUDA versions:

- **`scxpand`**: Standard package with CPU/MPS PyTorch support
- **`scxpand-cuda`**: CUDA-enabled package with GPU acceleration

## Key Components

### 1. Dual Package Configuration (`pyproject.toml`)

```toml
[project.optional-dependencies]
cpu = [
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
]
cuda = [
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
]

[tool.uv]
conflicts = [
    [[{ extra = "cpu" }, { extra = "cuda" }]]
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu", extra = "cpu" }
torchvision = { index = "pytorch-cpu", extra = "cpu" }
torchaudio = { index = "pytorch-cpu", extra = "cpu" }
```

### 2. Release Automation (`scripts/`)

- **`release.sh`**: Automated dual package release
- **`create_cuda_pyproject.py`**: Creates CUDA variant for publishing
- **`constants.py`**: Centralized CUDA/torch version management
- **`install_torch_for_dev.py`**: Development environment setup

### 3. Installation Instructions

#### For Users (PyPI):
```bash
# CPU version
pip install scxpand

# CUDA version
pip install scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128
```

#### For Developers:
```bash
# Install with optimal backend detection
python scripts/install_torch_for_dev.py
```

## Protection During Template Updates

This solution is protected from being overwritten by scverse template updates through:

1. **Cruft Skip List**: Added to `.cruft.json`:
   ```json
   "skip": [
     "scripts/**",
     "pyproject.toml"
   ]
   ```

2. **Backup Files**:
   - `pyproject.toml.cuda_backup` - Full backup of CUDA configuration
   - This documentation file

3. **Version Management**: Centralized in `scripts/constants.py`:
   ```python
   CUDA_VERSION = "cu128"  # central CUDA version spec
   TORCH_VERSION = "2.8.0"  # central Torch version spec
   ```

## Testing CUDA Installation

After any changes, verify CUDA installation works:

```bash
# Test CUDA wheel creation
python scripts/create_cuda_pyproject.py --dry-run

# Test release process
./scripts/release.sh --dry-run

# Test development setup
python scripts/install_torch_for_dev.py
```

## Key Benefits

1. **Automatic Backend Detection**: Uses `uv`'s `--torch-backend=auto`
2. **Dual Package Publishing**: Single command publishes both CPU and CUDA versions
3. **Version Consistency**: Centralized version management
4. **Platform Support**: Works on Windows, Linux, macOS (with appropriate fallbacks)
5. **Development Friendly**: Easy local development setup

## Migration Safety

This solution has been preserved during the scverse migration by:

- Adding CUDA-specific files to cruft skip list
- Creating backup configurations
- Maintaining all release automation scripts
- Preserving UV configuration in pyproject.toml

The CUDA solution will continue to work exactly as before, even after scverse template updates.
