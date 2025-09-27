Installation Guide
=================

.. note::

   This guide covers two installation methods:

   - **Installing the Published Package** (recommended for most users):
     Use this if you want to use scXpand for analysis or inference.
   - **Local Development Setup** (for contributors/developers):
     Use this if you want to contribute to scXpand or work with the latest source code from GitHub.

Installing the Published Package
--------------------------------

scXpand is available in two variants to match your hardware:

If you have an NVIDIA GPU with CUDA support:

With plain *pip* (add CUDA index):

.. code-block:: bash

   pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128

With **uv**:

.. code-block:: bash

   uv pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

Otherwise (CPU, Apple Silicon, or non-CUDA GPUs):

With plain *pip*:

.. code-block:: bash

   pip install --upgrade scxpand

With **uv**:

.. code-block:: bash

   uv pip install --upgrade scxpand


Development Setup (from Source)
-------------------------------

To work with the latest version on GitHub (for development or contributions):

.. code-block:: bash

    git clone https://github.com/yizhak-lab-ccg/scXpand.git
    cd scXpand

scXpand follows the `scverse ecosystem <https://scverse.org/>`_ standards and uses modern Python packaging tools.

**Install in development mode:**

.. code-block:: bash

    uv pip install -e ".[dev]"

This will:

* Install scXpand in editable mode
* Install all development dependencies (testing, linting, documentation)
* Set up PyTorch with automatic backend detection (CUDA if available, otherwise CPU/MPS)

**Install pre-commit hooks:**

.. code-block:: bash

    pre-commit install

**For PyTorch installation with uv (Recommended):**

uv provides excellent PyTorch integration with automatic backend selection:

.. code-block:: bash

    # Install PyTorch with automatic backend detection
    uv pip install torch torchvision torchaudio --torch-backend=auto

This will:
* Automatically detect your system's optimal PyTorch backend
* Install CUDA PyTorch if NVIDIA GPU is available
* Fall back to CPU/MPS PyTorch if CUDA is not available
* Use the most compatible PyTorch index for your system

**Alternative: Manual PyTorch installation with uv:**

For specific CUDA versions:

.. code-block:: bash

    # CUDA 12.8 (latest)
    uv pip install torch torchvision torchaudio --torch-backend=cu128

    # CUDA 12.6
    uv pip install torch torchvision torchaudio --torch-backend=cu126

    # CPU-only
    uv pip install torch torchvision torchaudio --torch-backend=cpu

**Using uv for the entire development setup:**

If you prefer using uv for everything:

.. code-block:: bash

    # Install scXpand in development mode with uv
    uv pip install -e ".[dev]"

    # Then install PyTorch with optimal backend
    uv pip install torch torchvision torchaudio --torch-backend=auto

    # Install pre-commit hooks
    pre-commit install
