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

    pip install -e ".[dev]"

This will:

* Install scXpand in editable mode
* Install all development dependencies (testing, linting, documentation)
* Set up PyTorch with automatic backend detection (CUDA if available, otherwise CPU/MPS)

**Install pre-commit hooks:**

.. code-block:: bash

    pre-commit install

**For CUDA development setup:**

If you need specific CUDA PyTorch installation for development:

.. code-block:: bash

    python scripts/install_torch_for_dev.py

This script will:

* Detect your system's optimal PyTorch backend
* Install CUDA PyTorch if NVIDIA GPU is available
* Fall back to CPU/MPS PyTorch if CUDA is not available
* Update the environment with the correct PyTorch version

**Alternative: Using uv (if you have it installed):**

.. code-block:: bash

    uv pip install -e ".[dev]"
    pre-commit install
