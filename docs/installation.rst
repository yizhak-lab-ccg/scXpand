Installation Guide
==================

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

   uv pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128

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

scXpand uses `uv <https://docs.astral.sh/uv/>`_ for fast, reliable dependency management. Use the provided install scripts:

**macOS/Linux:**

.. code-block:: bash

    ./install.sh

**Windows Command Prompt:**

.. code-block:: bash

    .\install.bat

These scripts will:

* Install Python 3.13 via uv
* Create a virtual environment in ``.venv/``
* Install all dependencies from the lock file
* Set up PyTorch with appropriate GPU support
* Register Jupyter kernel
* Set up pre-commit hooks

Then activate the environment:

.. code-block:: bash

    # macOS/Linux:
    source .venv/bin/activate

    # Window Command Prompt:
    .\.venv\Scripts\activate
