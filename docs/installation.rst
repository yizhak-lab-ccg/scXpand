Installation Guide
==================

.. note::

   This guide covers two installation methods:

   - **Installing the Published scXpand Package** (recommended for most users):
     Use this if you want to use scXpand for analysis or inference.
   - **Local Development Setup** (for contributors/developers):
     Use this if you want to contribute to scXpand or work with the source code from GitHub.

Installing the Published scXpand Package
--------------------------------

Option 1: Using Virtual Environment with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Create and activate virtual environment**

.. code-block:: bash

   python -m venv scxpand-env
   source scxpand-env/bin/activate  # On Windows: scxpand-env\Scripts\activate

**2. Install scXpand**

scXpand is available in two variants to match your hardware:

- **If you have an NVIDIA GPU with CUDA support:**

  .. code-block:: bash

     pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128

- **Otherwise (CPU, Apple Silicon, or non-CUDA GPUs):**

  .. code-block:: bash

     pip install --upgrade scxpand

Option 2: Using Conda with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Create and activate conda environment**

.. code-block:: bash

   conda create -n scxpand-env python=3.11
   conda activate scxpand-env

**2. Install scXpand**

scXpand is available in two variants to match your hardware:

- **If you have an NVIDIA GPU with CUDA support:**

  .. code-block:: bash

     pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128

- **Otherwise (CPU, Apple Silicon, or non-CUDA GPUs):**

  .. code-block:: bash

     pip install --upgrade scxpand

Option 3: Using the `uv <https://docs.astral.sh/uv/>`_ package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Install uv**

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager.

→ `Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_

**Windows users**: First install `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ if not already installed (required for compiling certain Python packages).

**2. Create and activate virtual environment**

.. code-block:: bash

   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

**3. Install scXpand**

scXpand is available in two variants to match your hardware:

- **If you have an NVIDIA GPU with CUDA support:**

  .. code-block:: bash

     uv pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

- **Otherwise (CPU, Apple Silicon, or non-CUDA GPUs):**

  .. code-block:: bash

     uv pip install --upgrade scxpand


Development Setup (from Source)
-------------------------------

To work with the latest version on GitHub (for development or contributions):

.. code-block:: bash

    git clone https://github.com/yizhak-lab-ccg/scXpand.git
    cd scXpand

scXpand uses `uv <https://docs.astral.sh/uv/>`_ for fast, reliable dependency management.

**Windows users**: Before proceeding, install `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ if not already installed.

Use the provided install scripts:

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

    # Windows Command Prompt:
    .\.venv\Scripts\activate
