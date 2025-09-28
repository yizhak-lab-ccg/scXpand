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

**Prerequisites:**

Before setting up the development environment, you need to install `uv`, a fast Python package installer and resolver:

* **Install uv**: Follow the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_ for your platform
* **Verify installation**: Run ``uv --version`` to confirm uv is installed

**Install in development mode:**


To set up a development environment, run the appropriate script for your system (now located in the main folder):

* **For Linux or macOS:**

   .. code-block:: bash

         ./install.sh

* **For Windows (command prompt):**

   .. code-block:: bash

         install.bat

These scripts will automatically:
1. Create a virtual environment using `uv`.
2. Activate the environment.
3. Install scXpand in editable mode with all development dependencies.
4. Install PyTorch with the optimal backend for your system.
5. Set up pre-commit hooks.

After the script completes, the development environment will be ready to use.

To activate the virtual environment in a new terminal session, run one of the following commands depending on your shell:

.. code-block:: shell

   # On Linux/macOS (bash/zsh):
   source .venv/bin/activate

   # On Windows (Command Prompt):
   .venv\Scripts\activate.bat
