Installation Guide
=================

.. note::

   **There are two ways to install scXpand:**

   - **Standard Installation** (recommended for most users):
     Use this if you simply want to use scXpand for analysis or inference.
   - **Development Setup** (for contributors/developers):
     Use this if you want to contribute to scXpand or work with the latest source code from GitHub.

Standard Installation
---------------------

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

scXpand uses `uv <https://docs.astral.sh/uv/>`_ for fast, reliable dependency management. Use the provided install scripts:

**macOS/Linux:**

.. code-block:: bash

    ./install.sh

**Windows PowerShell:**

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

    # macOS/Linux
    source .venv/bin/activate

    # Windows PowerShell
    .\.venv\Scripts\activate

Quick Start
-----------

After installation, you can immediately start using scXpand:

.. code-block:: python

    import scxpand
    # Make sure that "your_data.h5ad" includes only T cells for the results to be meaningful
    # Ensure that "your_data.var_names" are provided as Ensembl IDs (as the pre-trained models were trained using this gene representation)
    # Please refer to our documentation for more information

    # List available pre-trained models
    scxpand.list_pretrained_models()

    # Run inference with a pre-trained model
    results = scxpand.run_inference(
        model_name="pan_cancer_autoencoder",
        data_path="your_data.h5ad"
    )

Or use the command line interface:

.. code-block:: bash

    # List pre-trained models
    scxpand list-models

    # Run inference with pre-trained model
    scxpand predict --data_path your_data.h5ad --model_name pan_cancer_autoencoder

    # Run inference with local model
    scxpand predict --data_path your_data.h5ad --model_path results/my_model
