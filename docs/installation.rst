Installation
============

To use scXpand from another project, install it using pip:

.. code-block:: bash

    pip install scxpand

Development Version
-------------------

To work with the latest version on GitHub: clone the repository and ``cd`` into its root directory.

.. code-block:: bash

    git clone https://github.com/ronamit/scxpand.git
    cd scxpand

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
        data_path="your_data.h5ad",
        device=None  # Auto-detect best available device
    )

Or use the command line interface:

.. code-block:: bash

      # List pre-trained models
        scxpand list-models

        # Run inference with pre-trained model
        scxpand predict --data_path your_data.h5ad --model_name pan_cancer_autoencoder

        # Run inference with local model
        scxpand predict --data_path your_data.h5ad --model_path results/my_model
