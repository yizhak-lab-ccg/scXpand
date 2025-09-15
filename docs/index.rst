scXpand Documentation
=====================

scXpand is a framework for predicting T-cell clonal expansion from single-cell RNA sequencing data. It provides multiple methods for training and inference.

.. image:: _static/images/scXpand_datasets.jpeg
   :width: 100%
   :alt: scXpand Datasets Overview

.. toctree::
   :hidden:
   :maxdepth: 1

   user_guide
   tutorials
   contributing
   api/index

..
   The toctree below is for structuring the landing page, not the sidebar.

Features
--------

* **Multiple Model Architectures**: Autoencoder, MLP, LightGBM, Logistic Regression, and SVM
* **Scalable Processing**: Handles millions of cells with memory-efficient data streaming
* **Hyperparameter Optimization**: Built-in hyperparameter search for model tuning

Quick Start
-----------

**Installation:**

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

See the `full installation guide <https://scxpand.readthedocs.io/en/latest/installation.html>`_ for detailed setup instructions.

.. code-block:: python

   import scxpand
   # Make sure that "your_data.h5ad" includes only T cells for the results to be meaningful
   # Ensure that "your_data.var_names" are provided as Ensembl IDs (as the pre-trained models were trained using this gene representation)
   # Please refer to our documentation for more information

   # List available pre-trained models
   scxpand.list_pretrained_models()

   # Run inference with automatic model download
   results = scxpand.run_inference(
       model_name="pan_cancer_autoencoder",  # default model
       data_path="your_data.h5ad"
   )

   # Access predictions
   predictions = results.predictions
   if results.has_metrics:
       print(f"AUROC: {results.get_auroc():.3f}")

   # Or use the command line
   # scxpand inference --data_path your_data.h5ad --model_name pan_cancer_autoencoder

See :doc:`installation` for different installation methods and :doc:`user_guide` for comprehensive usage instructions.

Tutorials
------------
We provide a variety of tutorials to help you get started with scXpand:

- :doc:`Getting Started with scXpand <notebooks/scxpand_tutorial>` - Learn how to prepare scRNA-seq data and apply scXpand models for T cell expansion prediction using a breast cancer dataset example.

- :doc:`Model Inference and Evaluation Pipeline <notebooks/inference>` - Discover how to load trained models, run inference on new data, and evaluate performance using ROC curves and AUROC metrics across different tissue types and labels.

- :doc:`Autoencoder Embedding Visualization <notebooks/embeddings>` - Explore how to generate and visualize latent representations from autoencoder models, coloring plots by expansion status and tissue type for biological insights.

Support and Contact
-------------------
This project was created in favor of the scientific community worldwide, with a special dedication to the cancer research community.
We hope you’ll find this repository helpful, and we warmly welcome any requests or suggestions - please don’t hesitate to reach out!

Citation
--------
If you use scXpand in your research, please cite our paper:

Under preparation.
