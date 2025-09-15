User Guide
==========

Welcome to the scXpand User Guide! This comprehensive guide follows the typical user workflow for T-cell expansion prediction using single-cell RNA sequencing data.

Quick Start
-----------

Get up and running with scXpand in minutes:

.. code-block:: python

   import scxpand

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

For more detailed workflows, see the sections below.

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Setup & Installation

   installation

.. toctree::
   :maxdepth: 2
   :caption: Data Preparation

   data_format
   data_pipeline

.. toctree::
   :maxdepth: 2
   :caption: Using Pre-trained Models

   model_inference

.. toctree::
   :maxdepth: 2
   :caption: Training Your Own Models

   model_training
   data_splitting
   hyperparameter_optimization

.. toctree::
   :maxdepth: 2
   :caption: Understanding Models & Results

   model_architectures
   evaluation_metrics
   output_format
