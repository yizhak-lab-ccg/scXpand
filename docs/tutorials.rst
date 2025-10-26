Tutorials
=========

We provide a variety of tutorials to help you get started with scXpand:

- :doc:`Getting Started with scXpand <notebooks/scxpand_tutorial>` - Learn how to prepare scRNA-seq data and apply scXpand models for T cell expansion prediction using a breast cancer dataset example.

- :doc:`Data Preparation for Model Training <notebooks/data_preparation_tutorial>` - Complete pipeline for preparing paired scRNA/TCR-seq data from scratch, including quality control, MAGIC imputation, automatic cutoff determination for cell type classification, and expansion labeling using a brain tumor dataset.

- :doc:`Model Inference and Evaluation Pipeline <notebooks/inference>` - Discover how to load trained models, run inference on new data, and evaluate performance using ROC curves and AUROC metrics across different tissue types and labels.

- :doc:`Autoencoder Embedding Visualization <notebooks/embeddings>` - Explore how to generate and visualize latent representations from autoencoder models, coloring plots by expansion status and tissue type for biological insights.

.. toctree::
   :maxdepth: 1

   notebooks/scxpand_tutorial
   notebooks/data_preparation_tutorial
   notebooks/inference
   notebooks/embeddings
