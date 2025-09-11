Data Utilities Module
=====================

The data_util module provides utilities for data loading, preprocessing, formatting, and preparation for model training.

Data Preparation
----------------

Core data preparation pipeline for model training:

.. automodule:: scxpand.data_util.prepare_data_for_train
   :members: prepare_data_for_training, TrainingDataBundle
   :show-inheritance:

Data Format
-----------

Data format specification and preprocessing parameters for consistent data handling:

.. automodule:: scxpand.data_util.data_format
   :members: DataFormat, load_data_format
   :show-inheritance:

Data Splitting
--------------

Patient-level stratified data splitting utilities:

.. automodule:: scxpand.data_util.data_splitter
   :members: split_data, get_patient_identifiers, validate_patient_cancer_types
   :show-inheritance:

Dataset and Data Loaders
-------------------------

PyTorch dataset and dataloader implementations for efficient data loading:

.. automodule:: scxpand.data_util.dataset
   :members: CellsDataset
   :show-inheritance:

.. automodule:: scxpand.data_util.dataloaders
   :members: create_eval_dataloader, create_train_dataloader
   :show-inheritance:

Data Transforms
---------------

Data transformation utilities for preprocessing and feature extraction:

.. automodule:: scxpand.data_util.transforms
   :members: extract_is_expanded, load_and_preprocess_data_numpy
   :show-inheritance:

Statistics
----------

Statistical computation utilities for data normalization:

.. automodule:: scxpand.data_util.statistics
   :members: compute_preprocessed_genes_means_stds
   :show-inheritance:
