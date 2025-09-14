Data Utilities
===============

The data utilities module provides functions for data preparation and preprocessing. These are primarily used for advanced workflows or when preparing custom datasets for training.

.. note::
   Most users will not need to use these functions directly. The main :func:`scxpand.run_inference` function handles data preprocessing automatically.

Data Format
-----------

Data format specification for consistent data handling across the pipeline:

.. automodule:: scxpand.data_util.data_format
   :members: DataFormat, load_data_format
   :show-inheritance:

Data Preparation
----------------

Core data preparation pipeline for model training:

.. automodule:: scxpand.data_util.prepare_data_for_train
   :members: prepare_data_for_training, TrainingDataBundle
   :show-inheritance:

Data Preprocessing
------------------

Data transformation utilities for preprocessing and feature extraction:

.. automodule:: scxpand.data_util.transforms
   :members: extract_is_expanded, load_and_preprocess_data_numpy
   :show-inheritance:
