Utilities Module
================

The util module contains various utility functions for inference, metrics, plotting, training orchestration, and general operations.

Core Modules
------------

High-level domain logic and orchestration:

.. automodule:: scxpand.core.prediction
   :members: run_prediction_pipeline
   :show-inheritance:

.. automodule:: scxpand.core.evaluation
   :members: evaluate_predictions_and_save
   :show-inheritance:

Training Utilities
-------------------

Common utilities for training operations including validation and function routing:

.. automodule:: scxpand.util.training_utils
   :members: validate_and_setup_common, call_training_function
   :show-inheritance:

Inference Utilities
-------------------

Core inference functions for model loading and running predictions:

.. automodule:: scxpand.util.inference_utils
   :members: load_model, run_inference, setup_inference_environment
   :show-inheritance:

General Utilities
-----------------

General utility functions for device management, path handling, and parameter loading:

.. automodule:: scxpand.util.general_util
   :members: get_device, get_new_version_path, load_and_override_params
   :show-inheritance:

File I/O Operations
-------------------

File input/output utilities for data loading and result saving:

.. automodule:: scxpand.util.io
   :members: load_eval_indices, save_predictions_to_csv
   :show-inheritance:

Model Type Management
--------------------

Utilities for model type validation and conversion:

.. automodule:: scxpand.util.classes
   :members: ModelType, ensure_model_type
   :show-inheritance:

Metrics and Evaluation
----------------------

Performance metrics calculation and evaluation utilities:

.. automodule:: scxpand.util.metrics
   :members: calculate_metrics, metrics_dict_to_table
   :show-inheritance:

Visualization and Plotting
--------------------------

Plotting utilities for results visualization:

.. automodule:: scxpand.util.plots
   :members: plot_roc_curve, plot_pr_curve
   :show-inheritance:

Logging
-------

Logging utilities for consistent log management:

.. automodule:: scxpand.util.logger
   :members: get_logger
   :show-inheritance:
