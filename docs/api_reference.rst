API Reference
=============

This section documents the main ways to use scXpand for your analysis.

How to Run Inference
--------------------

Run predictions on new data using trained models:

.. automodule:: scxpand.core.inference
   :members: run_inference
   :show-inheritance:

Command-line interface for inference:

.. autofunction:: scxpand.main.inference

.. automodule:: scxpand.core.prediction
   :members: run_prediction_pipeline
   :show-inheritance:

How to Run Training with a Single Configuration
------------------------------------------------

Train a single model with specific configuration:

.. autofunction:: scxpand.main.train

How to Run Hyperparameter Search
--------------------------------

Optimize model hyperparameters:

.. autofunction:: scxpand.main.optimize

.. autofunction:: scxpand.main.optimize_all

.. automodule:: scxpand.hyperopt.hyperopt_optimizer
   :members: HyperparameterOptimizer
   :show-inheritance:

Pre-trained Models
------------------

Download and use pre-trained models:

.. automodule:: scxpand.pretrained.download_manager
   :members: download_pretrained_model, download_model
   :show-inheritance:

.. automodule:: scxpand.util.model_registry
   :members: list_pretrained_models, get_pretrained_model_info
   :show-inheritance:
