Hyperparameter Optimization
============================

The hyperparameter optimization module provides tools for automated hyperparameter tuning using Optuna.

.. note::
   Most users will use the command-line interface for hyperparameter optimization: ``scxpand optimize --model_type autoencoder --n_trials 100 --data_path data.h5ad``

Hyperparameter Optimizer
-------------------------

Main class for running hyperparameter optimization studies:

.. automodule:: scxpand.hyperopt.hyperopt_optimizer
   :members: HyperparameterOptimizer
   :show-inheritance:
