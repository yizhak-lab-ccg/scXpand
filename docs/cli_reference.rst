Command-Line Interface (CLI) Reference
=======================================

This page provides a complete reference for the command-line interface, which is the quickest way to get started with training, optimization, and inference.

All commands are accessible through ``python -m scxpand.main <COMMAND>``.

Main Commands
-------------

Train a Model
~~~~~~~~~~~~~

.. autofunction:: scxpand.main.train

Run Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scxpand.main.optimize

Run Optimization for All Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scxpand.main.optimize_all

Run Inference
~~~~~~~~~~~~~

.. autofunction:: scxpand.main.inference

List Pre-trained Models
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scxpand.util.model_registry.list_pretrained_models
