Autoencoders Module
===================

The autoencoders module provides autoencoder-based models for single-cell expansion prediction.

Model Architectures
-------------------

Autoencoder model implementations with reconstruction and classification heads:

.. automodule:: scxpand.autoencoders.ae_models
   :members: AutoencoderModel
   :show-inheritance:

Model Components
----------------

Neural network modules and components for autoencoder architecture:

.. automodule:: scxpand.autoencoders.ae_modules
   :members: Encoder, Decoder, ClassifierHead
   :show-inheritance:

Loss Functions
--------------

Custom loss functions for autoencoder training:

.. automodule:: scxpand.autoencoders.ae_losses
   :members: AutoencoderLoss
   :show-inheritance:

Model Output
------------

Output handling and processing for autoencoder predictions:

.. automodule:: scxpand.autoencoders.ae_model_output
   :members: AutoencoderOutput
   :show-inheritance:

Parameters
----------

Parameter classes for autoencoder configuration:

.. automodule:: scxpand.autoencoders.ae_params
   :members: AutoencoderParam
   :show-inheritance:

Training
--------

Training utilities and pipeline for autoencoder models:

.. automodule:: scxpand.autoencoders.ae_trainer
   :members: AutoencoderTrainer
   :show-inheritance:

Training Runner
--------------

Main training execution function:

.. automodule:: scxpand.autoencoders.run_ae_train
   :members: run_ae_training
   :show-inheritance:
