Model Training
==============

Learn how to train scXpand models for T-cell expansion prediction.

Overview
--------

scXpand supports five model architectures for training:

* **Neural Networks**: Autoencoder (with reconstruction + classification) and MLP (direct prediction)
* **Gradient Boosting**: LightGBM (optimized for tabular data)
* **Linear Models**: Logistic Regression and SVM (linear classifiers)

For detailed architecture descriptions, configuration options, and scientific foundations, see :doc:`model_architectures`.

Training API Reference
----------------------

Train a single model with specific configuration:

.. autofunction:: scxpand.main.train

Training Monitoring
-------------------

Monitor training progress with TensorBoard:

.. code-block:: bash

   # Start TensorBoard (view all training runs)
   tensorboard --logdir=results/

   # Or view a specific model type
   tensorboard --logdir=results/pan_cancer_autoencoder_v_0/

   # Access dashboard at http://localhost:6006

For detailed CLI examples and usage, see the `scxpand.main` module documentation.
