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

Training with the Command-Line
--------------------------------

You can train any of the supported models directly from the command line.

.. code-block:: bash

   # Autoencoder training
   python -m scxpand.main train --model_type autoencoder --data_path data/example_data.h5ad --n_epochs 100

   # MLP training
   python -m scxpand.main train --model_type mlp --data_path data/example_data.h5ad --n_epochs 50

   # LightGBM training (no epochs needed)
   python -m scxpand.main train --model_type lightgbm --data_path data/example_data.h5ad

   # Logistic Regression training
   python -m scxpand.main train --model_type logistic --data_path data/example_data.h5ad

   # SVM training
   python -m scxpand.main train --model_type svm --data_path data/example_data.h5ad

Using a Configuration File
--------------------------

For more complex configurations, you can use a JSON configuration file to specify parameters for a training run. This is useful for keeping track of different experimental setups.

Pass the file using the ``--config_path`` argument:

.. code-block:: bash

   python -m scxpand.main train --model_type autoencoder --config_path my_ae_config.json

**Parameter Precedence:**
Parameters are loaded in the following order (last one wins):
1. Default parameters in the model's `Param` class.
2. Parameters from your JSON configuration file.
3. Keyword arguments passed directly on the command line (e.g., `--n_epochs 100`).

This means a command-line argument will always override a setting in your config file.

**Example `my_ae_config.json`:**

.. code-block:: json

   {
       "latent_dim": 64,
       "n_epochs": 100,
       "init_learning_rate": 1e-4,
       "encoder_hidden_dims": [128, 64],
       "decoder_hidden_dims": [64, 128],
       "dropout_rate": 0.2
   }

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
