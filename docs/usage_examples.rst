Usage Examples
==============

This document provides usage examples for scXpand, including model training, hyperparameter optimization, and configuration management.

Model Training
--------------

Basic Training
~~~~~~~~~~~~~~

Train a single model with default parameters:

.. code-block:: bash

   # Basic training
   python -m scxpand.main train \
     --model_type mlp \
     --data_path data/example_data.h5ad

   # With custom parameters
   python -m scxpand.main train \
     --model_type autoencoder \
     --data_path data/example_data.h5ad \
     --n_epochs 10 \
     --batch_size 1024

Available Model Types
~~~~~~~~~~~~~~~~~~~~~

* ``mlp`` - Multi-layer perceptron (fully connected neural network)
* ``autoencoder`` - Autoencoder-based classifier with reconstruction and classification heads
* ``lightgbm`` - LightGBM gradient boosting (ensemble of decision trees)
* ``logistic`` - Logistic regression model trained with logistic loss
* ``svm`` - Support vector machine trained with the hinge loss

Hyperparameter Optimization
----------------------------

Optimize model hyperparameters using Optuna. The system saves progress and can resume from previous studies.

.. code-block:: bash

   # Run an optimization study for 100 trials
   python -m scxpand.main optimize \
     --model_type mlp \
     --data_path data/example_data.h5ad \
     --n_trials 100 \
     --study_name "mlp_optimization"

   # To resume the same study and add 50 more trials (default behavior)
   python -m scxpand.main optimize \
     --model_type mlp \
     --data_path data/example_data.h5ad \
     --n_trials 50 \
     --study_name "mlp_optimization"

   # To start a fresh study, you must manually delete the old study directory
   # or use the --resume False flag, which will raise an error if the study exists.
   python -m scxpand.main optimize \
     --model_type mlp \
     --data_path data/example_data.h5ad \
     --n_trials 100 \
     --study_name "mlp_optimization" \
     --resume False

   # Optimize all available models
   python -m scxpand.main optimize-all \
     --data_path data/example_data.h5ad \
     --n_trials 50 \
     --num_workers 6

Configuration Files
-------------------

Use configuration files for complex parameter sets:

.. code-block:: bash

   python -m scxpand.main train \
     --model_type svm \
     --data_path data/example_data.h5ad \
     --config_path config/svm_config.json

Training Monitoring
-------------------

Monitor training progress with TensorBoard:

.. code-block:: bash

   # Start TensorBoard (view all training runs)
   tensorboard --logdir=results/

   # Or view a specific model type
   tensorboard --logdir=results/pan_cancer_autoencoder_v_0/

   # Access dashboard at http://localhost:6006

Model Inference
---------------

Run inference on new data using trained models via CLI or programmatic API.

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Use the predict command for direct model inference:

.. code-block:: bash

   # Basic prediction with automatic model type detection
   python -m scxpand.main inference \
     --model_path results/autoencoder \
     --data_path new_data.h5ad

   # Pre-trained model from registry
   python -m scxpand.main inference \
     --model_name pan_cancer_autoencoder \
     --data_path new_data.h5ad

   # External model via direct URL (seamless model sharing!)
   python -m scxpand.main inference \
     --model_url "https://your-platform.com/model.zip" \
     --data_path new_data.h5ad

   # Prediction with custom save location and batch size
   python -m scxpand.main inference \
     --model_path results/lightgbm \
     --data_path test_data.h5ad \
     --save_path custom_predictions/ \
     --batch_size 2048

**Command Parameters:**
   - ``--model_path``: Directory containing local trained model files
   - ``--model_name``: Name of pre-trained model from registry (alternative to model_path/model_url)
   - ``--model_url``: Direct URL to model ZIP file (alternative to model_path/model_name)
   - ``--data_path``: Path to input data file (h5ad format)
   - ``--save_path``: Directory to save results (optional, auto-generated if not specified)
   - ``--batch_size``: Batch size for inference (default: 1024)

Programmatic API
~~~~~~~~~~~~~~~~

scXpand provides a unified inference API that supports local models, registry models, and direct URL models:

Unified Inference API (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the high-level ``scxpand.run_inference`` function for all model types:

.. code-block:: python

   import scxpand
   import anndata as ad

   # Local model inference (file-based)
   results = scxpand.run_inference(
       model_path='results/autoencoder',
       data_path='test_data.h5ad',
       save_path='predictions/',
       batch_size=1024,
       device='cuda'  # Optional: specify device (cpu, cuda, mps)
   )

   # Registry model inference (pre-trained models)
   results = scxpand.run_inference(
       model_name='pan_cancer_autoencoder',
       data_path='test_data.h5ad',
       save_path='predictions/',
       batch_size=1024,
       device=None  # Auto-detect best available device (cuda/mps/cpu)
   )

   # Direct URL model inference (any external model)
   results = scxpand.run_inference(
       model_url='https://your-platform.com/model.zip',
       data_path='test_data.h5ad',
       save_path='predictions/',
       batch_size=1024,
       device='mps'  # Use Apple Silicon GPU
   )

   # In-memory inference (any model type)
   adata = ad.read_h5ad('test_data.h5ad')
   results = scxpand.run_inference(
       model_path='results/autoencoder',
       adata=adata,  # In-memory AnnData object
       save_path='predictions/',
       batch_size=1024,
       device='cpu'  # Force CPU usage
   )

Lower-level API (Advanced Users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more granular control or custom workflows, use the individual functions.
The unified ``scxpand.run_inference`` is recommended for most use cases:

.. code-block:: python

   import scxpand

   # Run inference using the unified API (model type auto-detected)
   results = scxpand.run_inference(
       model_path='results/autoencoder',  # Local model
       data_path='test_data.h5ad',
       batch_size=1024,
       device=None  # Auto-detect device
   )
   y_pred_prob = results.predictions

.. note::
   **Unified Inference API Features:**

   - ``scxpand.run_inference`` provides a single function for all model types (local, registry, URL)
   - **Automatic model type detection**: ``model_type`` parameter is automatically detected
   - **Multiple model sources**: Local models (``model_path``), registry models (``model_name``), and direct URLs (``model_url``)
   - **Flexible data handling**: Both ``data_path`` (file-based) and ``adata`` (in-memory) are supported
   - **Device management**: ``device`` parameter supports 'cpu', 'cuda', 'mps', or None for auto-detection
   - **Automatic caching**: Pre-trained models are cached automatically using `Pooch <https://github.com/fatiando/pooch>`_
   - **Seamless model sharing**: Use any ZIP file URL for instant model sharing
   - All model types (autoencoder, mlp, lightgbm, logistic, svm) use the same unified API
   - Results include evaluation metrics (when ground truth is available) and saved prediction files

Tutorials
---------

scXpand provides Jupyter notebooks for hands-on tutorials and data analysis:

* **Getting Started with scXpand** (:doc:`notebooks/scxpand_tutorial`) - Data preprocessing and model application.
* **Inference** (:doc:`notebooks/inference`) - Load trained models and run predictions on new datasets for model deployment. Demonstrates both file-based and in-memory inference modes.
* **Embeddings Analysis** (:doc:`notebooks/embeddings`) - Visualize and analyze learned model representations to understand what patterns contribute to expansion prediction.
