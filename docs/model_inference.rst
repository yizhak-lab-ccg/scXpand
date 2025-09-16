Model Inference
===============

Run inference on new data using trained models via CLI or programmatic API.

Programmatic API
----------------

scXpand provides a unified inference API that supports local models, registry models, and direct URL models:

Unified Inference API
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
       batch_size=1024
   )

   # Registry model inference (pre-trained models)
   results = scxpand.run_inference(
       model_name='pan_cancer_autoencoder',
       data_path='test_data.h5ad',
       save_path='predictions/',
       batch_size=1024
   )

   # Direct URL model inference (any external model)
   results = scxpand.run_inference(
       model_url='https://your-platform.com/model.zip',
       data_path='test_data.h5ad',
       save_path='predictions/',
       batch_size=1024
   )

   # In-memory inference (any model type)
   adata = ad.read_h5ad('test_data.h5ad')
   results = scxpand.run_inference(
       model_path='results/autoencoder',
       adata=adata,  # In-memory AnnData object
       save_path='predictions/',
       batch_size=1024
   )

.. note::
   **Unified Inference API Features:**

   - ``scxpand.run_inference`` provides a single function for all model types (local, registry, URL)
   - **Automatic model type detection**: ``model_type`` parameter is automatically detected
   - **Multiple model sources**: Local models (``model_path``), registry models (``model_name``), and direct URLs (``model_url``)
   - **Flexible data handling**: Both ``data_path`` (file-based) and ``adata`` (in-memory) are supported
   - **Automatic device detection**: Device is automatically detected for optimal performance
   - **Automatic caching**: Pre-trained models are cached automatically using `Pooch <https://github.com/fatiando/pooch>`_
   - **Seamless model sharing**: Use any ZIP file URL for instant model sharing
   - All model types (autoencoder, mlp, lightgbm, logistic, svm) use the same unified API
   - Results include evaluation metrics (when ground truth is available) and saved prediction files

Handling Data Format Mismatches
--------------------------------

scXpand automatically handles common data format mismatches between training and inference data:

**Gene Set Flexibility:**
   - **Missing genes**: Automatically filled with zeros
   - **Extra genes**: Ignored during inference, only training genes are used
   - **Reordered genes**: Automatically reordered to match training format
   - **Mixed scenarios**: Handles combination of missing, extra, and reordered genes

**Data Format Consistency:**
   - Test data is automatically transformed to match training data format
   - Identical preprocessing pipeline as training (row norm → log → z-score)
   - Works consistently across all model types

For detailed technical information about the data transformation pipeline, see :doc:`data_pipeline`.

Inference API Reference
-----------------------

Run predictions on new data using trained models:

.. automodule:: scxpand.core.inference
   :members: run_inference
   :show-inheritance:

Command-line interface for inference:

.. autofunction:: scxpand.main.inference

For detailed CLI examples and usage, see the `scxpand.main` module documentation.

Pre-trained Models
------------------

Download and use pre-trained models:

.. automodule:: scxpand.pretrained.download_manager
   :members: download_pretrained_model, download_model
   :show-inheritance:

.. automodule:: scxpand.util.model_registry
   :members: list_pretrained_models, get_pretrained_model_info
   :show-inheritance:
