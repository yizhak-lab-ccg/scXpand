Data Pipeline & Normalization
=============================

.. note::
   This document provides an overview of the scXpand data processing pipeline, from raw single-cell RNA sequencing data to model-ready normalized expression matrices.

Overview
--------

The scXpand data pipeline transforms raw single-cell gene expression data through a series of preprocessing steps to prepare it for machine learning models. The pipeline ensures consistent data processing between training and inference while maintaining computational efficiency for large datasets.

Data Pipeline Architecture
============================

The data pipeline consists of three main components:

1. **Data Format Management**: Central configuration for preprocessing parameters
2. **Preprocessing Pipeline**: Sequential normalization and transformation steps
3. **Data Loading System**: Efficient batch processing for training and inference

Core Components
---------------

DataFormat Class
~~~~~~~~~~~~~~~~

The :py:class:`~scxpand.data_util.data_format.DataFormat` class serves as the central configuration hub for all data preprocessing operations. It ensures consistency between training and inference by storing:

* **Gene Information**: Gene names and ordering
* **Normalization Statistics**: Per-gene means and standard deviations for z-score normalization
* **Preprocessing Parameters**: Log transformation and normalization settings
* **Metadata**: Categorical feature mappings

.. code-block:: python

   from scxpand.data_util.data_format import DataFormat, load_data_format

   # During training: create data format
   data_format = DataFormat(
       use_log_transform=True,
       use_zscore_norm=True,
       target_sum=10000
   )
   data_format.create_data_format(
       data_path="data.h5ad",
       adata=adata,
       row_inds_train=train_indices
   )

   # During inference: load saved data format
   data_format = load_data_format("results/data_format.json")

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~

The preprocessing pipeline applies three sequential transformations to raw gene expression data:

**Notation**: Throughout this section, :math:`X_{raw}[i,j]` represents the raw read count for cell *i* and gene *j* from the single-cell RNA sequencing experiment, before any normalization or transformation.

**Step 1: Row Normalization**
   Normalizes each cell's total gene expression to a target sum (default: 10,000). This accounts for differences in sequencing depth between cells.

   .. math::
      X_{norm}[i,j] = X_{raw}[i,j] \times \frac{\text{target_sum}}{\sum_k X_{raw}[i,k]}

**Step 2: Log Transformation (Optional)**
   Applies log1p transformation to reduce the impact of highly expressed genes and stabilize variance.

   .. math::
      X_{log}[i,j] = \log(1 + X_{norm}[i,j])

**Step 3: Z-Score Normalization (Optional)**
   Standardizes gene expression using per-gene statistics computed from the training set.

   .. math::
      X_{zscore}[i,j] = \frac{X_{log}[i,j] - \mu_j}{\sigma_j + \epsilon}

   Where :math:`\mu_j` and :math:`\sigma_j` are the mean and standard deviation of gene *j* computed from training data.

.. code-block:: python

   from scxpand.data_util.transforms import preprocess_expression_data

   # Apply complete preprocessing pipeline
   X_processed = preprocess_expression_data(
       X=raw_expression_matrix,
       data_format=data_format
   )

Data Input Modes
----------------

scXpand supports two data input modes for different use cases:

File-Based Mode (Memory Efficient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: Large datasets that don't fit in memory (>10GB)

**How it works**: Data is loaded in batches directly from disk using AnnData's backed mode. Only the required cells and genes are loaded into memory at any given time.

.. code-block:: python

   # File-based inference example
   # Use unified API (model type auto-detected)
   results = scxpand.run_inference(
       model_path='results/autoencoder',  # Local model
       data_path='large_dataset.h5ad',  # File path
       adata=None,                      # No in-memory data
       device=None                      # Auto-detect device
   )

**Advantages**:
   * Memory efficient for very large datasets
   * Scales to datasets with millions of cells
   * Automatic memory management

**Considerations**:
   * Slower than in-memory mode due to disk I/O
   * Requires data to be stored in HDF5 format

In-Memory Mode (High Performance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: Smaller datasets that fit comfortably in RAM (<5GB)

**How it works**: The entire dataset is loaded into memory once, enabling faster batch access during training or inference.

.. code-block:: python

   # Load data into memory
   adata = ad.read_h5ad("dataset.h5ad")

   # In-memory inference example
   # Use unified API (model type auto-detected)
   results = scxpand.run_inference(
       model_path='results/autoencoder',  # Local model
       data_path=None,           # No file path
       adata=adata,             # In-memory data
       device=None              # Auto-detect device
   )

**Advantages**:
   * Faster data access during training/inference
   * No disk I/O bottlenecks
   * Better for iterative model development

**Considerations**:
   * Memory usage scales with dataset size
   * May cause out-of-memory errors with large datasets

Normalization Details
---------------------

Row Normalization
~~~~~~~~~~~~~~~~~~

Row normalization addresses the technical variability in sequencing depth between cells. Without normalization, cells with higher total read counts would appear to have higher expression across all genes.

.. code-block:: python

   from scxpand.data_util.transforms import apply_row_normalization

   # Normalize to 10,000 total counts per cell
   X_normalized = apply_row_normalization(X_raw, target_sum=10000)

Log Transformation
~~~~~~~~~~~~~~~~~~

Log transformation helps with:

.. code-block:: python

   from scxpand.data_util.transforms import apply_log_transform

   # Apply log1p transformation
   X_log = apply_log_transform(X_normalized, in_place=True)


Z-Score Normalization
~~~~~~~~~~~~~~~~~~~~~

Z-score normalization standardizes each gene's expression across cells using training set statistics. This step:


.. code-block:: python

   from scxpand.data_util.transforms import apply_zscore_normalization

   # Apply z-score normalization using training statistics
   X_zscore = apply_zscore_normalization(
       X_log,
       genes_mu=data_format.genes_mu,
       genes_sigma=data_format.genes_sigma,
       eps=1e-4
   )

**Statistical Considerations**:
   * Uses training set statistics for both training and inference
   * Applies clipping to limit outlier effects (±3σ by default)
   * Adds small epsilon for numerical stability


Gene Reordering and Subset Handling
------------------------------------

scXpand automatically handles cases where inference data has different gene ordering or subsets compared to training data.

Gene Reordering
~~~~~~~~~~~~~~~

When gene order differs between training and inference:

.. code-block:: python

   # Automatically reorder genes to match training format
   adata_reordered = data_format.reorder_genes_to_match_format(adata)

**Process**:
   1. Compare gene names between datasets
   2. Create mapping from inference to training gene order
   3. Reorder expression matrix columns
   4. Handle missing genes by zero-padding

Gene Subsetting
~~~~~~~~~~~~~~~~

For inference on specific gene subsets:

.. code-block:: python

   from scxpand.data_util.transforms import load_and_preprocess_data_numpy

   # Load and preprocess only specific genes
   X_subset = load_and_preprocess_data_numpy(
       data_path="data.h5ad",
       data_format=data_format,
       gene_subset=["CD3D", "CD8A", "CD4"]  # Specific genes
   )


Performance Optimization
------------------------

The data pipeline includes several optimizations for large-scale processing:

Batch Processing
~~~~~~~~~~~~~~~~

* **Streaming from Disk**: Processes data in configurable batches to control memory usage
* **Parallel Loading**: Supports multi-worker data loading for training

.. code-block:: python

   # Configure batch processing
   dataset = CellsDataset(
       data_format=data_format,
       data_path="large_dataset.h5ad",
       batch_size=1024
   )

Memory Management
~~~~~~~~~~~~~~~~~

* **Sparse Matrix Support**: Preserves sparsity through row normalization and log transformation
* **Backed Mode**: Uses AnnData's backed mode for memory-efficient file access



Best Practices
--------------


Inference Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Consistent Preprocessing**: Always use the same DataFormat from training
2. **Gene Compatibility**: Ensure inference data gene sets match training data as closely as possible
3. **Memory Mode Selection**: Choose based on dataset size and available RAM
4. **Batch Size Tuning**: Optimize for your hardware configuration

.. code-block:: python

   # Inference best practices
   data_format = load_data_format("results/data_format.json")  # From training

   # Verify gene compatibility
   missing_genes = set(data_format.gene_names) - set(adata.var_names)
   if missing_genes:
       print(f"Warning: {len(missing_genes)} genes missing from inference data")

Troubleshooting
~~~~~~~~~~~~~~~

**Common Issues**:

* **Memory Errors**: Switch to file-based mode or reduce batch size
* **Gene Mismatches**: Use gene reordering or subset functionality
* **Slow Processing**: Increase batch size or use more workers
* **Numerical Issues**: Check for extreme values or invalid statistics

**Debug Tools**:

.. code-block:: python

   # Debug preprocessing pipeline
   print(f"Data format: {data_format}")
   print(f"Gene statistics: μ={data_format.genes_mu.mean():.3f}, σ={data_format.genes_sigma.mean():.3f}")
   print(f"Matrix shape: {X.shape}, dtype: {X.dtype}")

Integration with Model Training
-------------------------------

The data pipeline integrates with scXpand's model training system:

Dataset Creation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from scxpand.data_util.dataset import CellsDataset
   from scxpand.data_util.dataloaders import create_train_dataloader

   # Create training dataset with preprocessing
   train_dataset = CellsDataset(
       data_format=data_format,
       row_inds=train_indices,
       is_train=True,
       data_path="data.h5ad"
   )

   # Create data loader with batching
   train_loader = create_train_dataloader(
       train_dataset=train_dataset,
       batch_size=512,
       num_workers=4
   )

Training Loop Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Training loop with automatic preprocessing
   for batch in train_loader:
       X = batch['x']  # Already preprocessed
       y = batch['y']  # Target labels

       # Model forward pass
       predictions = model(X)
       loss = criterion(predictions, y)

The preprocessing happens transparently in the background, ensuring that model training always receives properly normalized data.

Data Augmentation
~~~~~~~~~~~~~~~~~

Data augmentation is used **only during training** for the neural network models (MLP and Autoencoder) and the linear models (Logistic regression and SVM) to improve generalization and robustness.

**Augmentation Pipeline:**
Data augmentation is applied in two stages during the preprocessing pipeline:

1. **Pre-normalization augmentations** (applied to raw counts):
   - Gene masking to simulate technical dropouts

2. **Post-normalization augmentations** (applied after preprocessing):
   - Gaussian noise addition to normalized data

.. code-block:: python

   # Example configuration for neural network models
   from scxpand.mlp.mlp_params import MLPParam
   from scxpand.autoencoders.ae_params import AutoEncoderParams

   # MLP with augmentation
   mlp_params = MLPParam(
       mask_rate=0.1,                     # Mask 10% of genes (pre-normalization)
       noise_std=1e-4,                    # Add Gaussian noise (post-normalization)
       soft_loss_beta=1.0,                # Soft label scaling factor
       soft_loss_start_epoch=None         # Use soft labels from epoch 0
   )

   # Autoencoder with augmentation
   ae_params = AutoEncoderParams(
       mask_rate=0.1,                     # Mask 10% of genes
       noise_std=1e-4,                    # Add Gaussian noise
       soft_loss_beta=1.0,                # Soft label scaling factor
   )

**Augmentation Types:**

1. **Gene Masking** (Pre-normalization):
   - Randomly sets genes to zero before normalization
   - Simulates technical dropouts in single-cell data

2. **Gaussian Noise** (Post-normalization):
   - Adds small amounts of Gaussian noise to normalized expression data
   - Helps prevent overfitting and improves generalization

3. **Soft Labels**:
   - Uses continuous labels in [0,1] instead of binary {0,1} labels
   - Computed from clone size ratios using sigmoid scaling
   - Formula: ``sigmoid(soft_loss_beta * (clone_size_ratio - 1.5))``
   - Can be scheduled to start after specific epochs
   - Helps with label noise and improves model calibration
