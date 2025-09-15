Data Pipeline & Normalization
=============================

.. note::
   This document provides an overview of the scXpand data processing pipeline, from raw single-cell RNA sequencing data to model-ready normalized expression matrices.

Overview
--------

The scXpand data pipeline transforms raw single-cell gene expression data through a series of preprocessing steps to prepare it for machine learning models. The pipeline ensures consistent data processing between training and inference while maintaining computational efficiency for large datasets.

Pipeline Architecture
=====================

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

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~

The preprocessing pipeline applies three sequential transformations to raw gene expression data:

**Notation**: Throughout this section, :math:`X_{raw}[i,j]` represents the raw read count for cell *i* and gene *j* from the single-cell RNA sequencing experiment.

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

Data Input Modes
----------------

scXpand supports two data input modes for different use cases:

File-Based Mode (Memory Efficient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: Large datasets that don't fit in memory (>10GB)

**How it works**: Data is loaded in batches directly from disk using AnnData's backed mode. Only the required cells and genes are loaded into memory at any given time.

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

Log Transformation
~~~~~~~~~~~~~~~~~~

Log transformation helps with:
* Reducing the impact of highly expressed genes
* Stabilizing variance across the expression range
* Making the data more suitable for downstream analysis

Z-Score Normalization
~~~~~~~~~~~~~~~~~~~~~

Z-score normalization standardizes each gene's expression across cells using training set statistics. This step:

* Centers each gene's expression around zero
* Scales each gene to unit variance
* Uses robust clipping to handle outliers (±3σ by default)
* Adds small epsilon for numerical stability

**Gene Statistics Computation**:
   The per-gene means (μ) and standard deviations (σ) are computed once from the training set using the same preprocessing steps (row normalization and optional log transformation) but **without** masking or noise augmentation. These statistics are then saved in DataFormat and used for all future processing.



Gene Format Handling
---------------------

scXpand automatically handles cases where inference data has different gene ordering or subsets compared to training data.

**Gene Reordering Process**:
   1. Compare gene names between datasets
   2. Create mapping from inference to training gene order
   3. Reorder expression matrix columns
   4. Handle missing genes by zero-padding

**Gene Subsetting**: For inference on specific gene subsets, the system automatically filters to only include genes present in the training data.


Data Augmentation
-----------------

Data augmentation is used **only during training** for neural network models (MLP and Autoencoder) and linear models (Logistic regression and SVM) to improve generalization and robustness.

**Training Pipeline Sequence:**
1. **Load raw expression data** from AnnData file
2. **Apply pre-normalization augmentations** (gene masking)
3. **Apply core preprocessing pipeline**:
   a. Row normalization (target_sum = 10,000)
   b. Log transformation (if enabled)
   c. Z-score normalization (if enabled) using pre-computed training statistics
4. **Apply post-normalization augmentations** (Gaussian noise addition)

**Augmentation Types:**

1. **Gene Masking** (Pre-normalization):
   - Randomly sets genes to zero before any normalization steps
   - Simulates technical dropouts in single-cell data

2. **Gaussian Noise** (Post-normalization):
   - Adds small amounts of Gaussian noise to fully normalized expression data
   - Uses a small standard deviation (typically 1e-4) appropriate for normalized data scale
   - Helps prevent overfitting and improves generalization

3. **Soft Labels**:
   - Uses continuous labels in [0,1] instead of binary {0,1} labels
   - Computed from clone size ratios using sigmoid scaling
   - Formula: ``sigmoid(soft_loss_beta * (clone_size_ratio - 1.5))``
   - Helps with label noise and improves model calibration

**Important Notes:**
  - During **inference**, no augmentations are applied - only the core preprocessing pipeline runs
  - Gene statistics (μ, σ) for z-score normalization are **precomputed once** from clean training data (without masking or noise) and reused for all inference
  - Genes from training that are missing in inference data are filled with zeros and normalized using their training statistics
  - Genes in inference data that were not in training are discarded (only training genes are processed)

Inference Data Format Handling
===============================

The scXpand inference pipeline is designed to handle test data with different formats, gene sets, and structures than the training data while maintaining consistency with the training preprocessing pipeline.

**Gene Format Standardization Process:**

1. **Gene Mapping and Reordering**: All inference data goes through automatic gene format standardization
   - Genes are reordered to match ``data_format.gene_names``
   - Missing genes are added as zero columns at correct positions
   - Extra genes are removed
   - Final gene count matches training format exactly

2. **Preprocessing Pipeline**: The same preprocessing pipeline as training is applied
   - Row normalization: Each cell sums to ``target_sum`` (typically 10,000)
   - Log transformation: ``log1p()`` for variance stabilization
   - Z-score normalization: Per-gene normalization using precomputed ``genes_mu[i]`` and ``genes_sigma[i]``

**Example: Complex Gene Mismatch Handling**

**Training Data Format:**
::

   training_genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
   genes_mu = [100.0, 10.0, 50.0, 5.0]
   genes_sigma = [20.0, 100.0, 30.0, 200.0]

**Test Data (Complex Mismatch):**
::

   test_genes = ["GENE_C", "GENE_A", "EXTRA_1", "GENE_E", "EXTRA_2"]
   # Missing: GENE_B, GENE_D
   # Extra: EXTRA_1, EXTRA_2, GENE_E
   # Reordered: GENE_C, GENE_A

**Transformation Process:**

1. **Gene mapping**: GENE_A → position 0, GENE_C → position 2
2. **Missing genes**: GENE_B (position 1), GENE_D (position 3) filled with zeros
3. **Extra genes**: EXTRA_1, EXTRA_2, GENE_E ignored
4. **Result**: ``[100.0, 0.0, 50.0, 0.0]`` (missing genes filled with zeros)
5. **Preprocessing**: Row norm → log → z-score using training statistics

API Reference
=============

Key functions and classes for data pipeline operations:

* :py:class:`~scxpand.data_util.data_format.DataFormat` - Central configuration hub for preprocessing parameters
* :py:func:`~scxpand.data_util.data_format.load_data_format` - Load DataFormat from file
* :py:func:`~scxpand.data_util.transforms.preprocess_expression_data` - Apply complete preprocessing pipeline
* :py:func:`~scxpand.data_util.transforms.apply_row_normalization` - Row normalization step
* :py:func:`~scxpand.data_util.transforms.apply_log_transform` - Log transformation step
* :py:func:`~scxpand.data_util.transforms.apply_zscore_normalization` - Z-score normalization step
* :py:meth:`~scxpand.data_util.data_format.DataFormat.reorder_genes_to_match_format` - Gene reordering for inference
* :py:func:`~scxpand.data_util.transforms.load_and_preprocess_data_numpy` - Load and preprocess data for inference
* :py:func:`~scxpand.run_inference` - Complete inference pipeline with data loading options
