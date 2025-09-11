Input Data Format Specification
=================================

This document describes the expected input data format for scXpand training and inference.

The framework expects single-cell RNA sequencing data in ``.h5ad`` format with specific annotation fields in the ``obs`` (observations) metadata and gene expression data in the ``X`` matrix.

Gene Expression Matrix (X)
---------------------------

The ``X`` matrix contains the gene expression data for all cells:

* **Format**: Dense or sparse matrix (CSR format recommended for memory efficiency)
* **Shape**: ``(n_cells, n_genes)`` where rows are cells and columns are genes
* **Data Type**: Integer counts (stored as ``int32``, ``int64``, or ``float32``/``float64`` with integer values)
* **Values**: Raw gene expression counts (non-negative integers)
* **Preprocessing**: The framework expects raw counts and applies its own preprocessing pipeline

  * Raw counts are processed according to the data preprocessing configuration
  * Pre-normalized or log-transformed data is not supported

Gene Annotations (var)
----------------------

Gene information is stored in the ``var`` (variables) metadata:

* **Index**: Gene identifiers (e.g., gene symbols, Ensembl IDs)
* **Requirement**: Gene identifiers should be consistent across datasets if combining multiple studies

.. note::

   Our entire pipeline is designed to work only with ensembl_ids for consistency across datasets. If your data uses gene symbols, please convert them to ensembl_ids before using our pre-trained models.

Cell Annotations (obs)
----------------------

Required for Training
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 35 40

   * - Field
     - Type
     - Description
     - Usage
   * - ``study``
     - str
     - Study identifier for data splitting
     - **Training only** - Used for patient-level train/validation splits
   * - ``patient``
     - str
     - Unique patient identifier within study
     - **Training only** - Used for patient-level train/validation splits
   * - ``cancer_type``
     - str
     - Cancer type annotation
     - **Training only** - Used for stratified splitting
   * - ``sample``
     - str
     - Sample identifier within patient
     - **Training only** - Used for data organization
   * - ``expansion``
     - str
     - Expansion label ("expanded" or "non-expanded")
     - **Training only** - Target variable for model training
   * - ``clone_id_size``
     - int
     - Number of cells with this clone_id in the current sample
     - **Training only** - Used for soft label computation
   * - ``median_clone_size``
     - int
     - Median clone size in the sample
     - **Training only** - Used for soft label computation
   * - ``tissue_type``
     - str
     - Tissue type annotation
     - **Training only** - Used for evaluation stratification
   * - ``imputed_labels``
     - str
     - Cell type labels
     - **Training only** - Used for evaluation stratification

.. note::
   ``tissue_type`` and ``imputed_labels`` are only needed if you want to compute stratified evaluation metrics, but are not required for the actual inference/prediction process.

Expansion Definition
~~~~~~~~~~~~~~~~~~~

The ``expansion`` field should contain string values:

* ``"expanded"`` - for cells considered part of expanded clones
* ``"non-expanded"`` - for cells not considered part of expanded clones

The framework uses a 1.5× median clone size threshold: a cell is considered expanded if its ``clone_id_size`` > 1.5 × ``median_clone_size`` for that sample.

Required for Inference
~~~~~~~~~~~~~~~~~~~
Applying our pre-trained models only for inference purposes requires:

* Filtration of the gene expression matrix to include only T cells
* Gene representation using ensembl_ids

Our platform will be able to handle missing genes, different gene orders, and additional genes not used by the model.
