Data Splitting Strategy
=======================

.. note::
   This document explains how scXpand splits datasets into training, validation, and test sets while maintaining biological integrity and preventing data leakage.

Overview
--------

scXpand uses two distinct splitting strategies depending on the evaluation level:

1. **Train-Validation Split**: Patient-level splitting with stratification by cancer type
2. **Test Data Split**: Study-level separation where test data comes from entirely different studies without stratification considerations

This dual approach ensures both proper model selection during training and realistic evaluation of generalization across different study contexts.

.. raw:: html

   <div align="center">
     <br/>
     <h4>Multi-Level Data Splitting Strategy</h4>
     <br/>
   </div>

Key Principles
--------------

Train-Validation Split (Patient-Level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Patient-Level Splitting**:
   - Cells from the same patient share genetic background, treatment history, and disease progression
   - Random cell-level splitting would create data leakage where training and validation sets contain cells from the same patients
   - Patient-level splitting provides more realistic performance estimates for model selection

**Implementation**:
   The splitting algorithm operates on unique patient identifiers (combining study and patient information) rather than individual cells, ensuring complete separation of patients between training and validation sets.

Test Data Split (Across Studies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Study-Level Separation**:
   - Test data comes from entirely different studies than those used for training and validation
   - No stratification by cancer type or other variables is applied
   - Provides true assessment of model generalization across different experimental contexts

**Purpose**:
   - Evaluate model performance on completely unseen study populations
   - Test study-specific batch effects and methodological differences
   - Simulate real-world deployment scenarios where models encounter new study contexts

Stratified Splitting (Train-Validation Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Preserved Distributions** (applies only to train-validation split):
   - **Cancer Type Distribution**: Maintains similar proportions of cancer types in training and validation sets
   - **Tissue Type Distribution**: Balances tissue types across splits when possible
   - **Expansion Label Distribution**: Preserves the balance of expanded vs. non-expanded cells

**Benefits**:
   - Prevents bias toward specific cancer types in either set
   - Ensures validation set is representative of the overall patient population across studies
   - Maintains statistical power for rare cancer types during model selection

**Note**: Test data split does not use stratification, allowing evaluation of model performance on naturally occurring distributions in new studies.

Implementation Details
----------------------

Train-Validation Split Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patient Identifier Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

scXpand creates unique patient identifiers by combining study and patient information:

.. code-block:: python

   from scxpand.data_util.data_splitter import get_patient_identifiers

   # Generate composite patient IDs
   patient_identifiers = get_patient_identifiers(adata.obs)
   # Format: "study_name:patient_id"
   # Example: ["study1:P001", "study1:P002", "study2:P003"]

**Required Metadata Columns**:
   - ``study``: Study or dataset identifier
   - ``patient``: Patient identifier within each study
   - ``cancer_type``: Cancer type for stratification

Core Splitting Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`~scxpand.data_util.data_splitter.split_data` function implements the patient-aware splitting:

.. code-block:: python

   from scxpand.data_util.data_splitter import split_data

   # Split data by patients
   train_indices, dev_indices = split_data(
       adata=adata,
       dev_ratio=0.2,              # 20% for validation
       random_seed=42,             # Reproducible splits
       save_path=results_dir       # Save patient ID lists
   )

**Algorithm Steps**:

1. **Patient Enumeration**: Extract unique patient identifiers
2. **Cancer Type Mapping**: Map each patient to their cancer type
3. **Stratified Split**: Use scikit-learn's stratified splitting on patients
4. **Cell Index Generation**: Map patient splits back to cell-level indices
5. **Quality Validation**: Verify distribution preservation

Stratification Process
^^^^^^^^^^^^^^^^^^^^^^

The splitting uses scikit-learn's ``train_test_split`` with stratification:

.. code-block:: python

   from sklearn.model_selection import train_test_split

   # Stratify by cancer type at patient level
   train_patients, dev_patients = train_test_split(
       unique_patient_ids,
       test_size=dev_ratio,
       stratify=cancer_types_per_patient,  # One cancer type per patient
       random_state=random_seed
   )

**Stratification Variables** (train-validation split only):
   - **Primary**: Cancer type (ensures balanced representation)
   - **Secondary**: Tissue type and expansion status (monitored and reported)

Test Data Split Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For test evaluation, data comes from studies that are completely separate from those used in training and validation:

.. code-block:: python

   # Test data workflow (conceptual)
   # Training studies: ["study_A", "study_B", "study_C"]
   # Test studies: ["study_D", "study_E"]

   # No stratification applied - use natural distribution
   test_data = load_test_studies(["study_D", "study_E"])

   # Evaluate trained model on test data
   test_results = evaluate_model(model, test_data)

**Key Differences from Train-Validation Split**:
   - No patient-level splitting needed (entire studies are separate)
   - No stratification by cancer type or other variables
   - Evaluation reflects natural distribution in new study contexts
   - Tests true generalization across different experimental settings

Reproducibility
---------------

Deterministic Splitting
~~~~~~~~~~~~~~~~~~~~~~~~

The splitting process is fully deterministic when using a fixed random seed:

.. code-block:: python

   # Reproducible splits across runs
   train_indices, dev_indices = split_data(
       adata=adata,
       dev_ratio=0.2,
       random_seed=42  # Fixed seed ensures identical splits
   )

**Saved Artifacts**:
   - ``train_patient_ids.csv``: List of training patient identifiers
   - ``dev_patient_ids.csv``: List of validation patient identifiers
   - ``data_splits.npz``: Numpy arrays of cell indices for fast loading

Resumable Workflows
~~~~~~~~~~~~~~~~~~~~

Patient ID lists are saved to enable consistent splits across different runs:

.. code-block:: python

   # Load existing splits
   train_patients = pd.read_csv("results/train_patient_ids.csv").values.flatten()
   dev_patients = pd.read_csv("results/dev_patient_ids.csv").values.flatten()

   # Reconstruct cell indices
   patient_identifiers = get_patient_identifiers(adata.obs)
   train_indices = np.where(patient_identifiers.isin(train_patients))[0]
   dev_indices = np.where(patient_identifiers.isin(dev_patients))[0]

Integration with Training Pipeline
----------------------------------

Train-Validation Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data splitting for training and validation is automatically integrated into the training preparation:

.. code-block:: python

   from scxpand.data_util.prepare_data_for_train import prepare_data_for_training

   # Prepare data with automatic splitting
   bundle = prepare_data_for_training(
       data_path="data.h5ad",
       dev_ratio=0.2,           # Validation split ratio
       rand_seed=42,            # Reproducible splits
       save_dir="results/"      # Output directory
   )

   # Access split results
   train_indices = bundle.row_inds_train
   dev_indices = bundle.row_inds_dev
   data_format = bundle.data_format

Dataset Creation
~~~~~~~~~~~~~~~~

Split indices are used to create training and validation datasets:

.. code-block:: python

   from scxpand.data_util.dataset import CellsDataset

   # Create training dataset
   train_dataset = CellsDataset(
       data_format=data_format,
       row_inds=train_indices,    # Only training cells
       is_train=True,
       data_path="data.h5ad"
   )

   # Create validation dataset
   dev_dataset = CellsDataset(
       data_format=data_format,
       row_inds=dev_indices,      # Only validation cells
       is_train=False,
       data_path="data.h5ad"
   )
