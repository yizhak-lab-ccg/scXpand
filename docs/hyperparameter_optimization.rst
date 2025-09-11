Hyperparameter Optimization Guide
=================================

Overview
--------

scXpand provides automated hyperparameter optimization using `Optuna <https://optuna.org/>`_, an optimization framework.

**Key Features:**
   - **Bayesian Optimization**: Parameter space exploration using Tree-structured Parzen Estimator (TPE)
   - **Early Pruning**: Automatic termination of unpromising trials
   - **Resume Capability**: Continue optimization from previous studies
   - **Model-Specific Grids**: Tailored parameter spaces for each architecture

Reproducibility and Data Splitting
-----------------------------------

The hyperparameter optimization system ensures reproducible results through a fixed random seed approach:

**Fixed Random Seed System:**
- All trials use the same base random seed (default: 42)
- The seed is set globally using ``set_seed()``

**Consistent Train/Validation Splits:**
- The same random seed ensures identical train/validation splits across all trials
- Data splitting is performed at the patient level
- Split indices are saved and reused when resuming optimization
- This allows fair comparison between different hyperparameter combinations


Quick Start
-----------

Basic Optimization
~~~~~~~~~~~~~~~~~~

Optimize a single model with default settings:

.. code-block:: bash

   # Optimize MLP model
   python -m scxpand.main optimize \
       --model_type mlp \
       --data_path data/example_data.h5ad \
       --n_trials 100 \
       --study_name "mlp_optimization"

   # Optimize autoencoder with custom configuration
   python -m scxpand.main optimize \
       --model_type autoencoder \
       --data_path data/example_data.h5ad \
       --n_trials 200 \
       --study_name "autoencoder_deep_search" \
       --num_workers 4

Multi-Model Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Compare all available architectures:

.. code-block:: bash

   # Optimize all model types in parallel
   python -m scxpand.main optimize-all \
       --data_path data/example_data.h5ad \
       --n_trials 50 \
       --num_workers 4

This command will create separate optimization studies for each model type (autoencoder, mlp, lightgbm, logistic, svm) and run them sequentially.

Results Analysis
----------------

Study Outputs and Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each optimization study creates outputs for analysis:

.. code-block:: text

   results/optuna_studies/study_name/
   ├── optuna.db                    # SQLite database with all trials
   ├── info.json                    # Study summary and best trial info
   ├── trials.log                   # Detailed trial execution logs
   ├── trial_0/                     # Individual trial results
   │   ├── parameters.json          # Trial hyperparameters
   │   ├── results_dev.txt          # Evaluation metrics (text format)
   │   ├── results_table_dev.csv    # Per-cell predictions and metadata
   │   ├── summary_info.json        # Trial summary and run metadata
   │   ├── data_format.json         # Gene names and preprocessing statistics
   │   ├── data_format.npz          # Numerical normalization statistics
   │   ├── data_splits.npz          # Train/dev/test splits
   │   ├── model.joblib             # Trained model (sklearn: LightGBM, Logistic, SVM)
   │   ├── best_ckpt.pt            # Best checkpoint (neural networks: MLP, Autoencoder)
   │   ├── last_ckpt.pt            # Latest checkpoint (neural networks: MLP, Autoencoder)
   │   └── plots/
   │       └── roc_curve_dev.png    # ROC curve visualization
   └── trial_N/...                  # Additional trials

**Key Files:**

- **info.json**: Contains study metadata, trial counts, best trial parameters, and complete results
- **parameters.json**: Complete hyperparameter configuration for each trial
- **results_dev.txt**: Hierarchical evaluation metrics (AUROC, accuracy, etc.) by category
- **results_table_dev.csv**: Per-cell predictions with metadata for detailed analysis
- **summary_info.json**: Trial summary, best epoch information, and run metadata
- **data_format.json**: Gene names and data preprocessing statistics
- **data_format.npz**: Numerical data normalization statistics for consistent preprocessing
- **data_splits.npz**: Train/dev/test splits for reproducible evaluation
- **optuna.db**: SQLite database with complete trial history for programmatic access
- **model.joblib**: Trained model file (sklearn models: LightGBM, Logistic, SVM)
- **best_ckpt.pt**: Best model checkpoint (neural networks: MLP, Autoencoder)
- **last_ckpt.pt**: Latest checkpoint (neural networks: MLP, Autoencoder)

Inspecting Optimization Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

scXpand provides multiple ways to inspect and analyze your hyperparameter optimization results:

**Summary Files**
   Each study automatically generates summary files for quick inspection:

   - **Study Summary** (``info.json``): Contains best trial information, parameter values, and overall study statistics
   - **Trial Logs** (``trials.log``): Detailed execution logs showing progress of each trial
   - **Best Trial Results** (``trial_N/results_dev.txt``): Complete evaluation metrics for the best performing trial

   These files are located in your study directory: ``results/optuna_studies/{study_name}/``

**Optuna Dashboard**
   For interactive visualization and analysis, use the Optuna Dashboard:

.. code-block:: bash

   # Install Optuna Dashboard
   pip install optuna-dashboard

   # Launch dashboard for your study
   optuna-dashboard sqlite:///results/optuna_studies/mlp_optimization/optuna.db

   # Access the dashboard at http://localhost:8080

The Optuna Dashboard provides rich visualizations including parameter importance plots, optimization history, and interactive parameter relationships. For more details, see the `Optuna Dashboard documentation <https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html>`_.

Loading and Analyzing Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access study results programmatically:

.. code-block:: python

   import json
   import optuna
   import pandas as pd
   from pathlib import Path

   # Load study from database
   study = optuna.load_study(
       study_name="mlp_optimization",
       storage="sqlite:///results/optuna_studies/mlp_optimization/optuna.db"
   )

   # Load study summary
   with open("results/optuna_studies/mlp_optimization/info.json") as f:
       study_info = json.load(f)

   print(f"Best trial: {study_info['best_trial_number']}")
   print(f"Best value: {study_info['best_value']:.4f}")
   print(f"Completed trials: {study_info['completed_trials']}")

   # Load best trial detailed results
   best_trial_dir = f"results/optuna_studies/mlp_optimization/trial_{study_info['best_trial_number']}"

   # Load per-cell predictions for analysis
   predictions_df = pd.read_csv(f"{best_trial_dir}/results_table_dev.csv")

   # Load complete evaluation metrics
   with open(f"{best_trial_dir}/results_dev.txt") as f:
       detailed_metrics = f.read()


Training with Best Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the optimized parameters to train a new model:

**Fresh Training (Default)**

.. code-block:: bash

   # Train a new model from scratch using the best parameters
   python -m scxpand.main train \
       --model_type mlp \
       --data_path data/example_data.h5ad \
       --config_path results/optuna_studies/mlp_optimization/trial_42/parameters.json \
       --save_dir results/final_model/ \
       --resume false

**Resume Training**

.. code-block:: bash

   # Resume training from existing checkpoint with optimal parameters
   python -m scxpand.main train \
       --model_type mlp \
       --data_path data/example_data.h5ad \
       --config_path results/optuna_studies/mlp_optimization/trial_42/parameters.json \
       --save_dir results/final_model/ \
       --resume true


Inference with Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the already trained model from optimization for predictions:

.. code-block:: bash

   # Use the trained model directly for inference on new data
   python -m scxpand.main predict \
       --model_path results/optuna_studies/mlp_optimization/trial_42 \
       --data_path new_data.h5ad \
       --save_path predictions/




Custom Configuration
--------------------

Custom Parameter Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override specific parameters while optimizing others:

.. code-block:: bash

   # Fix batch size while optimizing other parameters
   python -m scxpand.main optimize \
       --model_type mlp \
       --data_path data.h5ad \
       --n_trials 100 \
       --batch_size 4096 \
       --use_log_transform true

You can override any parameter that appears in the model's parameter class.

Configuration Files
~~~~~~~~~~~~~~~~~~~

Use JSON configuration files for complex parameter sets:

.. code-block:: json

   {
       "use_log_transform": true,
       "use_zscore_norm": true,
       "n_epochs": 50,
       "early_stopping_patience": 10,
       "learning_rate": 1e-4,
       "target_sum": 10000
   }

.. code-block:: bash

   # Use configuration file
   python -m scxpand.main optimize \
       --model_type autoencoder \
       --data_path data.h5ad \
       --config_path config/autoencoder_config.json \
       --n_trials 200

Study Resumption
~~~~~~~~~~~~~~~~

scXpand provides resume functionality controlled by the `--resume` flag.

**Resuming a Study (`--resume True`, Default):**

By default, scXpand will automatically detect and continue a study if it finds an existing one with the same name.

.. code-block:: bash

   # This will resume the study "existing_mlp_study" if it exists,
   # or create it if it doesn't.
   python -m scxpand.main optimize \
       --model_type mlp \
       --data_path data/example_data.h5ad \
       --study_name "existing_mlp_study" \
       --n_trials 50  # Run 50 additional trials

**Starting a Fresh Study (`--resume False`):**

To ensure you don't accidentally overwrite results, you must explicitly set `--resume False` to start a new study if one with the same name already exists. If existing trials are found, the program will stop and provide instructions.

.. code-block:: bash

   # This will fail if "mlp_fresh_study" already has trials.
   python -m scxpand.main optimize \
       --model_type mlp \
       --data_path data/example_data.h5ad \
       --study_name "mlp_fresh_study" \
       --resume False \
       --n_trials 100


Optimization System Architecture
--------------------------------

Study Management
~~~~~~~~~~~~~~~~

Each optimization run creates a persistent study stored in SQLite:

.. code-block:: text

   results/optuna_studies/
   ├── mlp_optimization/
   │   ├── optuna.db              # Trial database
   │   ├── info.json              # Study summary
   │   ├── trials.log             # Detailed trial logs
   │   └── trial_0/, trial_1/...  # Individual trial results
   └── autoencoder_study/
       └── ...

**Study Components:**
   - **Database**: Persistent storage of all trials and results
   - **Metadata**: Study configuration and best trial information
   - **Trial Artifacts**: Complete model outputs for each trial
   - **Logs**: Detailed execution logs for debugging

Optimization Algorithm
~~~~~~~~~~~~~~~~~~~~~~

scXpand uses Optuna's TPE (Tree-structured Parzen Estimator) sampler with automated pruning:

.. code-block:: python

   from scxpand.hyperopt.hyperopt_optimizer import HyperparameterOptimizer

   # Create optimizer with custom configuration
   optimizer = HyperparameterOptimizer(
       model_type="autoencoder",
       data_path="data.h5ad",
       study_name="custom_ae_opt",
       score_metric="harmonic_avg/AUROC",  # Optimization target
       seed_base=42,                       # Reproducibility
       num_workers=4,                      # Parallel trials
       resume=True,                        # Resume existing study
       fail_fast=False                     # Continue on errors
   )




Parameter Spaces Reference
--------------------------

This section provides detailed hyperparameter ranges and distributions for each model type. These ranges are automatically used during optimization and can serve as reference for manual configuration.

MLP Parameters
~~~~~~~~~~~~~~

Multi-layer perceptron optimization focuses on architecture and regularization. Neural network models use a fixed training duration of 30 epochs:

.. list-table:: MLP Hyperparameter Ranges
   :header-rows: 1
   :widths: 25 20 35 20

   * - Parameter
     - Type
     - Range/Options
     - Distribution
   * - **num_layers**
     - Integer
     - 2 to 5
     - Uniform
   * - **n_epochs**
     - Integer
     - 30
     - Fixed
   * - **layer_units**
     - Variable
     - [512, 1024, 2048, 4096]
     - Uniform per layer
   * - **init_learning_rate**
     - Float
     - 1e-5 to 1e-3
     - Log-uniform
   * - **train_batch_size**
     - Categorical
     - 2048, 4096
     - Uniform
   * - **dropout_rate**
     - Float
     - 0.1 to 0.3
     - Uniform
   * - **weight_decay**
     - Float
     - 1e-5 to 1e-2
     - Log-uniform
   * - **mask_rate**
     - Float
     - 0.05 to 0.3
     - Uniform
   * - **noise_std**
     - Float
     - 1e-5 to 1e-3
     - Log-uniform
   * - **positives_weight**
     - Float
     - 0.1 to 10.0
     - Log-uniform
   * - **use_soft_loss**
     - Boolean
     - True, False
     - Uniform
   * - **sampler_type**
     - Categorical
     - ``"random"``, ``"balanced_labels"``, ``"balanced_types"``
     - Uniform
   * - **lr_scheduler_type**
     - Categorical
     - ``"NoScheduler"``, ``"ReduceLROnPlateau"``, ``"OneCycleLR"``, ``"StepLR"``, ``"CosineAnnealingLR"``, ``"ConstantLR"``
     - Uniform
   * - **adam_beta1**
     - Float
     - 0.8 to 0.95
     - Uniform
   * - **adam_beta2**
     - Float
     - 0.95 to 0.999
     - Uniform
   * - **cat_loss_weight**
     - Float
     - 0.1 to 10.0
     - Log-uniform
   * - **aux_categorical_types**
     - Categorical
     - ``"none"``, ``"tissue_type"``, ``"imputed_labels"``, ``"tissue_type,imputed_labels"``
     - Uniform
   * - **soft_loss_beta**
     - Float
     - 0.5 to 10.0
     - Uniform (when use_soft_loss=True)
   * - **soft_loss_start_epoch**
     - Integer
     - 0 to n_epochs-1
     - Uniform (when use_soft_loss=True)


Autoencoder Parameters
~~~~~~~~~~~~~~~~~~~~~~

The autoencoder optimization explores a large parameter space with 30 epochs of training:

.. list-table:: Autoencoder Hyperparameter Ranges
   :header-rows: 1
   :widths: 25 20 35 20

   * - Parameter
     - Type
     - Range/Options
     - Distribution
   * - **model_type**
     - Categorical
     - ``"standard"``, ``"fork"``
     - Uniform
   * - **loss_type**
     - Categorical
     - ``"mse"``, ``"nb"``, ``"zinb"``
     - Uniform
   * - **n_epochs**
     - Integer
     - 30
     - Fixed
   * - **use_soft_loss**
     - Boolean
     - True, False
     - Uniform
   * - **latent_dim**
     - Categorical
     - 16, 32, 64, 128
     - Uniform
   * - **init_learning_rate**
     - Float
     - 1e-6 to 1e-3
     - Log-uniform
   * - **train_batch_size**
     - Categorical
     - 2048, 4096
     - Uniform
   * - **dropout_rate**
     - Float
     - 0.1 to 0.5
     - Uniform
   * - **weight_decay**
     - Float
     - 1e-6 to 1e-2
     - Log-uniform
   * - **recon_loss_weight**
     - Float
     - 0.1 to 10.0
     - Log-uniform
   * - **cls_loss_weight**
     - Float
     - 0.1 to 10.0
     - Log-uniform
   * - **cat_loss_weight**
     - Float
     - 0.1 to 10.0
     - Log-uniform
   * - **sampler_type**
     - Categorical
     - ``"random"``, ``"balanced_labels"``, ``"balanced_types"``
     - Uniform
   * - **lr_scheduler_type**
     - Categorical
     - ``"NoScheduler"``, ``"ReduceLROnPlateau"``, ``"OneCycleLR"``, ``"StepLR"``, ``"CosineAnnealingLR"``, ``"ConstantLR"``
     - Uniform
   * - **adam_beta1**
     - Float
     - 0.8 to 0.95
     - Uniform
   * - **adam_beta2**
     - Float
     - 0.95 to 0.999
     - Uniform
   * - **encoder_hidden_dims**
     - Variable
     - [32, 64, 128, 256, 512, 1024]
     - Uniform per layer (1-3 layers)
   * - **decoder_hidden_dims**
     - Variable
     - [32, 64, 128, 256, 512, 1024]
     - Uniform per layer (1-3 layers)
   * - **classifier_hidden_dims**
     - Variable
     - [16, 32, 64]
     - Uniform per layer (1-2 layers)
   * - **l1_lambda**
     - Float
     - 1e-5 to 1e-2
     - Log-uniform
   * - **ridge_lambda**
     - Float
     - 1e-4 to 1e-1
     - Log-uniform (ZINB only)
   * - **aux_categorical_types**
     - Categorical
     - ``"none"``, ``"tissue_type"``, ``"imputed_labels"``, ``"tissue_type,imputed_labels"``
     - Uniform
   * - **soft_loss_beta**
     - Float
     - 0.5 to 10.0
     - Uniform (when use_soft_loss=True)
   * - **soft_loss_start_epoch**
     - Integer
     - 0 to n_epochs-1
     - Uniform (when use_soft_loss=True)


LightGBM Parameters
~~~~~~~~~~~~~~~~~~~

Gradient boosting optimization explores tree structure and learning parameters:

.. list-table:: LightGBM Hyperparameter Ranges
   :header-rows: 1
   :widths: 25 20 35 20

   * - Parameter
     - Type
     - Range/Options
     - Distribution
   * - **n_estimators**
     - Integer
     - 50 to 300
     - Uniform
   * - **max_depth**
     - Integer
     - 3 to 12
     - Uniform
   * - **num_leaves**
     - Integer
     - 15 to 127
     - Uniform
   * - **learning_rate**
     - Float
     - 1e-3 to 0.1
     - Log-uniform
   * - **min_child_samples**
     - Integer
     - 5 to 100
     - Uniform
   * - **feature_fraction**
     - Float
     - 0.7 to 0.95
     - Uniform
   * - **bagging_fraction**
     - Float
     - 0.7 to 0.95
     - Uniform
   * - **reg_alpha**
     - Float
     - 1e-8 to 10.0
     - Log-uniform
   * - **reg_lambda**
     - Float
     - 1e-8 to 10.0
     - Log-uniform
   * - **min_split_gain**
     - Float
     - 0.0 to 1.0
     - Uniform
   * - **class_weight**
     - Categorical
     - ``"balanced"``, ``None``
     - Uniform
   * - **use_zscore_norm**
     - Boolean
     - True, False
     - Uniform
   * - **min_child_weight**
     - Float
     - 1e-3 to 10.0
     - Log-uniform
   * - **boosting_type**
     - Categorical
     - ``"gbdt"``, ``"dart"``, ``"goss"``
     - Uniform
   * - **objective**
     - Categorical
     - ``"binary"``
     - Fixed
   * - **metric**
     - Categorical
     - ``"binary_logloss"``, ``"auc"``
     - Uniform


Linear Model Parameters
~~~~~~~~~~~~~~~~~~~~~~~

scXpand supports two linear model types with different loss functions and optimization characteristics:

**Logistic Regression**
   Uses logistic loss (cross-entropy)
**Support Vector Machine (SVM)**
   Uses hinge loss

**Shared Parameters:**
Both models use SGD (Stochastic Gradient Descent) optimization and share most hyperparameters:

.. list-table:: Linear Model Hyperparameter Ranges
   :header-rows: 1
   :widths: 25 20 35 20

   * - Parameter
     - Type
     - Range/Options
     - Distribution
   * - **use_log_transform**
     - Boolean
     - True, False
     - Uniform
   * - **penalty**
     - Categorical
     - ``"l2"``, ``"elasticnet"``
     - Uniform
   * - **alpha**
     - Float
     - 1e-6 to 1e-1
     - Log-uniform
   * - **n_epochs**
     - Integer
     - 30
     - Fixed
   * - **tol**
     - Float
     - 1e-6 to 1e-3
     - Log-uniform
   * - **class_weight**
     - Categorical
     - ``"balanced"``, ``None``
     - Uniform
   * - **batch_size**
     - Categorical
     - 512, 1024, 2048
     - Uniform
   * - **sampler_type**
     - Categorical
     - ``"random"``, ``"balanced_labels"``, ``"balanced_types"``
     - Uniform
   * - **init_learning_rate**
     - Float
     - 1e-5 to 1e-2
     - Log-uniform
   * - **learning_rate**
     - Categorical
     - ``"optimal"``, ``"constant"``, ``"invscaling"``, ``"adaptive"``
     - Uniform
   * - **eta0**
     - Float
     - 1e-3 to 1.0
     - Log-uniform
   * - **power_t**
     - Float
     - 0.25 to 0.75
     - Uniform
   * - **lr_scheduler_type**
     - Categorical
     - ``"NoScheduler"``, ``"ReduceLROnPlateau"``, ``"OneCycleLR"``, ``"StepLR"``, ``"CosineAnnealingLR"``, ``"ConstantLR"``
     - Uniform
   * - **l1_ratio**
     - Float
     - 0.1 to 0.9
     - Uniform (elasticnet only)
   * - **mask_rate**
     - Float
     - 0.0 to 0.3
     - Uniform
   * - **noise_std**
     - Float
     - 1e-5 to 1e-3
     - Log-uniform
   * - **soft_loss_beta**
     - Float
     - 0.5 to 10.0
     - Uniform
   * - **warm_start**
     - Boolean
     - True, False
     - Uniform
   * - **average**
     - Boolean
     - True, False
     - Uniform
   * - **n_iter_no_change**
     - Integer
     - 3 to 10
     - Uniform
   * - **validation_fraction**
     - Float
     - 0.05 to 0.2
     - Uniform
   * - **fit_intercept**
     - Boolean
     - True, False
     - Uniform
   * - **shuffle**
     - Boolean
     - True, False
     - Uniform
