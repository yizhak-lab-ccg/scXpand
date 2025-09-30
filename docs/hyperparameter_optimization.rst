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

Optimization of Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   python -m scxpand.main inference \
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




API Reference
=============

Hyperparameter Optimization Functions
--------------------------------------

Single Model Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scxpand.main.optimize

Multi-Model Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scxpand.main.optimize_all

Optimization Framework
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: scxpand.hyperopt.hyperopt_optimizer
   :members: HyperparameterOptimizer
   :show-inheritance:

Hyperparameter Search Ranges
-----------------------------

All hyperparameter definitions and ranges are maintained in ``scxpand/hyperopt/param_grids.py``.
