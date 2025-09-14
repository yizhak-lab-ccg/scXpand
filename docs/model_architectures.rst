Model Architectures Guide
=========================

.. note::
   This guide explains all model architectures available in scXpand, their strengths, use cases, and configuration options.

Overview
--------

scXpand provides five distinct model architectures, each designed for different use cases and data characteristics. The framework allows you to experiment with multiple approaches for T-cell expansion prediction tasks.

Available Model Architectures
==============================

**Neural Network Models:**
   - **Autoencoder**: Deep count autoencoder with reconstruction and classification heads
   - **MLP**: Multi-layer perceptron for direct expansion prediction

**Gradient Boosting:**
   - **LightGBM**: Gradient boosted decision trees optimized for tabular data

**Linear Models:**
   - **Logistic Regression**: Linear classifier with logistic loss
   - **SVM**: Support vector machine with hinge loss

Autoencoder-Based Models
------------------------

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

scXpand's autoencoder architecture is inspired by the Deep Count Autoencoder (DCA) approach introduced by `Eraslan et al. (2019) <https://www.nature.com/articles/s41467-018-07931-2>`_ for single-cell RNA-seq data denoising. Our implementation extends this concept by combining reconstruction learning with expansion classification.

.. code-block:: python

   from scxpand.autoencoders.ae_models import AutoencoderModel
   from scxpand.autoencoders.ae_params import AutoEncoderParams

   # Create autoencoder model
   params = AutoEncoderParams(
       model_type="standard",           # or "fork"
       loss_type="zinb",               # "mse", "nb", or "zinb"
       latent_dim=32,
       encoder_hidden_dims=(128, 64),
       decoder_hidden_dims=(64, 128)
   )

Scientific Foundation
~~~~~~~~~~~~~~~~~~~~~

The autoencoder approach addresses several key challenges in single-cell data analysis:

1. **Count Data Distribution**: Single-cell RNA-seq data follows count distributions (Negative Binomial, Zero-inflated Negative Binomial) rather than Gaussian distributions assumed by traditional methods.

2. **Zero-Inflation**: The high sparsity in single-cell data requires specialized handling of true biological zeros vs. technical dropouts.

3. **Overdispersion**: Gene expression exhibits variance greater than the mean.

As described in the `DCA paper <https://www.nature.com/articles/s41467-018-07931-2>`_, the autoencoder learns to map noisy observations back to an underlying "clean" data manifold, effectively denoising the expression while preserving biological signal.

Architecture Variants
~~~~~~~~~~~~~~~~~~~~~

scXpand provides two distinct autoencoder architectures that differ fundamentally in how they handle the decoder pathway for reconstruction tasks.

**Standard Autoencoder**
   Uses a **shared decoder pathway** with multiple output heads:

   .. code-block:: text

      Input (genes) → Encoder → Latent → Shared Decoder → Mean Head (μ)
                                  ↓                    → Pi Head (π)
                              Classifier              → Theta Head (θ)
                                  ↓
                          Expansion Prediction

**Fork Autoencoder**
   Uses **separate decoder pathways** for each reconstruction parameter:

   .. code-block:: text

      Input (genes) → Encoder → Latent → Mean Decoder → Mean Head (μ)
                                  ↓   → Pi Decoder → Pi Head (π)
                              Classifier → Theta Decoder → Theta Head (θ)
                                  ↓
                          Expansion Prediction


Loss Functions
~~~~~~~~~~~~~~

scXpand supports three loss functions for the reconstruction component:

**Mean Squared Error (MSE)**
   Traditional L2 loss:

   .. math::
      \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{G} (x_{ij} - \mu_{ij})^2

**Negative Binomial (NB)**
   Accounts for overdispersion in count data:

   .. math::
      \mathcal{L}_{NB} = -\sum_{i=1}^{N} \sum_{j=1}^{G} \log \text{NB}(x_{ij}; \mu_{ij}, \theta_{ij})

**Zero-Inflated Negative Binomial (ZINB)**
   Handles both overdispersion and zero-inflation:

   .. math::
      \mathcal{L}_{ZINB} = -\sum_{i=1}^{N} \sum_{j=1}^{G} \log \text{ZINB}(x_{ij}; \mu_{ij}, \theta_{ij}, \pi_{ij})

   Where:
   - :math:`\mu_{ij}`: Mean expression for gene j in cell i
   - :math:`\theta_{ij}`: Dispersion parameter
   - :math:`\pi_{ij}`: Zero-inflation probability


Multi-Layer Perceptron (MLP)
----------------------------

Architecture Design
~~~~~~~~~~~~~~~~~~~

The MLP model provides a direct approach to expansion prediction without reconstruction learning. It uses fully connected layers with dropout regularization and optional auxiliary classification heads.

.. code-block:: python

   from scxpand.mlp.mlp_params import MLPParam
   from scxpand.mlp.mlp_model import MLPModel

   # Configure MLP
   mlp_params = MLPParam(
       layer_units=[512, 256, 128, 64],    # Hidden layer sizes
       dropout_rate=0.3,
       learning_rate=1e-3,
       n_epochs=30
   )

**Architecture Flow:**

.. code-block:: text

   Input (genes) → FC Layer 1 → Dropout → ReLU
                 → FC Layer 2 → Dropout → ReLU
                 → ...
                 → Output Layer → Sigmoid → Expansion Probability

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mlp_config = {
       # Architecture
       "layer_units": [1024, 512, 256, 128],  # Layer sizes
       "dropout_rate": 0.25,                  # Regularization

       # Training
       "learning_rate": 5e-4,
       "weight_decay": 1e-4,
       "n_epochs": 25,
       "batch_size": 2048,

       # Data augmentation
       "mask_rate": 0.1,                      # Gene masking
       "noise_std": 1e-4,                     # Gaussian noise

       # Loss function
       "positives_weight": 2.0,               # Class imbalance handling
       "use_soft_loss": True                  # Soft vs hard labels
   }

LightGBM Models
---------------

Gradient Boosting Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightGBM provides a non-neural approach using gradient boosted decision trees. This method excels on tabular data and often serves as a strong baseline for genomics applications.

.. code-block:: python

   from scxpand.lightgbm.lightgbm_params import LightGBMParams

   # Configure LightGBM
   lgbm_params = LightGBMParams(
       n_estimators=200,
       learning_rate=0.1,
       max_depth=8,
       num_leaves=64,
       class_weight="balanced"
   )


Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   lightgbm_config = {
       # Tree structure
       "n_estimators": 150,               # Number of trees
       "max_depth": 10,                   # Maximum tree depth
       "num_leaves": 31,                  # Maximum leaves per tree

       # Learning
       "learning_rate": 0.05,             # Shrinkage rate
       "feature_fraction": 0.8,           # Feature sampling
       "bagging_fraction": 0.8,           # Row sampling

       # Regularization
       "reg_alpha": 0.1,                  # L1 regularization
       "reg_lambda": 0.1,                 # L2 regularization
       "min_child_samples": 20,           # Minimum samples per leaf

       # Class imbalance
       "class_weight": "balanced",        # Auto weight adjustment
       "is_unbalance": True
   }


Linear Models
-------------

Logistic Regression
~~~~~~~~~~~~~~~~~~~

Classic linear model using logistic loss function for binary classification. Provides interpretable coefficients and fast training.

.. code-block:: python

   from scxpand.linear.linear_params import LinearClassifierParam

   # Configure logistic regression
   logistic_params = LinearClassifierParam(
       model_name="LogisticRegression",
       C=1.0,                             # Inverse regularization strength
       penalty="l2",                      # L1, L2, or elastic net
       max_iter=1000,
       class_weight="balanced"
   )

**Mathematical Model:**

.. math::
   P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \sum_{j=1}^{p} \beta_j x_j)}}


Support Vector Machine (SVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear SVM using hinge loss, optimized for maximum margin classification.

.. code-block:: python

   # Configure SVM
   svm_params = LinearClassifierParam(
       model_name="LinearSVC",
       C=1.0,                             # Regularization parameter
       loss="hinge",                      # Loss function
       penalty="l2",                      # Regularization type
       dual=False,                        # Primal vs dual formulation
       class_weight="balanced"
   )

**Mathematical Objective:**

.. math::
   \min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))


Multi-task Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both autoencoder and MLP models support auxiliary classification tasks for predicting cell types or tissue types alongside expansion:

.. code-block:: python

   # Enable auxiliary classification
   params = AutoEncoderParams(
       aux_categorical_types=("tissue_type", "imputed_labels"),
       cat_loss_weight=0.5                # Weight for auxiliary losses
   )
