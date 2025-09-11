Evaluation & Metrics Guide
==========================

.. note::
   This guide explains scXpand's evaluation system, including stratified metrics, performance analysis, and interpretation guidelines for T-cell expansion prediction.

Overview
--------

scXpand provides an evaluation framework that computes performance metrics stratified by cell type and tissue type. The system uses aggregation methods to combine results across strata and provides detailed performance breakdowns.

.. raw:: html

   <div align="center">
     <br/>
     <h4>Multi-Level Performance Evaluation</h4>
     <br/>
   </div>

**Key Features:**
   - **Stratified Analysis**: Performance broken down by cell type and tissue type
   - **Harmonic Mean Aggregation**: Balanced performance assessment across strata
   - **Clinical Relevance**: Metrics aligned with real-world deployment needs
   - **Visualization Tools**: ROC curves and performance plots for each stratum

Core Metrics
------------

Primary Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

scXpand computes standard binary classification metrics for T-cell expansion prediction:

**Area Under ROC Curve (AUROC)**
   The primary metric for model selection and comparison. Measures the model's ability to distinguish between expanded and non-expanded T-cells across all classification thresholds.

   .. math::
      \text{AUROC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)

   Where TPR is True Positive Rate and FPR is False Positive Rate.

**F1 Score**
   Harmonic mean of precision and recall, providing balanced assessment of classification performance:

   .. math::
      \text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

**Error Rate**
   Simple classification error using 0.5 threshold:

   .. math::
      \text{Error Rate} = \frac{\text{False Predictions}}{\text{Total Predictions}}

.. code-block:: python

   from scxpand.util.metrics import calculate_metrics

   # Compute metrics
   metrics = calculate_metrics(
       y_true=true_labels,           # Binary labels (0/1)
       y_pred_prob=predictions,      # Predicted probabilities
       obs_df=metadata_df,          # Cell metadata
       threshold=0.5                # Classification threshold
   )

   print(f"Overall AUROC: {metrics['overall']['AUROC']:.3f}")
   print(f"Overall F1: {metrics['overall']['F1']:.3f}")

Detailed Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**False Positive Rate (FPR)**
   Proportion of non-expanded cells incorrectly classified as expanded:

   .. math::
      \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}

**False Negative Rate (FNR)**
   Proportion of expanded cells incorrectly classified as non-expanded:

   .. math::
      \text{FNR} = \frac{\text{False Negatives}}{\text{False Negatives} + \text{True Positives}}

**Root Mean Squared Error (RMSE)**
   Measures prediction calibration quality:

   .. math::
      \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{p}_i)^2}

**Positive Rate**
   Fraction of cells predicted as expanded (dataset balance indicator):

   .. math::
      \text{Positive Rate} = \frac{\text{Predicted Positives}}{\text{Total Cells}}

Stratified Evaluation
---------------------

Biological Stratification
~~~~~~~~~~~~~~~~~~~~~~~~~

scXpand performs stratified evaluation across biologically meaningful groups to ensure performance across different contexts:

**Stratification Dimensions:**
   - **Tissue Type**: Tumor vs. Blood performance
   - **Cell Type**: Performance across different T-cell subtypes (CD4+, CD8+, regulatory T-cells, etc.)
   - **Combined Strata**: Intersection of tissue type and cell type

.. code-block:: python

   # Stratified metrics are automatically computed
   metrics = calculate_metrics(y_true, y_pred_prob, obs_df)

   # Access per-stratum results
   for stratum_name, stratum_metrics in metrics.items():
       if stratum_name not in ['overall', 'average', 'harmonic_avg']:
           tissue, cell_type = stratum_name.split('__')
           print(f"{cell_type} in {tissue}: AUROC = {stratum_metrics['AUROC']:.3f}")



The stratified approach ensures models that perform well in clinical deployment scenarios rather than just on aggregate statistics.

Aggregation Methods
---------------------------

Harmonic Mean Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~

scXpand uses harmonic mean aggregation for performance assessment across strata:

.. math::
   \text{Harmonic Mean} = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}

**Why Harmonic Mean?**
   - **Sensitive to Poor Performance**: Penalizes models that fail on specific strata
   - **Clinically Relevant**: Ensures consistent performance across all biological contexts
   - **Balanced**: Less affected by strata with many samples

.. code-block:: python

   # Harmonic mean is automatically computed
   harmonic_auroc = metrics['harmonic_avg']['AUROC']
   arithmetic_auroc = metrics['average']['AUROC']

   print(f"Harmonic Mean AUROC: {harmonic_auroc:.3f}")
   print(f"Arithmetic Mean AUROC: {arithmetic_auroc:.3f}")



Evaluation Workflow
-------------------

Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

The evaluation system provides end-to-end performance assessment:

.. code-block:: python

   from scxpand.util.metrics import evaluate_and_save

   # Complete evaluation with visualization
   results = evaluate_and_save(
       y_true=validation_labels,
       y_pred_prob=model_predictions,
       obs_df=validation_metadata,
       eval_name="dev",                    # Evaluation set name
       save_path=results_directory,        # Output directory
       plots_dir=plots_directory,          # Visualization output
       threshold=0.5,                      # Classification threshold
       trial=optuna_trial                  # Optional: for optimization
   )

**Generated Outputs:**
   - **Text Report**: Detailed metrics in human-readable format
   - **CSV Table**: Per-cell predictions and metadata
   - **ROC Curves**: Overall and per-stratum visualizations
   - **Summary JSON**: Machine-readable results

Visualization Components
~~~~~~~~~~~~~~~~~~~~~~~~

**Overall ROC Curve**
   Standard ROC analysis for aggregate performance:

.. code-block:: python

   from scxpand.util.plots import plot_roc_curve

   # Generate overall ROC curve
   overall_auroc = plot_roc_curve(
       labels=y_true,
       probs_pred=y_pred_prob,
       show_plot=True,
       plot_save_dir=output_directory,
       plot_name="overall_roc",
       title="Overall ROC Curve"
   )

**Per-Stratum ROC Curves**
   Individual ROC curves for each biological stratum:

.. code-block:: python

   # Automatic per-stratum ROC generation
   strata_cols = ["tissue_type", "imputed_labels"]
   strata_df = obs_df[strata_cols]
   strata = strata_df.astype(str).agg(" - ".join, axis=1)

   for stratum_name in strata.unique():
       mask = strata == stratum_name
       if mask.sum() > 10:  # Minimum sample size
           stratum_auroc = plot_roc_curve(
               labels=y_true[mask],
               probs_pred=y_pred_prob[mask],
               plot_name=f"roc_{stratum_name}",
               title=f"ROC: {stratum_name}"
           )
