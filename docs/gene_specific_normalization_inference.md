# Inference and Test Data Format Handling

## Overview

This document explains how the scXpand inference pipeline handles test data with different formats, gene sets, and structures than the training data. The system is designed to be robust and automatically adapt to various data format mismatches while maintaining consistency with the training pipeline.

## Key Principles

1. **Data Format Consistency**: Test data is automatically transformed to match the training data format
2. **Gene Set Flexibility**: Handles missing, extra, and reordered genes gracefully
3. **Preprocessing Consistency**: Applies identical preprocessing steps as training
4. **Model Agnostic**: Works consistently across all model types (MLP, Autoencoder, Logistic, SVM, LightGBM)

## Data Format Mismatch Scenarios

### 1. **Gene Count Mismatches**
- **Fewer genes**: Missing genes are filled with zeros and normalized appropriately
- **More genes**: Extra genes are ignored, only training genes are used
- **Different gene sets**: Partial overlap is handled by mapping and filling missing genes

### 2. **Gene Order Differences**
- **Reordered genes**: Automatically reordered to match training format
- **Mixed scenarios**: Combination of reordering, missing, and extra genes

### 3. **Data Structure Variations**
- **In-memory vs file-based**: Both AnnData objects and file paths are supported
- **Different cell counts**: Flexible handling of varying sample sizes
- **Metadata differences**: Optional columns are handled gracefully

## How the System Works

### **Step 1: Data Loading and Validation**
```python
# Supports both in-memory and file-based data
if adata is not None:
    # Use in-memory AnnData object
    source_data = adata
else:
    # Load from file (supports backed mode for memory efficiency)
    source_data = ad.read_h5ad(data_path, backed="r")
```

### **Step 2: Gene Format Standardization**
```python
# All models use data_format.prepare_adata_for_training() for consistency
adata_standardized = data_format.prepare_adata_for_training(
    source_data,
    reorder_genes=True
)
```

**What happens during standardization:**
- Genes are reordered to match `data_format.gene_names`
- Missing genes are added as zero columns at correct positions
- Extra genes are removed
- Gene count matches training format exactly

### **Step 3: Preprocessing Pipeline**
```python
# Same preprocessing as training: row norm → log → z-score
X_processed = preprocess_expression_data(
    X=adata_standardized.X,
    data_format=data_format
)
```

**Preprocessing steps:**
1. **Row normalization**: Each cell sums to `target_sum` (typically 10,000)
2. **Log transformation**: `log1p()` for variance stabilization
3. **Z-score normalization**: Per-gene normalization using `genes_mu[i]` and `genes_sigma[i]`

## Model-Specific Implementation

### **Neural Network Models (MLP, Autoencoder)**
```python
# Use CellsDataset for efficient gene transformation and preprocessing
dataset = CellsDataset(
    data_format=data_format,
    row_inds=eval_row_inds,
    is_train=False,  # Inference mode
    adata=adata,     # or data_path
)

# Gene transformation happens automatically during batch loading
dataloader = create_eval_dataloader(dataset, batch_size, num_workers)
```

**Advantages:**
- Memory efficient: gene transformation happens once per sample
- Batch processing: supports large datasets
- Automatic gene mapping: handles any gene mismatch scenario

### **Linear Models (Logistic, SVM)**
```python
# Same CellsDataset approach as neural networks
# Ensures identical preprocessing across all model types
```

### **LightGBM Models**
```python
# Direct preprocessing approach
def _prepare_data_for_lightgbm_inference(data_format, adata, data_path, eval_row_inds):
    # 1. Standardize gene format
    adata_standardized = data_format.prepare_adata_for_training(adata, reorder_genes=True)

    # 2. Apply preprocessing pipeline
    return preprocess_expression_data(adata_standardized.X, data_format)
```

## Example: Complex Gene Mismatch

### **Training Data Format**
```python
training_genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
genes_mu = [100.0, 10.0, 50.0, 5.0]
genes_sigma = [20.0, 100.0, 30.0, 200.0]
```

### **Test Data (Complex Mismatch)**
```python
test_genes = ["GENE_C", "GENE_A", "EXTRA_1", "GENE_E", "EXTRA_2"]
# Missing: GENE_B, GENE_D
# Extra: EXTRA_1, EXTRA_2, GENE_E
# Reordered: GENE_C, GENE_A
```

### **Transformation Process**
```python
# 1. Gene mapping and reordering
# GENE_A -> position 0, GENE_C -> position 2
# GENE_B -> missing (position 1), GENE_D -> missing (position 3)
# EXTRA_1, EXTRA_2, GENE_E -> ignored

# 2. Result after transformation
X_transformed = [
    [100.0, 0.0, 50.0, 0.0]  # Missing genes filled with zeros
]

# 3. After preprocessing pipeline
# Row norm: Each row sums to 10,000
# Log: log1p(normalized_values)
# Z-score: (log_value - genes_mu[i]) / genes_sigma[i]
```

## Robustness Features

### **1. Memory Efficiency**
- **Backed mode support**: Large files loaded efficiently without loading entire dataset into memory
- **Batch processing**: Neural networks process data in configurable batches
- **Gene transformation caching**: Transformation happens once per sample, not per batch

### **2. Error Handling**
- **Validation**: Checks for required data before processing
- **Graceful degradation**: Handles missing metadata columns
- **Clear error messages**: Identifies specific issues (e.g., "missing genes", "extra genes")

### **3. Flexibility**
- **Mixed data sources**: Can combine in-memory and file-based approaches
- **Partial evaluation**: Supports evaluating only specific rows (`eval_row_inds`)
- **Configurable parameters**: Batch size, workers, device selection

## Testing and Validation

### **Comprehensive Test Coverage**
The system includes extensive tests covering:

1. **Gene mismatch scenarios**: Missing, extra, reordered, partial overlap
2. **Data loading modes**: In-memory, file-based, backed mode
3. **Model types**: All 5 supported model architectures
4. **Edge cases**: Single gene overlap, large gene sets, memory efficiency
5. **Preprocessing consistency**: Identical pipeline to training

## Best Practices

### **1. Data Preparation**
- Ensure test data has at least some overlapping genes with training data
- Use consistent gene naming conventions
- Provide metadata columns when possible for evaluation

### **2. Performance Optimization**
- Use file-based loading for large datasets
- Adjust batch size based on available memory
- Use appropriate number of workers for parallel processing

### **3. Validation**
- Verify gene overlap between training and test data
- Check that preprocessing produces reasonable values
- Monitor memory usage for very large datasets

## Troubleshooting

### **Common Issues**

1. **"No overlapping genes"**
   - Check gene naming conventions
   - Verify training data format
   - Ensure test data contains expected genes

2. **Memory errors with large datasets**
   - Use file-based loading instead of in-memory
   - Reduce batch size
   - Use backed mode for very large files

3. **Preprocessing errors**
   - Verify data format compatibility
   - Check for NaN or infinite values in input data
   - Ensure positive values for log transformation

### **Debugging Tips**
- Enable logging to see gene transformation details
- Check intermediate data shapes and gene names
- Verify preprocessing pipeline steps individually

## Summary

The scXpand inference pipeline handles test data with different formats through:

- **Automatic adaptation** to various data format mismatches
- **Consistent preprocessing** identical to training
- **Memory efficient** handling of large datasets
- **Model agnostic** approach across all supported architectures
- **Comprehensive testing** ensures reliability and correctness

This design allows users to run inference on new data without format compatibility concerns, while maintaining the statistical relationships learned during training.
