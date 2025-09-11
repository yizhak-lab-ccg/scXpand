# Model Sharing

Guide for sharing your trained scXpand models with the community.




## Upload Steps

### 1. Prepare Your Model Archive

```bash
# Required files: model.joblib/model.pt, data_format.json, parameters.json, *.npz, *.csv
zip -r logistic_model.zip model.joblib data_format.json parameters.json *.npz *.csv
```

### 2. Upload to File Hosting Platform

Upload your ZIP archive to any file hosting platform that provides direct stable public download URLs like Figshare.

### 3. Get Direct Download URL


### 4. Add to scXpand Registry

```python
# In scxpand/pretrained/model_registry.py
"my_tumor_classifier": PretrainedModelInfo(  # model_name: any descriptive identifier
    name="my_tumor_classifier",
    url="https://your-platform.com/direct-download-url.zip",
    version="1.0.0",
),
```

**Important Notes:**
- **`model_name`**: Registry identifier - can be any descriptive string (e.g., "tumor_classifier", "blood_model")
- **Model type**: Automatically detected from `model_type.txt` file in the model archive

**Note**: The system automatically handles files with numeric prefixes (e.g., `57702349_data_format.json`) by normalizing them to standard names (`data_format.json`).

### 5. Test

```python
import scxpand

# Test download and inference
results = scxpand.run_inference(
    model_name="my_tumor_classifier",  # Use your model_name (registry identifier)
    data_path="test_data.h5ad",
    device=None  # Auto-detect best available device
)
```

## Usage

Users can access models in two ways:

### Registry Models (Curated)
```bash
# List curated models
scxpand list-models

# Use curated model
scxpand predict --data_path data.h5ad --model_name pan_cancer_logistic
```

### Direct URL Access (Any External Model)
```bash
# Use any external model directly via URL - no code changes needed!
scxpand predict --data_path data.h5ad --model_url "https://your-platform.com/model.zip"
```

```python
# Python API also supports both approaches
import scxpand

# Registry model
results = scxpand.run_inference(
    model_name="pan_cancer_logistic",  # model_name: registry identifier
    data_path="data.h5ad",
    device=None  # Auto-detect device
)

# Direct URL (seamless model sharing!)
results = scxpand.run_inference(
    model_url="https://your-platform.com/model.zip",
    data_path="data.h5ad",
    device=None  # Auto-detect device
)
```
