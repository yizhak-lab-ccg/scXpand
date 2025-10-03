<div align="center">
  <p style="margin: 0 0 20px 0;">
    <a href="https://pypi.org/project/scxpand"><img src="https://img.shields.io/pypi/v/scxpand" alt="PyPI version" /></a>
    <a href="https://pypi.org/project/scxpand"><img src="https://img.shields.io/pypi/pyversions/scxpand" alt="Python versions" /></a>
    <a href="https://github.com/yizhak-lab-ccg/scXpand/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/yizhak-lab-ccg/scXpand/test.yml?branch=main&label=tests" alt="Tests" /></a>
    <a href="https://scxpand.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/scxpand?branch=main" alt="Documentation" /></a>
    <a href="https://pepy.tech/project/scxpand"><img src="https://static.pepy.tech/badge/scxpand" alt="Downloads" /></a>
  </p>
  <img src="https://raw.githubusercontent.com/yizhak-lab-ccg/scXpand/main/docs/images/scXpand_logo_gray.png" alt="scXpand Logo" width="280"/>

  <h1 style="margin: 10px 0 5px 0;">scXpand: Pan-cancer Detection of T-cell Clonal Expansion</h1>
</div>

  <p style="margin: 0 0 25px 0; font-size: 0.95em; max-width: 800px; line-height: 1.4;">
    Detect T-cell clonal expansion from single-cell RNA sequencing data without paired TCR sequencing
  </p>

  <p style="margin: 0; text-align: center;">
    <a href="https://www.biorxiv.org/content/10.1101/2025.09.14.676069v1" style="margin: 0 8px;">Preprint</a> •
    <a href="https://scxpand.readthedocs.io/en/latest/" style="margin: 0 8px;">Documentation</a> •
    <a href="#installation" style="margin: 0 8px;">Installation</a> •
    <a href="#quick-start" style="margin: 0 8px;">Quick Start</a> •
    <a href="https://scxpand.readthedocs.io/en/latest/user_guide.html" style="margin: 0 8px;">Usage Guide</a> •
    <a href="#citation" style="margin: 0 8px;">Citation</a>
  </p>
</div>

<div style="width: 100vw; margin-left: calc(-50vw + 50%); margin-right: calc(-50vw + 50%); margin-top: 20px; margin-bottom: 40px; padding: 0 40px;">
  <img src="https://raw.githubusercontent.com/yizhak-lab-ccg/scXpand/main/docs/images/scXpand_datasets.jpeg" alt="scXpand Datasets Overview" style="width: 100%; height: auto; display: block; margin: 0; padding: 0;"/>
</div>

A framework for predicting T-cell clonal expansion from single-cell RNA sequencing data.

**Manuscript in preparation** - detailed methodology and benchmarks coming soon.

**[View full documentation](https://scxpand.readthedocs.io/en/latest/)** for comprehensive guides and API reference.

---

## Features

- **Multiple Model Architectures**:
  - **Autoencoder-based**: Encoder-decoder with reconstruction and classification heads
  - **MLP**: Multi-layer perceptron
  - **LightGBM**: Gradient boosted decision trees
  - **Linear Models**: Logistic regression and support vector machines
- **Scalable Processing**: Handles millions of cells with memory-efficient data streaming from disk during training
- **Automated Hyperparameter Optimization**: Built-in Optuna integration for model tuning

---

## Installation

For detailed installation instructions, please refer to our **[Installation Guide](https://scxpand.readthedocs.io/en/latest/installation.html)**.

### Published Version Install

**CUDA version (NVIDIA GPU):**

With pip:
```bash
pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128
```

With uv:
```bash
uv pip install --upgrade scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

**CPU/Apple Silicon/Other GPUs:**

With pip:
```bash
pip install --upgrade scxpand
```

With uv:
```bash
uv pip install --upgrade scxpand
```

### Development Setup (Install from Source)

See the [Installation Guide](https://scxpand.readthedocs.io/en/latest/installation.html)

---

## Quick Start

```python
import scxpand
# Make sure that "your_data.h5ad" includes only T cells for the results to be meaningful
# Ensure that "your_data.var_names" are provided as Ensembl IDs (as the pre-trained models were trained using this gene representation)
# Please refer to our documentation for more information

# List available pre-trained models
scxpand.list_pretrained_models()

# Run inference with automatic model download
results = scxpand.run_inference(
    model_name="pan_cancer_autoencoder",  # default model
    data_path="your_data.h5ad"
)

# Access predictions
predictions = results.predictions
if results.has_metrics:
    print(f"AUROC: {results.get_auroc():.3f}")
```

See our **[Tutorial Notebook](docs/notebooks/scxpand_tutorial.ipynb)** for a complete example with data preprocessing, T-cell filtering, gene ID conversion, and model application using a real breast cancer dataset.

---

## Documentation

**Setup & Getting Started:**
- **[Installation Guide](https://scxpand.readthedocs.io/en/latest/installation.html)** - Setup for local development of scXpand
- **[User Guide](https://scxpand.readthedocs.io/en/latest/user_guide.html)** - Quick start and comprehensive workflow guide
- **[Data Format](https://scxpand.readthedocs.io/en/latest/data_format.html)** - Input data requirements and specifications

**Using Pre-trained Models:**
- **[Model Inference](https://scxpand.readthedocs.io/en/latest/model_inference.html)** - Run predictions on new data with pre-trained models

**Training Your Own Models:**
- **[Model Training](https://scxpand.readthedocs.io/en/latest/model_training.html)** - Train models with CLI and programmatic API
- **[Hyperparameter Optimization](https://scxpand.readthedocs.io/en/latest/hyperparameter_optimization.html)** - Automated model tuning with Optuna

**Understanding Results:**
- **[Model Architectures](https://scxpand.readthedocs.io/en/latest/model_architectures.html)** - Detailed architecture descriptions and configurations
- **[Evaluation Metrics](https://scxpand.readthedocs.io/en/latest/evaluation_metrics.html)** - Performance assessment and interpretation
- **[Output Format](https://scxpand.readthedocs.io/en/latest/output_format.html)** - Understanding model outputs and results

**[📖 Full Documentation](https://scxpand.readthedocs.io/en/latest/)** - Complete guides, API reference, and interactive tutorials

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use scXpand in your research, please cite:

> Shorer, O., Amit, R., and Yizhak, K. (2025). scXpand: Pan-cancer detection of T-cell clonal expansion from single-cell RNA sequencing without paired single-cell TCR sequencing.
> Preprint at bioRxiv, https://doi.org/10.1101/2025.09.14.676069.

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@article{shorer2025scxpand,
  title={scXpand: Pan-cancer detection of T-cell clonal expansion from single-cell RNA sequencing without paired single-cell TCR sequencing},
  author={Shorer, Ofir and Amit, Ron and Yizhak, Keren},
  year={2025},
  journal={bioRxiv},
  doi={https://doi.org/10.1101/2025.09.14.676069}
}
```
</details>

---

<div align="center">
  <p><em>This project was created in favor of the scientific community worldwide, with a special dedication to the cancer research community.</em></p>
  <p><em>We hope you'll find this repository helpful, and we warmly welcome any requests or suggestions - please don't hesitate to reach out!</em></p>

  <a href="https://mapmyvisitors.com/web/1bz9s">
     <img src="https://mapmyvisitors.com/map.png?d=hwaNi7bepoJeL9CYnuB3WjMT-liNG4MvcmwecZk3aNA&cl=ffffff" alt="Visitor Map">
  </a>
</div>
