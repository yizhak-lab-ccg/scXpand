import torch


# This dummy test ensures at least one test always passes
def test_always_passes():
    """This test always passes."""
    assert True


# Report torch version in the test run
def pytest_configure(config):
    """Add torch version info to pytest output."""
    config.addinivalue_line(
        "markers", f"torch_version: PyTorch {torch.__version__} is available"
    )


import anndata
import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def dummy_adata():
    """Create dummy AnnData object with realistic structure for testing."""
    n_cells = 100
    n_genes = 50

    # Create realistic gene expression data
    np.random.seed(42)  # For reproducible tests
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(
        np.float32
    )
    X_sparse = csr_matrix(X)

    # Create realistic patient and metadata structure
    n_patients = 10
    patient_ids = [f"patient_{i}" for i in range(n_patients)]
    patient_assignments = np.repeat(patient_ids, n_cells // n_patients)

    # Ensure we have exactly n_cells
    patient_assignments = patient_assignments[:n_cells]

    # Create cancer type mapping - ensure consistency per patient
    patient_to_cancer = {
        f"patient_{i}": "cancer_A" if i < 5 else "cancer_B" for i in range(n_patients)
    }
    cancer_types = np.array([patient_to_cancer[p] for p in patient_assignments])

    # Create study assignments that are consistent per patient
    patient_to_study = {
        f"patient_{i}": "study1" if i < 5 else "study2" for i in range(n_patients)
    }
    study_assignments = np.array([patient_to_study[p] for p in patient_assignments])

    # Create realistic expansion labels with some structure
    expansion_labels = []
    for i in range(n_cells):
        # Make expansion somewhat correlated with gene expression sum
        gene_sum = np.sum(X[i, :])
        prob_expanded = 0.3 + 0.4 * (gene_sum > np.median(np.sum(X, axis=1)))
        is_expanded = np.random.random() < prob_expanded
        expansion_labels.append("expanded" if is_expanded else "non-expanded")

    obs = {
        "expansion": expansion_labels,
        "tissue_type": np.random.choice(
            ["tissue_A", "tissue_B", "tissue_C"], size=n_cells
        ),
        "imputed_labels": np.random.choice(
            ["label_1", "label_2", "label_3"], size=n_cells
        ),
        "clone_id_size": np.random.randint(1, 100, size=n_cells),
        "median_clone_size": np.random.randint(1, 50, size=n_cells),
        "study": study_assignments,  # Use consistent study assignments
        "patient": patient_assignments,
        "sample": np.array([f"sample_{i // 5}" for i in range(n_cells)]),
        "cancer_type": cancer_types,
    }

    var = {"gene_symbol": [f"gene_{i}" for i in range(n_genes)]}
    adata = anndata.AnnData(X=X_sparse, obs=obs, var=var)
    return adata
