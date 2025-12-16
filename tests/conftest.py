import anndata
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

# =============================================================================
# NumPy 2.0 Polyfills
# =============================================================================

# Polyfill for numpy.trapz (removed in NumPy 2.0)
try:
    from scipy.integrate import trapezoid
except ImportError:

    def trapezoid(y, x=None, dx=1.0, axis=-1):
        return np.sum(y * dx, axis=axis)  # fallback


if not hasattr(np, "trapz"):
    np.trapz = trapezoid

# Polyfill for numpy.in1d (removed in NumPy 2.0, replaced by isin)
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# =============================================================================
# AnnData Compatibility Patches
# =============================================================================

# Patch anndata.read_h5ad to avoid backed="r" issues on Python 3.13/Windows
# This fixes:
# - AttributeError: 'backed_csr_matrix' object has no attribute '_validate_indices'
# - PermissionError: [WinError 32] file in use
_original_read_h5ad = anndata.read_h5ad


def _patched_read_h5ad(filename, backed=None, *args, **kwargs):
    """Force backed=None to avoid Python 3.13 and Windows compatibility issues."""
    if backed is not None:
        backed = None
    return _original_read_h5ad(filename, *args, backed=backed, **kwargs)


anndata.read_h5ad = _patched_read_h5ad

# Enable writing nullable string arrays to HDF5 files (required for pandas StringDtype)
anndata.settings.allow_write_nullable_strings = True


# =============================================================================
# Pytest Configuration
# =============================================================================


def test_always_passes():
    """This test always passes."""
    assert True


def pytest_configure(config):
    """Add torch version info to pytest output."""
    config.addinivalue_line(
        "markers", f"torch_version: PyTorch {torch.__version__} is available"
    )


# =============================================================================
# Fixtures
# =============================================================================


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
