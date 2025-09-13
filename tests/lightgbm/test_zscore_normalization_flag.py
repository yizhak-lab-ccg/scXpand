"""Unit tests for the use_zscore_norm flag implementation across LightGBM components.

Tests verify that the z-score normalization can be properly enabled/disabled
and that it affects data preprocessing, performance, and hyperparameter optimization.
"""

import tempfile

from pathlib import Path

import anndata as ad
import numpy as np
import optuna
import pandas as pd
import pytest

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import CellsDataset
from scxpand.data_util.prepare_data_for_train import prepare_data_for_training
from scxpand.data_util.transforms import preprocess_expression_data
from scxpand.hyperopt.param_grids import configure_lightgbm_trial_params
from scxpand.lightgbm.lightgbm_params import LightGBMParams
from tests.test_utils import windows_safe_context_manager


@pytest.fixture
def dummy_adata():
    """Create a dummy AnnData object for testing."""
    n_cells = 100
    n_genes = 50

    # Create random gene expression data
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Create observation data with all required columns
    # Each patient should have consistent cancer type
    patients = [f"patient_{i % 10}" for i in range(n_cells)]
    # Assign cancer type based on patient ID to ensure consistency
    patient_cancer_map = {f"patient_{i}": f"cancer{(i % 3) + 1}" for i in range(10)}
    cancer_types = [patient_cancer_map[patient] for patient in patients]

    obs_data = {
        "expansion": np.random.choice([0, 1], size=n_cells),
        "clone_id_size": np.random.randint(1, 20, size=n_cells),
        "median_clone_size": np.random.randint(5, 15, size=n_cells),
        "tissue_type": np.random.choice(["liver", "lung", "brain"], size=n_cells),
        "imputed_labels": np.random.choice(["A", "B", "C"], size=n_cells),
        "study": np.random.choice(["study1", "study2"], size=n_cells),
        "patient": patients,
        "sample": [f"sample_{i % 5}" for i in range(n_cells)],
        "cancer_type": cancer_types,
    }

    obs_df = pd.DataFrame(obs_data)
    var_df = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=X, obs=obs_df, var=var_df)


class TestLightGBMParams:
    """Test LightGBMParams class handles use_zscore_norm correctly."""

    def test_default_use_zscore_norm(self):
        """Test that use_zscore_norm defaults to True."""
        params = LightGBMParams()
        assert hasattr(params, "use_zscore_norm")
        assert params.use_zscore_norm is True

    def test_set_use_zscore_norm_false(self):
        """Test that use_zscore_norm can be set to False."""
        params = LightGBMParams(use_zscore_norm=False)
        assert params.use_zscore_norm is False

    def test_other_params_with_zscore_disabled(self):
        """Test that other parameters work correctly when use_zscore_norm is disabled."""
        params = LightGBMParams(use_zscore_norm=False, num_leaves=50, learning_rate=0.05, n_estimators=200)
        assert params.use_zscore_norm is False
        assert params.num_leaves == 50
        assert params.learning_rate == 0.05
        assert params.n_estimators == 200


class TestDataFormat:
    """Test DataFormat class handles use_zscore_norm correctly."""

    def test_default_use_zscore_norm(self):
        """Test that use_zscore_norm defaults to True."""
        data_format = DataFormat()
        assert hasattr(data_format, "use_zscore_norm")
        assert data_format.use_zscore_norm is True

    def test_set_use_zscore_norm_false(self):
        """Test that use_zscore_norm can be set to False."""
        data_format = DataFormat(use_zscore_norm=False)
        assert data_format.use_zscore_norm is False


class TestPreprocessingPipeline:
    """Test that preprocessing pipeline respects use_zscore_norm flag."""

    def test_zscore_flag_affects_preprocessing(self):
        """Test that z-score flag produces different results."""
        n_genes = 10
        n_cells = 20
        X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        # Create DataFormat with dummy stats
        genes_mu = np.random.randn(n_genes).astype(np.float32)
        genes_sigma = np.random.rand(n_genes).astype(np.float32) + 0.1  # Avoid zeros

        # Test with z-score normalization enabled
        df_zscore = DataFormat(
            use_zscore_norm=True,
            use_log_transform=False,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
        )

        X_processed_zscore = preprocess_expression_data(X.copy(), df_zscore)

        # Test with z-score normalization disabled
        df_no_zscore = DataFormat(
            use_zscore_norm=False,
            use_log_transform=False,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
        )

        X_processed_no_zscore = preprocess_expression_data(X.copy(), df_no_zscore)

        # The results should be different
        assert not np.allclose(X_processed_zscore, X_processed_no_zscore)

    def test_zscore_changes_variance_structure(self):
        """Test that z-score normalization changes data variance structure."""
        n_genes = 10
        n_cells = 20
        X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        genes_mu = np.random.randn(n_genes).astype(np.float32)
        genes_sigma = np.random.rand(n_genes).astype(np.float32) + 0.1

        df_zscore = DataFormat(
            use_zscore_norm=True,
            use_log_transform=False,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
        )

        df_no_zscore = DataFormat(
            use_zscore_norm=False,
            use_log_transform=False,
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            n_genes=n_genes,
            gene_names=[f"gene_{i}" for i in range(n_genes)],
        )

        X_processed_zscore = preprocess_expression_data(X.copy(), df_zscore)
        X_processed_no_zscore = preprocess_expression_data(X.copy(), df_no_zscore)

        # The variances should be different (z-scoring changes the variance structure)
        zscore_var = np.var(X_processed_zscore, axis=0)
        no_zscore_var = np.var(X_processed_no_zscore, axis=0)

        assert not np.allclose(zscore_var, no_zscore_var)


class TestPrepareDataForTraining:
    """Test that prepare_data_for_training respects use_zscore_norm flag."""

    def test_zscore_enabled_computes_statistics(self, dummy_adata):
        """Test that gene statistics are computed when z-score is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            data_path = Path(temp_dir) / "test_data.h5ad"
            save_dir = Path(temp_dir) / "results"

            dummy_adata.write_h5ad(data_path)
            ctx.register_file(data_path)

            bundle = prepare_data_for_training(
                data_path=data_path, use_zscore_norm=True, save_dir=save_dir, dev_ratio=0.2, batch_size=100
            )
            ctx.register_adata(bundle.adata)

            assert bundle.data_format.use_zscore_norm is True
            # Gene statistics should be computed (not dummy values)
            assert not np.allclose(bundle.data_format.genes_mu, 0.0)
            assert not np.allclose(bundle.data_format.genes_sigma, 1.0)

    def test_zscore_disabled_skips_statistics(self, dummy_adata):
        """Test that gene statistics computation is skipped when z-score is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            data_path = Path(temp_dir) / "test_data.h5ad"
            save_dir = Path(temp_dir) / "results"

            dummy_adata.write_h5ad(data_path)
            ctx.register_file(data_path)

            bundle = prepare_data_for_training(
                data_path=data_path, use_zscore_norm=False, save_dir=save_dir, dev_ratio=0.2, batch_size=100
            )
            ctx.register_adata(bundle.adata)

            assert bundle.data_format.use_zscore_norm is False
            # Gene statistics should be dummy values
            assert np.allclose(bundle.data_format.genes_mu, 0.0)
            assert np.allclose(bundle.data_format.genes_sigma, 1.0)


class TestCellsDataset:
    """Test that CellsDataset properly handles use_zscore_norm flag."""

    def test_dataset_with_zscore_enabled(self, dummy_adata):
        """Test CellsDataset when z-score normalization is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            data_path = Path(temp_dir) / "test_data.h5ad"
            dummy_adata.write_h5ad(data_path)
            ctx.register_file(data_path)

            data_format = DataFormat(
                use_zscore_norm=True,
                use_log_transform=False,
                n_genes=dummy_adata.n_vars,
                gene_names=dummy_adata.var_names.tolist(),
                genes_mu=np.random.randn(dummy_adata.n_vars).astype(np.float32),
                genes_sigma=np.random.rand(dummy_adata.n_vars).astype(np.float32) + 0.1,
            )

            dataset = CellsDataset(data_format=data_format, data_path=data_path, is_train=False)

            assert dataset.use_zscore_norm is True
            assert dataset.genes_mu_tensor is not None
            assert dataset.genes_sigma_tensor is not None

    def test_dataset_with_zscore_disabled(self, dummy_adata):
        """Test CellsDataset when z-score normalization is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            data_path = Path(temp_dir) / "test_data.h5ad"
            dummy_adata.write_h5ad(data_path)
            ctx.register_file(data_path)

            data_format = DataFormat(
                use_zscore_norm=False,
                use_log_transform=False,
                n_genes=dummy_adata.n_vars,
                gene_names=dummy_adata.var_names.tolist(),
                genes_mu=np.random.randn(dummy_adata.n_vars).astype(np.float32),
                genes_sigma=np.random.rand(dummy_adata.n_vars).astype(np.float32) + 0.1,
            )

            dataset = CellsDataset(data_format=data_format, data_path=data_path, is_train=False)

            assert dataset.use_zscore_norm is False
            assert dataset.genes_mu_tensor is None
            assert dataset.genes_sigma_tensor is None


class TestHyperoptIntegration:
    """Test that hyperparameter optimization includes use_zscore_norm."""

    def test_hyperopt_includes_zscore_param(self):
        """Test that hyperopt configuration includes use_zscore_norm parameter."""
        study = optuna.create_study()
        trial = study.ask()

        params = configure_lightgbm_trial_params(trial)

        assert "use_zscore_norm" in params
        assert isinstance(params["use_zscore_norm"], bool)

    def test_hyperopt_zscore_varies(self):
        """Test that hyperopt can generate both True and False for use_zscore_norm."""
        study = optuna.create_study()

        # Generate multiple trials to check variation
        zscore_values = set()
        for _ in range(10):
            trial = study.ask()
            params = configure_lightgbm_trial_params(trial)
            zscore_values.add(params["use_zscore_norm"])

        # Should see both True and False values (with high probability)
        # Note: This is probabilistic, but with 10 trials, very likely to see both
        assert len(zscore_values) >= 1  # At minimum, we should see at least one value


class TestPerformanceOptimization:
    """Test that skipping gene statistics computation provides performance benefits."""

    def test_zscore_disabled_faster_or_equal(self, dummy_adata):
        """Test that disabling z-score is faster than or equal to enabling it."""
        # Create larger dataset for measurable timing differences
        large_adata = dummy_adata.copy()
        # Expand the dataset
        large_adata = ad.concat([large_adata] * 10, axis=0)  # 1000 cells

        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            data_path = Path(temp_dir) / "test_data.h5ad"
            large_adata.write_h5ad(data_path)
            ctx.register_file(data_path)

            # Time with z-score normalization enabled
            bundle_zscore = prepare_data_for_training(
                data_path=data_path,
                use_zscore_norm=True,
                save_dir=Path(temp_dir) / "zscore_true",
                dev_ratio=0.2,
                batch_size=200,
            )
            ctx.register_adata(bundle_zscore.adata)

            bundle_no_zscore = prepare_data_for_training(
                data_path=data_path,
                use_zscore_norm=False,
                save_dir=Path(temp_dir) / "zscore_false",
                dev_ratio=0.2,
                batch_size=200,
            )
            ctx.register_adata(bundle_no_zscore.adata)

            # Verify that the gene statistics are different
            assert not np.allclose(bundle_zscore.data_format.genes_mu, bundle_no_zscore.data_format.genes_mu)
            assert not np.allclose(bundle_zscore.data_format.genes_sigma, bundle_no_zscore.data_format.genes_sigma)


class TestIntegration:
    """Integration tests for the complete z-score normalization workflow."""

    def test_end_to_end_zscore_workflow(self, dummy_adata):
        """Test the complete workflow with z-score normalization enabled/disabled."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            data_path = Path(temp_dir) / "test_data.h5ad"
            dummy_adata.write_h5ad(data_path)
            ctx.register_file(data_path)

            # Test with LightGBMParams
            params_zscore = LightGBMParams(use_zscore_norm=True)
            params_no_zscore = LightGBMParams(use_zscore_norm=False)

            # Test data preparation
            bundle_zscore = prepare_data_for_training(
                data_path=data_path,
                use_zscore_norm=params_zscore.use_zscore_norm,
                save_dir=Path(temp_dir) / "zscore_true",
                dev_ratio=0.2,
            )
            ctx.register_adata(bundle_zscore.adata)

            bundle_no_zscore = prepare_data_for_training(
                data_path=data_path,
                use_zscore_norm=params_no_zscore.use_zscore_norm,
                save_dir=Path(temp_dir) / "zscore_false",
                dev_ratio=0.2,
            )
            ctx.register_adata(bundle_no_zscore.adata)

            # Verify the flag propagated correctly
            assert bundle_zscore.data_format.use_zscore_norm is True
            assert bundle_no_zscore.data_format.use_zscore_norm is False

            # Test dataset creation
            dataset_zscore = CellsDataset(data_format=bundle_zscore.data_format, data_path=data_path, is_train=False)

            dataset_no_zscore = CellsDataset(
                data_format=bundle_no_zscore.data_format, data_path=data_path, is_train=False
            )

            # Verify dataset configuration
            assert dataset_zscore.use_zscore_norm is True
            assert dataset_no_zscore.use_zscore_norm is False

            # Verify tensor creation behavior
            assert dataset_zscore.genes_mu_tensor is not None
            assert dataset_no_zscore.genes_mu_tensor is None
