import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scxpand.util.metrics import calculate_metrics, compute_basic_metrics, safe_hmean


class TestMetrics:
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing metrics."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.4, 0.5, 0.8, 0.9, 0.95])

        # Create a mock AnnData object with required columns for calculate_metrics
        obs = {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"],
            "imputed_labels": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A"],
            "tissue_type": ["X", "X", "Y", "Y", "X", "X", "Y", "Y", "X", "X"],
        }
        adata = AnnData(X=np.random.rand(10, 5), obs=pd.DataFrame(obs))

        return y_true, y_pred_prob, adata

    def test_safe_hmean(self):
        """Test safe_hmean function."""
        # Test with normal values
        values = [2.0, 4.0, 8.0]
        result = safe_hmean(values)
        expected = 3 / (1 / 2 + 1 / 4 + 1 / 8)
        assert np.isclose(result, expected)

        # Test with a zero value
        values = [2.0, 0.0, 8.0]
        result = safe_hmean(values)
        assert result == 0.0

        # Test with all NaN values
        values = [np.nan, np.nan]
        result = safe_hmean(values)
        assert np.isnan(result)

        # Test with some NaN values
        values = [2.0, np.nan, 8.0]
        result = safe_hmean(values)
        expected = 2 / (1 / 2 + 1 / 8)
        assert np.isclose(result, expected)

    def test_compute_basic_metrics(self, mock_data):
        """Test compute_basic_metrics function."""
        y_true, y_pred_prob, _ = mock_data

        metrics = compute_basic_metrics(y_true=y_true, y_pred_prob=y_pred_prob)

        # Check that the metrics dictionary contains the expected keys
        assert "error_rate" in metrics
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics
        assert "AUROC" in metrics
        assert "F1" in metrics
        assert "RMSE" in metrics
        assert "positives_rate" in metrics

        # Check that the metrics are within expected ranges
        assert 0 <= metrics["error_rate"] <= 1
        assert 0 <= metrics["AUROC"] <= 1
        assert 0 <= metrics["F1"] <= 1

    def test_calculate_metrics(self, mock_data):
        """Test the main metrics calculation function."""
        y_true, y_pred_prob, adata = mock_data
        row_inds_dev = list(range(len(y_true)))

        # Use AnnData slicing to get a view of the obs dataframe instead of using iloc
        # This avoids the ImplicitModificationWarning
        subset_adata = adata[row_inds_dev]
        metrics = calculate_metrics(
            y_true=y_true, y_pred_prob=y_pred_prob, obs_df=subset_adata.obs
        )

        # Check that the metrics dictionary has the expected structure
        # The metrics should include overall metrics like AUROC and F1
        assert "AUROC" in metrics
        assert "F1" in metrics

        # Check that per-category metrics are calculated (A__X and B__Y)
        assert "A__X" in metrics
        assert "B__Y" in metrics

        # Check that the per-category metrics contain the expected keys
        category_metrics = metrics["A__X"]
        assert "AUROC" in category_metrics
        assert "F1" in category_metrics
        assert "error_rate" in category_metrics
