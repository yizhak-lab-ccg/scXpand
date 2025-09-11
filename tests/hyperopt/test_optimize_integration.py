"""Integration tests for hyperparameter optimization functionality."""

import tempfile

from pathlib import Path
from unittest.mock import patch

from scxpand.main import optimize, optimize_all
from tests.test_utils import create_temp_h5ad_file, windows_safe_context_manager


def test_optimize_single_model(dummy_adata):
    """Test hyperparameter optimization for a single model type."""
    with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
        # Create test file
        test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
        ctx.register_file(test_file_path)

        results = optimize(
            model_type="mlp",
            data_path=test_file_path,
            n_trials=2,  # Small number for testing
            storage_path=Path(temp_dir) / "optuna_studies",
            num_workers=0,
            fail_fast=True,  # Fail immediately on bugs for integration tests
            n_epochs=1,  # Quick training for testing
        )

        # Since optuna studies are complex objects, we mainly check that
        # the function completes without error
        assert results is None  # optimize doesn't return anything

        # Check that study files were created
        study_path = Path(temp_dir) / "optuna_studies"
        assert study_path.exists(), "Study directory should be created"


def test_optimize_all_models(dummy_adata):
    """Test hyperparameter optimization for all model types."""
    with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
        # Create test file
        test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
        ctx.register_file(test_file_path)

        # Mock optimize to track calls
        with patch("scxpand.main.optimize") as mock_optimize:
            optimize_all(
                data_path=test_file_path,
                n_trials=2,  # Small number for testing
                storage_path=Path(temp_dir) / "optuna_studies",
                num_workers=0,
                fail_fast=True,  # Fail immediately on bugs for integration tests
            )

            # Check that optimize was called for each model type
            assert mock_optimize.call_count > 0, "Should optimize at least one model"

            # Check that each call was made with correct base parameters
            for call in mock_optimize.call_args_list:
                args, kwargs = call
                assert "data_path" in kwargs, "data_path should be passed"
                assert kwargs["n_trials"] == 2, "n_trials should be passed"
                assert kwargs["num_workers"] == 0, "num_workers should be passed"
                assert kwargs["fail_fast"] is True, "fail_fast should be passed"
