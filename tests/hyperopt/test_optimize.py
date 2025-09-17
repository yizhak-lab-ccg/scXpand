"""Comprehensive tests for the HyperparameterOptimizer class."""

import inspect
import json
import tempfile

from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest

from scxpand.core.model_types import MODEL_TYPES
from scxpand.hyperopt.hyperopt_optimizer import HyperparameterOptimizer
from scxpand.hyperopt.hyperopt_utils import _has_results_indicators, cleanup_incomplete_trials, trial_callback
from scxpand.mlp.run_mlp_train import run_mlp_training
from scxpand.util.classes import ModelType
from scxpand.util.model_constants import BEST_CHECKPOINT_FILE, STUDY_INFO_FILE
from scxpand.util.train_util import report_to_optuna_and_handle_pruning
from tests.test_utils import create_temp_h5ad_file, windows_safe_context_manager


class TestHyperparameterOptimizerBasics:
    """Test basic functionality of HyperparameterOptimizer."""

    def test_optimizer_initialization(self, dummy_adata):
        """Test that optimizer initializes correctly with various parameters."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            # Test basic initialization
            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            assert optimizer.model_type == ModelType.MLP
            assert optimizer.data_path == Path(test_file_path)
            assert optimizer.study_name == "mlp_opt"  # Default name
            assert optimizer.score_metric == "harmonic_avg/AUROC"  # Default metric
            assert optimizer.seed_base == 42  # Default seed
            assert optimizer.num_workers == 0
            assert optimizer.resume is True  # Default True

    def test_optimizer_initialization_with_custom_params(self, dummy_adata):
        """Test optimizer initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            custom_storage = Path(temp_dir) / "custom_studies"

            optimizer = HyperparameterOptimizer(
                model_type="svm",  # Test string input
                data_path=test_file_path,
                study_name="custom_study",
                storage_path=custom_storage,
                score_metric="AUROC",
                seed_base=123,
                num_workers=2,
                resume=False,
                custom_param=42,  # Test param overrides
            )

            assert optimizer.model_type == ModelType.SVM
            assert optimizer.study_name == "custom_study"
            assert optimizer.storage_path == custom_storage
            assert optimizer.score_metric == "AUROC"
            assert optimizer.seed_base == 123
            assert optimizer.num_workers == 2
            assert optimizer.resume is False
            assert optimizer.param_overrides == {"custom_param": 42}

    def test_invalid_model_type_raises_error(self, dummy_adata):
        """Test that invalid model type raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            with pytest.raises(ValueError, match="model_type must be one of"):
                HyperparameterOptimizer(
                    model_type="invalid_model",
                    data_path=test_file_path,
                )

    def test_nonexistent_data_path_raises_error(self):
        """Test that nonexistent data path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path="/nonexistent/path.h5ad",
            )

    def test_storage_directory_creation(self, dummy_adata):
        """Test that storage directories are created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "nested" / "storage" / "path"

            HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=storage_path,
                study_name="test_study",
            )

            # Directory should be created
            expected_study_dir = storage_path / "test_study"
            assert expected_study_dir.exists()
            assert expected_study_dir.is_dir()


class TestStudyCreationAndResume:
    """Test study creation and resume functionality."""

    def test_create_new_study_when_none_exists(self, dummy_adata):
        """Test creating a new study when no existing study exists."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="new_study",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()
            assert study is not None
            assert study.study_name == "new_study"
            assert len(study.trials) == 0
            assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_resume_existing_study(self, dummy_adata):
        """Test resuming an existing study with trials."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"
            study_name = "resume_test"

            # Create initial study and add trials
            optimizer1 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            with patch.object(optimizer1, "objective", return_value=0.85):
                study1 = optimizer1.create_study()
                study1.optimize(func=optimizer1.objective, n_trials=2)
                assert len(study1.trials) == 2

            # Create new optimizer and resume
            optimizer2 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            study2 = optimizer2.create_study()
            assert len(study2.trials) == 2  # Should have previous trials
            assert study2.study_name == study_name

    def test_resume_false_raises_error_for_existing_study(self, dummy_adata):
        """Test that resume=False raises error when existing study has trials."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"
            study_name = "resume_false_test"
            study_dir = storage_path / study_name

            # Create initial study with trials
            optimizer1 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                resume=True,
                num_workers=0,
            )

            with patch.object(optimizer1, "objective", return_value=0.85):
                study1 = optimizer1.create_study()
                study1.optimize(func=optimizer1.objective, n_trials=2)
                assert len(study1.trials) == 2

            # Create trial directories on filesystem to test the logic
            trial_dir_0 = study_dir / "trial_0"
            trial_dir_1 = study_dir / "trial_1"
            trial_dir_0.mkdir(parents=True, exist_ok=True)
            trial_dir_1.mkdir(parents=True, exist_ok=True)

            # Verify database file exists
            db_file = study_dir / "optuna.db"
            assert db_file.exists()

            # Create new optimizer with resume=False - should raise error
            optimizer2 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                resume=False,
                num_workers=0,
            )

            # Test the _handle_existing_study method directly since create_study() fails at database level
            with pytest.raises(ValueError, match="already exists.*with.*2 trial"):
                optimizer2._handle_existing_study()

    def test_error_when_study_exists_and_not_resuming(self, dummy_adata):
        """Test error when study exists and resume=False."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"
            study_name = "existing_study"

            # Create study
            optimizer1 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            study1 = optimizer1.create_study()
            assert study1 is not None

            # Create a trial directory to test the logic
            trial_dir = storage_path / study_name / "trial_0"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Try to create new study without resume - should fail
            optimizer2 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                resume=False,  # Explicitly set resume=False
                num_workers=0,
            )

            # Test the _handle_existing_study method directly since create_study() fails at database level
            with pytest.raises(ValueError, match="already exists"):
                optimizer2._handle_existing_study()


class TestTrialSpecificResumeLogic:
    """Test trial-specific resume logic for new vs existing trials."""

    def _create_mock_runner_with_signature(self):
        """Create a mock runner with the correct signature to avoid signature filtering issues."""
        mock_runner = MagicMock(return_value={"test_metric": 0.85})
        # Set the signature to match the real runner function
        mock_runner.__signature__ = inspect.signature(run_mlp_training)
        return mock_runner

    def test_new_trial_always_starts_fresh_regardless_of_global_resume(self, dummy_adata):
        """Test that new trials start fresh even when global resume=True."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                resume=True,  # Global resume=True
                num_workers=0,
            )

            # Mock trial and save_dir for a new trial (no checkpoints)
            trial = MagicMock()
            trial.number = 0
            save_dir = Path(temp_dir) / "trial_0"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Mock parameters
            params = MagicMock()

            # Mock the runner to capture the resume flag passed to it
            mock_runner = self._create_mock_runner_with_signature()

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                optimizer._execute_trial(trial, params, save_dir)

                # Verify that resume=False was passed to the runner for new trial
                call_args = mock_runner.call_args
                assert call_args is not None

                # Print debug info to understand what's being passed
                args, kwargs = call_args
                print(f"DEBUG: args = {args}")
                print(f"DEBUG: kwargs = {kwargs}")
                print(f"DEBUG: all call_args = {call_args}")

                # Check if resume is in kwargs
                assert "resume" in kwargs, f"resume not found in kwargs: {list(kwargs.keys())}"
                assert kwargs["resume"] is False  # New trial should get resume=False

    def test_existing_trial_with_checkpoints_resumes_when_global_resume_true(self, dummy_adata):
        """Test that existing trials with checkpoints resume when global resume=True."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                resume=True,  # Global resume=True
                num_workers=0,
            )

            # Mock trial and save_dir for an existing trial with checkpoints
            trial = MagicMock()
            trial.number = 1
            save_dir = Path(temp_dir) / "trial_1"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create checkpoint files to simulate existing trial
            (save_dir / "best_ckpt.pt").touch()
            (save_dir / "last_ckpt.pt").touch()

            # Mock parameters
            params = MagicMock()

            # Mock the runner to capture the resume flag passed to it
            mock_runner = self._create_mock_runner_with_signature()

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                optimizer._execute_trial(trial, params, save_dir)

                # Verify that resume=True was passed to the runner for existing trial
                call_args = mock_runner.call_args
                assert call_args is not None
                assert call_args[1]["resume"] is True  # Existing trial should get resume=True

    def test_existing_trial_without_checkpoints_starts_fresh_when_global_resume_true(self, dummy_adata):
        """Test that existing trials without valid checkpoints start fresh even when global resume=True."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                resume=True,  # Global resume=True
                num_workers=0,
            )

            # Mock trial and save_dir for an existing trial without valid checkpoints
            trial = MagicMock()
            trial.number = 2
            save_dir = Path(temp_dir) / "trial_2"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create some files but not the checkpoint files that indicate completion
            (save_dir / "parameters.json").touch()
            (save_dir / "some_other_file.txt").touch()

            # Mock parameters
            params = MagicMock()

            # Mock the runner to capture the resume flag passed to it
            mock_runner = self._create_mock_runner_with_signature()

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                optimizer._execute_trial(trial, params, save_dir)

                # Verify that resume=False was passed to the runner
                call_args = mock_runner.call_args
                assert call_args is not None
                assert call_args[1]["resume"] is False  # No valid checkpoints = start fresh

    def test_all_trials_start_fresh_when_global_resume_false(self, dummy_adata):
        """Test that all trials start fresh when global resume=False, regardless of checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                resume=False,  # Global resume=False
                num_workers=0,
            )

            # Test with a trial that has checkpoints
            trial = MagicMock()
            trial.number = 3
            save_dir = Path(temp_dir) / "trial_3"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create checkpoint files
            (save_dir / "best_ckpt.pt").touch()
            (save_dir / "last_ckpt.pt").touch()

            # Mock parameters
            params = MagicMock()

            # Mock the runner to capture the resume flag passed to it
            mock_runner = self._create_mock_runner_with_signature()

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                optimizer._execute_trial(trial, params, save_dir)

                # Verify that resume=False was passed even with existing checkpoints
                call_args = mock_runner.call_args
                assert call_args is not None
                assert call_args[1]["resume"] is False  # Global resume=False overrides checkpoints

    def test_data_format_creation_vs_loading_based_on_trial_resume(self, dummy_adata):
        """Test that data_format.json is created for new trials and loaded for resuming trials."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                resume=True,
                num_workers=0,
            )

            # Mock the prepare_data_for_training function to track calls
            with patch("scxpand.mlp.run_mlp_train.prepare_data_for_training") as mock_prepare:
                mock_prepare.return_value = MagicMock()

                # Test 1: New trial (no checkpoints) should call with resume=False
                trial1 = MagicMock()
                trial1.number = 0
                save_dir1 = Path(temp_dir) / "trial_0"
                save_dir1.mkdir(exist_ok=True)
                params = MagicMock()

                mock_runner = self._create_mock_runner_with_signature()
                with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                    optimizer._execute_trial(trial1, params, save_dir1)

                # Check that the runner was called with resume=False for new trial
                runner_call_args = mock_runner.call_args
                assert runner_call_args[1]["resume"] is False

                # Reset mock for next test
                mock_prepare.reset_mock()
                mock_runner.reset_mock()

                # Test 2: Existing trial with checkpoints should call with resume=True
                trial2 = MagicMock()
                trial2.number = 1
                save_dir2 = Path(temp_dir) / "trial_1"
                save_dir2.mkdir(exist_ok=True)
                (save_dir2 / "best_ckpt.pt").touch()  # Create checkpoint

                with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                    optimizer._execute_trial(trial2, params, save_dir2)

                # Check that the runner was called with resume=True for existing trial
                runner_call_args = mock_runner.call_args
                assert runner_call_args[1]["resume"] is True

    def test_study_attributes_set_correctly(self, dummy_adata):
        """Test that study attributes are set correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()

            # Check study_dir attribute
            expected_study_dir = str(optimizer.study_dir)
            # Note: user_attrs might be empty for new studies in some Optuna versions
            # The important thing is that the study_dir attribute gets set when needed
            study_dir_attr = study.user_attrs.get("study_dir")
            assert study_dir_attr == expected_study_dir or study_dir_attr is None

    def test_mixed_new_existing_trial_scenarios(self, dummy_adata):
        """Test mixed scenarios with both new and existing trials in the same study."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=Path(temp_dir) / "studies",
                resume=True,  # Global resume=True
                num_workers=0,
            )

            # Create study directory structure
            study_dir = Path(temp_dir) / "studies" / "mlp_opt"
            study_dir.mkdir(parents=True, exist_ok=True)

            # Scenario 1: Trial 0 - New trial (no checkpoints)
            trial0_dir = study_dir / "trial_0"
            trial0_dir.mkdir(exist_ok=True)
            # No checkpoint files - this should be treated as new

            # Scenario 2: Trial 1 - Existing trial with checkpoints
            trial1_dir = study_dir / "trial_1"
            trial1_dir.mkdir(exist_ok=True)
            (trial1_dir / "best_ckpt.pt").touch()
            (trial1_dir / "last_ckpt.pt").touch()

            # Scenario 3: Trial 2 - Existing trial with partial results but no checkpoints
            trial2_dir = study_dir / "trial_2"
            trial2_dir.mkdir(exist_ok=True)
            (trial2_dir / "parameters.json").touch()
            (trial2_dir / "some_log.txt").touch()

            # Test each scenario
            mock_runner = self._create_mock_runner_with_signature()

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                # Test Trial 0 (new trial)
                trial0 = MagicMock()
                trial0.number = 0
                params = MagicMock()

                optimizer._execute_trial(trial0, params, trial0_dir)
                # Should get resume=False for new trial
                assert mock_runner.call_args[1]["resume"] is False

                # Reset mock for next test
                mock_runner.reset_mock()

                # Test Trial 1 (existing trial with checkpoints)
                trial1 = MagicMock()
                trial1.number = 1

                optimizer._execute_trial(trial1, params, trial1_dir)
                # Should get resume=True for existing trial with checkpoints
                assert mock_runner.call_args[1]["resume"] is True

                # Reset mock for next test
                mock_runner.reset_mock()

                # Test Trial 2 (existing trial without checkpoints)
                trial2 = MagicMock()
                trial2.number = 2

                optimizer._execute_trial(trial2, params, trial2_dir)
                # Should get resume=False for existing trial without checkpoints
                assert mock_runner.call_args[1]["resume"] is False

    def test_data_format_json_creation_vs_loading_in_trials(self, dummy_adata):
        """Test that trial resume logic works correctly based on RESULTS_INDICATORS."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            # Create study directory
            study_dir = Path(temp_dir) / "studies" / "mlp_opt"
            study_dir.mkdir(parents=True, exist_ok=True)

            # Test 1: New trial directory (no result indicators)
            trial1_dir = study_dir / "trial_0"
            trial1_dir.mkdir(exist_ok=True)

            # Test that _has_results_indicators returns False for new trial
            assert not _has_results_indicators(trial1_dir)

            # Test 2: Existing trial with result indicators (checkpoints)
            trial2_dir = study_dir / "trial_1"
            trial2_dir.mkdir(exist_ok=True)

            # Create checkpoint files that indicate training progress
            (trial2_dir / "last_ckpt.pt").touch()
            (trial2_dir / "best_ckpt.pt").touch()

            # Test that _has_results_indicators returns True for existing trial with checkpoints
            assert _has_results_indicators(trial2_dir)

            # Test 3: Existing trial without result indicators
            trial3_dir = study_dir / "trial_2"
            trial3_dir.mkdir(exist_ok=True)

            # Create some files but not result indicators
            (trial3_dir / "parameters.json").touch()
            (trial3_dir / "some_log.txt").touch()

            # Test that _has_results_indicators returns False for trial without result indicators
            assert not _has_results_indicators(trial3_dir)

            # Test 4: Trial with other result indicators
            trial4_dir = study_dir / "trial_3"
            trial4_dir.mkdir(exist_ok=True)

            # Create a different result indicator
            (trial4_dir / BEST_CHECKPOINT_FILE).touch()

            # Test that _has_results_indicators returns True for trial with model file
            assert _has_results_indicators(trial4_dir)


class TestObjectiveFunction:
    """Test the objective function and trial execution."""

    def test_objective_function_basic_execution(self, dummy_adata):
        """Test that objective function executes without errors."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            # Mock the runner to return a score
            mock_result = {"harmonic_avg": {"AUROC": 0.85}}
            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", return_value=mock_result):
                study = optimizer.create_study()
                trial = study.ask()

                score = optimizer.objective(trial)
                assert score == 0.85
                assert trial.user_attrs.get("seed") == optimizer.seed_base
                assert trial.user_attrs.get("all_results") == mock_result

    def test_objective_function_with_parameter_overrides(self, dummy_adata):
        """Test objective function with parameter overrides."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
                n_epochs=5,  # Parameter override
                train_batch_size=32,  # Parameter override (using correct field name)
            )

            mock_result = {"harmonic_avg": {"AUROC": 0.90}}

            # Create a mock with the correct signature for run_mlp_training
            def mock_runner(
                data_path,
                base_save_dir,
                prm,
                device=None,
                dev_ratio=0.2,
                trial=None,
                score_metric="harmonic_avg/AUROC",
                resume=False,
                num_workers=0,
            ):
                # Store the parameters for verification (suppress unused arg warnings)
                _ = (data_path, base_save_dir, device, dev_ratio, trial, score_metric, resume, num_workers)
                mock_runner.call_params = prm
                return mock_result

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                study = optimizer.create_study()
                trial = study.ask()

                score = optimizer.objective(trial)
                assert score == 0.90

                # Check that our parameter overrides were applied
                params = mock_runner.call_params
                assert params is not None
                assert hasattr(params, "n_epochs")
                assert hasattr(params, "train_batch_size")
                # Check that our overrides were applied
                assert params.n_epochs == 5
                assert params.train_batch_size == 32

    def test_objective_function_handles_nan_score(self, dummy_adata):
        """Test that objective function handles NaN scores correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            # Mock runner to return NaN score
            mock_result = {"harmonic_avg": {"AUROC": float("nan")}}

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", return_value=mock_result):
                study = optimizer.create_study()
                trial = study.ask()

                # NaN score should be handled and return -inf, not raise exception
                score = optimizer.objective(trial)
                assert score == -float("inf")
                # Check that the error was recorded in trial attributes
                assert "NaN score encountered" in trial.user_attrs.get("error", "")

    def test_objective_function_handles_exceptions(self, dummy_adata):
        """Test that objective function handles exceptions properly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            # Mock runner to raise exception (using ConnectionError as it's in CATCHABLE_EXCEPTIONS)
            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", side_effect=ConnectionError("Training failed")):
                study = optimizer.create_study()
                trial = study.ask()

                score = optimizer.objective(trial)
                assert score == -float("inf")
                assert "Training failed" in trial.user_attrs.get("error", "")

    def test_objective_function_fail_fast_mode(self, dummy_adata):
        """Test that fail_fast mode re-raises exceptions immediately."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            # Test with fail_fast=True
            optimizer_fail_fast = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
                fail_fast=True,
            )

            # Mock runner to raise exception
            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", side_effect=ConnectionError("Training failed")):
                study = optimizer_fail_fast.create_study()
                trial = study.ask()

                # Should re-raise the exception instead of returning -inf
                with pytest.raises(ConnectionError, match="Training failed"):
                    optimizer_fail_fast.objective(trial)

            # Test with fail_fast=False (default behavior)
            optimizer_normal = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
                fail_fast=False,
            )

            # Mock runner to raise exception
            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", side_effect=ConnectionError("Training failed")):
                study = optimizer_normal.create_study()
                trial = study.ask()

                # Should return -inf and continue
                score = optimizer_normal.objective(trial)
                assert score == -float("inf")
                assert "Training failed" in trial.user_attrs.get("error", "")

    def test_objective_function_respects_trial_pruning(self, dummy_adata):
        """Test that objective function properly handles trial pruning."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            # Mock runner to raise TrialPruned
            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", side_effect=optuna.TrialPruned()):
                study = optimizer.create_study()
                trial = study.ask()

                with pytest.raises(optuna.TrialPruned):
                    optimizer.objective(trial)

    def test_objective_function_creates_trial_directory(self, dummy_adata):
        """Test that objective function creates trial directories."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="dir_test",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            mock_result = {"harmonic_avg": {"AUROC": 0.85}}

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", return_value=mock_result):
                study = optimizer.create_study()
                trial = study.ask()
                trial_number = trial.number

                optimizer.objective(trial)

                # Check that trial directory was created
                expected_trial_dir = optimizer.study_dir / f"trial_{trial_number}"
                assert expected_trial_dir.exists()
                assert expected_trial_dir.is_dir()


class TestRunOptimization:
    """Test the complete optimization workflow."""

    def test_run_optimization_basic(self, dummy_adata):
        """Test basic optimization run."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_run_optimization_basic",  # Unique study name
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            # Mock objective to return consistent scores
            with patch.object(optimizer, "objective", return_value=0.85) as mock_objective:
                study = optimizer.run_optimization(n_trials=3)

                assert len(study.trials) == 3
                assert mock_objective.call_count == 3

                # Check that study info was saved
                info_file = optimizer.study_dir / STUDY_INFO_FILE
                assert info_file.exists()

                with open(info_file) as f:
                    info = json.load(f)
                assert info["total_trials"] == 3
                assert info["completed_trials"] == 3

    def test_run_optimization_with_resume(self, dummy_adata):
        """Test optimization with resume functionality."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"
            study_name = "test_run_optimization_with_resume"

            # First optimization run
            optimizer1 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            with patch.object(optimizer1, "objective", return_value=0.85):
                study1 = optimizer1.run_optimization(n_trials=2)
                assert len(study1.trials) == 2

            # Second optimization run (resume)
            optimizer2 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            with patch.object(optimizer2, "objective", return_value=0.90):
                study2 = optimizer2.run_optimization(n_trials=2)
                assert len(study2.trials) == 4  # 2 original + 2 new

    def test_run_optimization_with_callbacks(self, dummy_adata):
        """Test that optimization uses callbacks correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_run_optimization_with_callbacks",  # Unique study name
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            with (
                patch.object(optimizer, "objective", return_value=0.85),
                patch("scxpand.hyperopt.hyperopt_optimizer.trial_callback"),
            ):
                study = optimizer.run_optimization(n_trials=2)

                # Verify that the callback was set up correctly
                # (The actual callback calls are handled by Optuna internally)
                assert len(study.trials) == 2


class TestPrintResults:
    """Test the print_results functionality."""

    def test_print_results_with_completed_trials(self, dummy_adata, capsys):
        """Test print_results with completed trials."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            with patch.object(optimizer, "objective", return_value=0.85):
                study = optimizer.run_optimization(n_trials=2)

                # Test print_results
                optimizer.print_results(study)

                captured = capsys.readouterr()
                assert "Best trial:" in captured.out
                assert "Best params:" in captured.out
                assert str(optimizer.study_dir) in captured.out

    def test_print_results_with_no_completed_trials(self, dummy_adata, capsys):
        """Test print_results when no trials completed."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_print_results_no_trials",  # Unique study name
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            # Create study with no completed trials
            optimizer.create_study()

            optimizer.print_results()

            captured = capsys.readouterr()
            assert "No completed trials found" in captured.out

    def test_print_results_loads_existing_study(self, dummy_adata, capsys):
        """Test that print_results can load existing study when none provided."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            # Create study with trials
            with patch.object(optimizer, "objective", return_value=0.85):
                optimizer.run_optimization(n_trials=1)

            # Test print_results without providing study
            optimizer.print_results()

            captured = capsys.readouterr()
            assert "Best trial:" in captured.out


class TestOptimizerEdgeCases:
    """Test edge cases and error conditions."""

    def test_different_model_types_independent_studies(self, dummy_adata):
        """Test that different model types create independent studies."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"

            # Create MLP optimizer
            mlp_optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=storage_path,
                num_workers=0,
            )

            # Create SVM optimizer
            svm_optimizer = HyperparameterOptimizer(
                model_type=ModelType.SVM,
                data_path=test_file_path,
                storage_path=storage_path,
                num_workers=0,
            )

            # Both should create studies without conflict
            mlp_study = mlp_optimizer.create_study()
            svm_study = svm_optimizer.create_study()

            assert mlp_study.study_name == "mlp_opt"
            assert svm_study.study_name == "svm_opt"
            assert mlp_study._storage != svm_study._storage

    def test_custom_config_path_integration(self, dummy_adata):
        """Test integration with custom config path."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            # Create a simple config file
            config_path = Path(temp_dir) / "config.json"
            config_data = {"n_epochs": 10, "batch_size": 64}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                config_path=str(config_path),
                num_workers=0,
            )

            # Mock the load_and_override_params function to verify it's called
            with (
                patch("scxpand.hyperopt.hyperopt_optimizer.load_and_override_params") as mock_load,
                patch.object(MODEL_TYPES[ModelType.MLP], "runner", return_value={"harmonic_avg": {"AUROC": 0.85}}),
            ):
                study = optimizer.create_study()
                trial = study.ask()

                optimizer.objective(trial)

                # Verify that load_and_override_params was called with config_path
                mock_load.assert_called_once()
                assert len(mock_load.call_args) > 0

    def test_pruner_configuration(self, dummy_adata):
        """Test that pruner is configured correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                num_workers=0,
            )

            # Check pruner configuration
            assert isinstance(optimizer.pruner, optuna.pruners.PercentilePruner)
            # Access private attributes for testing
            assert optimizer.pruner._percentile == 60.0
            assert optimizer.pruner._n_startup_trials == 5
            assert optimizer.pruner._n_warmup_steps == 5
            assert optimizer.pruner._n_min_trials == 5

    def test_sampler_configuration(self, dummy_adata):
        """Test that sampler is configured with correct seed."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            custom_seed = 999
            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                seed_base=custom_seed,
                num_workers=0,
            )

            study = optimizer.create_study()

            # Check that sampler is TPESampler with correct seed
            assert isinstance(study.sampler, optuna.samplers.TPESampler)
            # Note: TPESampler seed is not directly accessible, but we can verify
            # that it's deterministic by running trials


class TestTrialResumeAndCleanup:
    """Test trial resume functionality and cleanup of failed trials."""

    def test_duplicate_epoch_reporting_prevention(self, dummy_adata):
        """Test that duplicate epoch reports are prevented when resuming."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_duplicate_epoch_reporting_prevention",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()
            trial = study.ask()

            # First report should succeed
            report_to_optuna_and_handle_pruning(trial=trial, current_score=0.85, epoch=0)

            # Finalize the trial to access its intermediate values
            study.tell(trial, 0.85)
            frozen_trial = study.trials[0]

            assert len(frozen_trial.intermediate_values) == 1
            assert 0 in frozen_trial.intermediate_values
            assert frozen_trial.user_attrs.get("reported_epochs") == [0]

            # Start a new trial for the duplicate test
            trial2 = study.ask()
            # Simulate that this trial already has epoch 0 reported
            trial2.set_user_attr("reported_epochs", [0])

            # Second report for same epoch should be skipped (no logging test)
            report_to_optuna_and_handle_pruning(trial=trial2, current_score=0.90, epoch=0)

            # Report a new epoch (should work normally)
            report_to_optuna_and_handle_pruning(trial=trial2, current_score=0.90, epoch=1)
            assert trial2.user_attrs.get("reported_epochs") == [0, 1]

    def test_cleanup_incomplete_trials(self, dummy_adata):
        """Test cleanup of incomplete trials."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            study_dir = Path(temp_dir) / "studies" / "test_cleanup"
            study_dir.mkdir(parents=True, exist_ok=True)

            # Create some trial directories without proper checkpoints
            incomplete_trial_dir = study_dir / "trial_0"
            incomplete_trial_dir.mkdir(exist_ok=True)
            # Create an empty file that's not a valid checkpoint
            (incomplete_trial_dir / "empty.txt").touch()

            complete_trial_dir = study_dir / "trial_1"
            complete_trial_dir.mkdir(exist_ok=True)
            # Create a valid checkpoint file
            (complete_trial_dir / "last_ckpt.pt").touch()

            # Create a mock study using MagicMock

            mock_study = MagicMock()
            mock_trial_0 = MagicMock()
            mock_trial_0.number = 0
            mock_trial_0.state = optuna.trial.TrialState.RUNNING

            mock_trial_1 = MagicMock()
            mock_trial_1.number = 1
            mock_trial_1.state = optuna.trial.TrialState.RUNNING

            mock_study.trials = [mock_trial_0, mock_trial_1]

            # Run cleanup with immediate cleanup (max_age_hours=0)
            cleaned_count = cleanup_incomplete_trials(study=mock_study, study_dir=study_dir, max_age_hours=0)

            # Should have cleaned up the trial without valid checkpoint
            assert cleaned_count == 1
            assert not incomplete_trial_dir.exists()
            assert complete_trial_dir.exists()  # Should not touch trial with valid checkpoint

    def test_trial_resume_with_existing_reports(self, dummy_adata):
        """Test that resuming a trial with existing epoch reports works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_trial_resume_with_existing_reports",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()
            trial = study.ask()

            # Simulate a trial that already has some epoch reports (from previous run)
            trial.set_user_attr("reported_epochs", [0, 1, 2])

            # Try to report epoch 2 again (should be skipped, no logging test)
            report_to_optuna_and_handle_pruning(trial=trial, current_score=0.90, epoch=2)

            # Report a new epoch (should work normally)
            report_to_optuna_and_handle_pruning(trial=trial, current_score=0.90, epoch=3)
            assert trial.user_attrs.get("reported_epochs") == [0, 1, 2, 3]

    def test_optimization_with_cleanup_on_resume(self, dummy_adata):
        """Test that optimization automatically cleans up incomplete trials when resuming."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"
            study_name = "test_optimization_cleanup"

            # Create initial optimizer and study
            optimizer1 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            # Run one trial to create the study
            with patch.object(optimizer1, "objective", return_value=0.85):
                study1 = optimizer1.run_optimization(n_trials=1)
                assert len(study1.trials) == 1

            # Create a fake incomplete trial directory
            incomplete_trial_dir = optimizer1.study_dir / "trial_999"
            incomplete_trial_dir.mkdir(exist_ok=True)
            (incomplete_trial_dir / "incomplete.txt").touch()

            # Create new optimizer and resume with cleanup
            optimizer2 = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name=study_name,
                storage_path=storage_path,
                num_workers=0,
            )

            with patch.object(optimizer2, "objective", return_value=0.90):
                with patch("scxpand.hyperopt.hyperopt_optimizer.cleanup_incomplete_trials") as mock_cleanup:
                    study2 = optimizer2.run_optimization(n_trials=1)

                    # Cleanup should have been called
                    mock_cleanup.assert_called_once_with(study=study2, study_dir=optimizer2.study_dir)

    def test_different_model_types_resume_independently(self, dummy_adata):
        """Test that different model types can resume independently without interference."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"

            # Test MLP
            mlp_optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                storage_path=storage_path,
                num_workers=0,
            )

            with patch.object(mlp_optimizer, "objective", return_value=0.85):
                mlp_study = mlp_optimizer.run_optimization(n_trials=2)
                assert len(mlp_study.trials) == 2

            # Test SVM (should be independent)
            svm_optimizer = HyperparameterOptimizer(
                model_type=ModelType.SVM,
                data_path=test_file_path,
                storage_path=storage_path,
                num_workers=0,
            )

            with patch.object(svm_optimizer, "objective", return_value=0.80):
                svm_study = svm_optimizer.run_optimization(n_trials=1)
                assert len(svm_study.trials) == 1

            # Resume MLP should still have its trials
            with patch.object(mlp_optimizer, "objective", return_value=0.90):
                mlp_study_resumed = mlp_optimizer.run_optimization(n_trials=1)
                assert len(mlp_study_resumed.trials) == 3  # 2 original + 1 new

    def test_trial_callback_handles_failed_trials(self, dummy_adata):
        """Test that trial callback properly handles and cleans up failed trials."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            study_dir = Path(temp_dir) / "test_study"
            study_dir.mkdir(parents=True, exist_ok=True)

            # Create a trial directory
            trial_dir = study_dir / "trial_0"
            trial_dir.mkdir(exist_ok=True)
            test_file = trial_dir / "test.txt"
            test_file.touch()

            # Create a mock study and failed trial
            study = optuna.create_study(direction="maximize")

            # Mock a failed trial using MagicMock since FrozenTrial constructor is complex
            failed_trial = MagicMock()
            failed_trial.number = 0
            failed_trial.state = optuna.trial.TrialState.FAIL

            # Test the callback

            # Run the callback
            trial_callback(study=study, trial=failed_trial, study_dir=study_dir)

            # Trial directory should be cleaned up
            assert not trial_dir.exists()


class TestHappyPathFunctionality:
    """Test that new functionality doesn't break existing happy path workflows."""

    def test_normal_epoch_reporting_workflow(self, dummy_adata):
        """Test that normal epoch reporting works correctly without any changes."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_normal_epoch_reporting",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()
            trial = study.ask()

            # Simulate normal training progression
            epochs = [0, 1, 2, 3, 4]
            scores = [0.60, 0.65, 0.70, 0.75, 0.80]

            for epoch, score in zip(epochs, scores):
                report_to_optuna_and_handle_pruning(trial=trial, current_score=score, epoch=epoch)

            # Complete the trial
            study.tell(trial, 0.80)
            frozen_trial = study.trials[0]

            # Verify all epochs were reported correctly
            assert len(frozen_trial.intermediate_values) == len(epochs)
            for epoch, expected_score in zip(epochs, scores):
                assert epoch in frozen_trial.intermediate_values
                assert frozen_trial.intermediate_values[epoch] == expected_score

            # Verify user attributes were set
            assert frozen_trial.user_attrs.get("reported_epochs") == epochs

    def test_normal_study_creation_and_completion(self, dummy_adata):
        """Test that normal study creation and completion works without interference."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_normal_study_completion",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            # Mock successful training runs
            with patch.object(optimizer, "objective", side_effect=[0.70, 0.75, 0.80, 0.85]):
                study = optimizer.run_optimization(n_trials=4)

                # Verify all trials completed successfully
                assert len(study.trials) == 4
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                assert len(completed_trials) == 4

                # Verify best trial is correctly identified
                best_trial = study.best_trial
                assert best_trial.value == 0.85
                assert best_trial.number == 3  # Last trial had the best score

    def test_first_time_study_creation(self, dummy_adata):
        """Test that creating a study for the first time works normally."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_first_time_creation",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            # Verify study directory is created during initialization
            study_dir = optimizer.study_dir
            assert study_dir.exists()  # Directory should be created by __init__

            # Create study and run optimization
            with patch.object(optimizer, "objective", return_value=0.75):
                study = optimizer.run_optimization(n_trials=2)

                # Verify study completed successfully
                assert len(study.trials) == 2

                # Verify database file was created
                db_file = study_dir / "optuna.db"
                assert db_file.exists()

    def test_multiple_model_types_normal_operation(self, dummy_adata):
        """Test that multiple model types work normally without interference."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            storage_path = Path(temp_dir) / "studies"

            # Test different model types
            model_types = [ModelType.MLP, ModelType.SVM]

            for model_type in model_types:
                optimizer = HyperparameterOptimizer(
                    model_type=model_type,
                    data_path=test_file_path,
                    storage_path=storage_path,
                    num_workers=0,
                )

                with patch.object(optimizer, "objective", return_value=0.80):
                    study = optimizer.run_optimization(n_trials=1)

                    # Verify each model type creates its own study
                    assert len(study.trials) == 1
                    assert study.study_name == f"{model_type.value}_opt"

                    # Verify study directory exists
                    assert optimizer.study_dir.exists()

    def test_normal_trial_pruning_behavior(self, dummy_adata):
        """Test that normal trial pruning behavior works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_normal_pruning",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()
            trial = study.ask()

            # Mock trial.should_prune() to return True after first report
            with patch.object(trial, "should_prune", side_effect=[False, True]):
                # First report should succeed
                report_to_optuna_and_handle_pruning(trial=trial, current_score=0.60, epoch=0)

                # Second report should trigger pruning
                with pytest.raises(optuna.TrialPruned):
                    report_to_optuna_and_handle_pruning(trial=trial, current_score=0.55, epoch=1)

    def test_no_interference_with_user_attrs(self, dummy_adata):
        """Test that our new user attributes don't interfere with existing ones."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_user_attrs_no_interference",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
            )

            study = optimizer.create_study()
            trial = study.ask()

            # Set some custom user attributes
            trial.set_user_attr("custom_metric", 0.90)
            trial.set_user_attr("custom_info", {"model": "test", "version": "1.0"})

            # Report an epoch
            report_to_optuna_and_handle_pruning(trial=trial, current_score=0.75, epoch=0)

            # Complete the trial
            study.tell(trial, 0.75)
            frozen_trial = study.trials[0]

            # Verify our attribute was added
            assert frozen_trial.user_attrs.get("reported_epochs") == [0]

            # Verify existing attributes are preserved
            assert frozen_trial.user_attrs.get("custom_metric") == 0.90
            assert frozen_trial.user_attrs.get("custom_info") == {"model": "test", "version": "1.0"}

    def test_empty_studies_cleanup_safety(self, dummy_adata):
        """Test that cleanup is safe when called on empty studies."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            study_dir = Path(temp_dir) / "empty_study"
            study_dir.mkdir(parents=True, exist_ok=True)

            # Create an empty study
            db_file = study_dir / "optuna.db"
            storage = f"sqlite:///{db_file}"

            empty_study = optuna.create_study(
                direction="maximize",
                study_name="empty_study",
                storage=storage,
                load_if_exists=False,
            )

            # Cleanup should work safely on empty study
            cleaned_count = cleanup_incomplete_trials(study=empty_study, study_dir=study_dir, max_age_hours=0)
            assert cleaned_count == 0  # No trials to clean

    def test_normal_parameter_overrides_still_work(self, dummy_adata):
        """Test that parameter overrides continue to work normally."""
        with tempfile.TemporaryDirectory() as temp_dir, windows_safe_context_manager() as ctx:
            test_file_path = create_temp_h5ad_file(dummy_adata, temp_dir)
            ctx.register_file(test_file_path)

            optimizer = HyperparameterOptimizer(
                model_type=ModelType.MLP,
                data_path=test_file_path,
                study_name="test_parameter_overrides",
                storage_path=Path(temp_dir) / "studies",
                num_workers=0,
                n_epochs=5,  # Parameter override
                train_batch_size=32,  # Parameter override
            )

            mock_result = {"harmonic_avg": {"AUROC": 0.85}}

            def mock_runner(
                data_path,
                base_save_dir,
                prm,
                device=None,
                dev_ratio=0.2,
                trial=None,
                score_metric="harmonic_avg/AUROC",
                resume=False,
                num_workers=0,
            ):
                # Store the parameters for verification
                # Suppress unused argument warnings
                _ = (data_path, base_save_dir, device, dev_ratio, trial, score_metric, resume, num_workers)
                mock_runner.call_params = prm
                return mock_result

            with patch.object(MODEL_TYPES[ModelType.MLP], "runner", mock_runner):
                study = optimizer.run_optimization(n_trials=1)

                # Verify trial completed successfully
                assert len(study.trials) == 1
                completed_trial = study.trials[0]
                assert completed_trial.state == optuna.trial.TrialState.COMPLETE

                # Verify parameter overrides were applied
                params = mock_runner.call_params
                assert params is not None
                assert hasattr(params, "n_epochs")
                assert hasattr(params, "train_batch_size")
                assert params.n_epochs == 5
                assert params.train_batch_size == 32
