"""Hyperparameter optimization using Optuna with robust trial management."""

import inspect
import math

from functools import partial
from pathlib import Path

import optuna

from scxpand.core.model_types import MODEL_TYPES
from scxpand.hyperopt.hyperopt_utils import (
    _has_results_indicators,
    cleanup_incomplete_trials,
    save_study_info,
    trial_callback,
)
from scxpand.util.classes import ModelType, ensure_model_type
from scxpand.util.general_util import load_and_override_params, set_seed
from scxpand.util.logger import get_logger
from scxpand.util.metrics import get_score_from_nested_dict
from scxpand.util.model_constants import STUDY_INFO_FILE


logger = get_logger()

# Constants for cleaner code
DEFAULT_STORAGE_PATH = "results/optuna_studies"
DEFAULT_SCORE_METRIC = "harmonic_avg/AUROC"
DEFAULT_SEED = 42
FAILED_TRIAL_SCORE = -float("inf")

# Error types that should cause trial failure vs. continuation
CRITICAL_ERRORS = (
    MemoryError,  # Out of memory - system level issue
    RuntimeError,  # CUDA errors, training failures - critical hardware/model issues
    ValueError,  # Invalid parameter values - model misconfiguration
    FileNotFoundError,  # Missing data/model files - setup issue
)

CATCHABLE_EXCEPTIONS = (
    # Transient/recoverable errors that shouldn't stop optimization
    ConnectionError,  # Network issues
    TimeoutError,  # Timeouts
    ImportError,  # Missing optional dependencies
    OSError,  # File system issues
)


class HyperparameterOptimizer:
    """Robust hyperparameter optimizer using Optuna with enhanced trial management.

    Features:
    - Automatic cleanup of incomplete trials
    - Proper exception handling and categorization
    - Resume capability with duplicate prevention
    - Comprehensive logging and monitoring
    """

    def __init__(
        self,
        model_type: ModelType | str,
        data_path: str | Path,
        study_name: str | None = None,
        storage_path: str | Path = DEFAULT_STORAGE_PATH,
        score_metric: str = DEFAULT_SCORE_METRIC,
        seed_base: int = DEFAULT_SEED,
        num_workers: int = 0,
        config_path: str | None = None,
        resume: bool = True,
        fail_fast: bool = False,
        **param_overrides,
    ):
        """Initialize the hyperparameter optimizer.

        Args:
            model_type: Type of model to optimize (MLP, SVM, etc.).
            data_path: Path to the training data file.
            study_name: Name for the Optuna study (auto-generated if None).
            storage_path: Directory to store study results.
            score_metric: Metric to optimize (e.g., "harmonic_avg/AUROC").
            seed_base: Base seed for reproducibility.
            num_workers: Number of parallel workers (0 for single-threaded).
            config_path: Path to configuration file for parameter overrides.
            resume: Whether to resume existing study (False = start fresh).
            fail_fast: Whether to fail immediately on any exception (for testing).
            **param_overrides: Additional parameter overrides.
        """
        self.model_type = self._validate_model_type(model_type)
        self.data_path = self._validate_data_path(data_path)
        self.study_name = study_name or f"{self.model_type.value}_opt"
        self.storage_path = Path(storage_path)
        self.score_metric = score_metric
        self.seed_base = seed_base
        self.num_workers = num_workers
        self.config_path = config_path
        self.resume = resume
        self.fail_fast = fail_fast
        self.param_overrides = param_overrides

        # Setup study infrastructure
        self.study_dir, self.storage = self._setup_study_infrastructure()
        self.pruner = self._create_pruner()

    def _validate_model_type(self, model_type: ModelType | str) -> ModelType:
        """Validate and convert model type."""
        model_type = ensure_model_type(model_type)

        if model_type not in MODEL_TYPES:
            valid_types = [m.value for m in ModelType]
            raise ValueError(f"model_type must be one of {valid_types}")

        return model_type

    def _validate_data_path(self, data_path: str | Path) -> Path:
        """Validate that data path exists."""
        data_path = Path(data_path)

        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")

        return data_path

    def _setup_study_infrastructure(self) -> tuple[Path, str]:
        """Setup study directory and storage configuration."""
        study_dir = self.storage_path / self.study_name
        study_dir.mkdir(parents=True, exist_ok=True)

        db_file = study_dir / "optuna.db"
        storage = f"sqlite:///{db_file}"

        return study_dir, storage

    def _create_pruner(self) -> optuna.pruners.PercentilePruner:
        """Create and configure the pruner for trial optimization."""
        return optuna.pruners.PercentilePruner(
            percentile=60.0,  # Keep top 40% of trials
            n_startup_trials=5,
            n_warmup_steps=5,
            n_min_trials=5,
        )

    def create_study(self) -> optuna.Study:
        """Create or load an Optuna study based on the resume setting.

        Returns:
            The Optuna study object.

        Raises:
            ValueError: If study exists but resume=False.
        """
        self._handle_existing_study()

        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.resume,
            sampler=optuna.samplers.TPESampler(seed=self.seed_base),
            pruner=self.pruner,
        )

        # Set study directory for callback access
        if not study.user_attrs.get("study_dir"):
            study.user_attrs["study_dir"] = str(self.study_dir)

        return study

    def _handle_existing_study(self) -> None:
        """Handle existing study based on resume flag."""
        if not self.resume and self.study_dir.exists():
            # Check if study has any meaningful content
            db_file = self.study_dir / "optuna.db"
            trial_dirs = []

            if db_file.exists():
                # Check if there are any trial directories
                trial_dirs = [d for d in self.study_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]

            if len(trial_dirs) > 0:
                raise ValueError(
                    f"Study '{self.study_name}' already exists at {self.study_dir} with {len(trial_dirs)} trial(s). "
                    f"To start fresh, you can:\n"
                    f"  1. Use resume=True to continue the existing study, OR\n"
                    f"  2. Delete the study directory and run again with resume=False, OR\n"
                    f"  3. Use a different --study_name or --storage_path to avoid conflicts"
                )
            if db_file.exists():
                # Has database but no trials - warn and continue
                logger.warning(f"Found existing study database with no trials at {self.study_dir}. Continuing...")
            # If directory exists but has no database or trials, we can proceed (empty directory)

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna trials.

        Args:
            trial: The Optuna trial object.

        Returns:
            The trial score (higher is better) or -inf if failed.
        """
        logger.info(f"Running trial {trial.number}")

        # Setup trial with reproducible seed
        trial_seed = self.seed_base
        trial.set_user_attr("seed", trial_seed)
        set_seed(trial_seed)

        # Prepare parameters and trial directory
        params = self._prepare_trial_parameters(trial, trial_seed)
        save_dir = self._create_trial_directory(trial)

        # Execute the trial
        try:
            return self._execute_trial(trial, params, save_dir)
        except optuna.TrialPruned:
            raise  # Let Optuna handle pruning properly
        except (KeyboardInterrupt, SystemExit):
            raise  # Allow user to interrupt optimization
        except Exception as e:
            return self._handle_trial_exception(trial, e)

    def _prepare_trial_parameters(self, trial: optuna.Trial, trial_seed: int) -> object:
        """Prepare parameters for the trial."""
        spec = MODEL_TYPES[self.model_type]
        param_dict = spec.config_func(trial)
        param_dict["random_seed"] = trial_seed

        # Apply parameter overrides if provided
        if self.param_overrides:
            param_dict.update(self.param_overrides)
            logger.info(f"Applied parameter overrides: {self.param_overrides}")

        # Create parameters object
        if self.config_path or self.param_overrides:
            return load_and_override_params(
                param_class=spec.param_class,
                config_path=self.config_path,
                **param_dict,
            )
        else:
            return spec.param_class(**param_dict)

    def _create_trial_directory(self, trial: optuna.Trial) -> Path:
        """Create and return the trial directory."""
        save_dir = self.study_dir / f"trial_{trial.number}"
        save_dir.mkdir(exist_ok=True)
        return save_dir

    def _execute_trial(self, trial: optuna.Trial, params: object, save_dir: Path) -> float:
        """Execute the trial and return the score."""
        spec = MODEL_TYPES[self.model_type]

        # Determine if this specific trial should resume based on existing checkpoint files
        # New trials should always start fresh, regardless of the global resume setting
        trial_should_resume = self.resume and _has_results_indicators(save_dir)

        if trial_should_resume:
            logger.debug(f"Trial {trial.number} will resume from checkpoints")
        else:
            logger.debug(f"Trial {trial.number} will start fresh")

        # Prepare runner arguments
        runner_args = {
            "data_path": self.data_path,
            "prm": params,
            "base_save_dir": save_dir,
            "trial": trial,
            "score_metric": self.score_metric,
            "num_workers": self.num_workers,
            "resume": trial_should_resume,  # Pass trial-specific resume flag to runner
        }

        # Filter arguments based on runner signature to avoid passing invalid parameters
        try:
            sig = inspect.signature(spec.runner)
            filtered_args = {k: v for k, v in runner_args.items() if k in sig.parameters}
        except Exception as e:
            logger.warning(f"Could not inspect runner signature: {e}")
            # Fall back to passing all arguments
            filtered_args = runner_args

        result = spec.runner(**filtered_args)
        return self._process_trial_result(trial, result)

    def _process_trial_result(self, trial: optuna.Trial, result: dict) -> float:
        """Process the trial result and extract the score."""
        if not isinstance(result, dict):
            return FAILED_TRIAL_SCORE

        trial.set_user_attr("all_results", result)
        score = get_score_from_nested_dict(nested_metrics_dict=result, metric_name=self.score_metric)

        if score is not None and isinstance(score, float) and math.isnan(score):
            logger.warning(f"Trial {trial.number} returned NaN score, marking as failed.")
            trial.set_user_attr("error", "NaN score encountered.")
            return FAILED_TRIAL_SCORE

        return score if score is not None else FAILED_TRIAL_SCORE

    def _handle_trial_exception(self, trial: optuna.Trial, exception: Exception) -> float:
        """Handle trial exceptions and decide whether to fail or continue."""
        trial.set_user_attr("error", str(exception))
        logger.exception(f"Trial {trial.number} failed with error: {exception}")

        # In fail_fast mode, re-raise all exceptions immediately for debugging
        if self.fail_fast:
            raise exception

        # Critical errors should fail the trial properly
        if isinstance(exception, CRITICAL_ERRORS):
            raise exception

        # Other errors return -inf to continue optimization
        return FAILED_TRIAL_SCORE

    def _resume_existing_trials(self, study: optuna.Study) -> int:
        """Resume existing trials that have valid checkpoints and are in resumable states.

        Args:
            study: The Optuna study object.

        Returns:
            Number of trials that were resumed.
        """
        # Consider trials in states that can be resumed
        # RUNNING: Currently running trials that might have been interrupted
        # FAIL: Failed trials that might have checkpoints from before failure
        resumable_states = ["RUNNING", "FAIL"]
        resumable_trials = [t for t in study.trials if t.state.name in resumable_states]

        if not resumable_trials:
            logger.debug("No resumable trials found")
            return 0

        logger.info(
            f"Found {len(resumable_trials)} trial(s) in resumable states: {[t.state.name for t in resumable_trials]}"
        )
        resumed_count = 0

        for trial in resumable_trials:
            trial_dir = self.study_dir / f"trial_{trial.number}"

            # Only resume trials that have checkpoints
            if trial_dir.exists() and _has_results_indicators(trial_dir):
                logger.info(f"Resuming {trial.state.name} trial {trial.number} from checkpoints")

                try:
                    # Resume this specific trial by calling objective function
                    score = self.objective(trial)

                    if score != FAILED_TRIAL_SCORE:
                        logger.info(f"Successfully resumed trial {trial.number} with score {score:.4f}")
                        resumed_count += 1

                except Exception as e:
                    logger.error(f"Failed to resume trial {trial.number}: {e}")

        return resumed_count

    def run_optimization(self, n_trials: int = 100) -> optuna.Study:
        """Run the hyperparameter optimization.

        Args:
            n_trials: Number of trials to run.

        Returns:
            The completed Optuna study.
        """
        optuna.logging.set_verbosity(optuna.logging.WARN)
        study = self.create_study()

        # Clean up incomplete trials if resuming
        if self.resume:
            cleanup_incomplete_trials(study=study, study_dir=self.study_dir)

            # Resume existing trials with checkpoints first
            resumed_trials = self._resume_existing_trials(study)
            if resumed_trials > 0:
                logger.info(f"Resumed {resumed_trials} existing trial(s) from checkpoints")
                # Reduce n_trials by the number we just resumed
                n_trials = max(0, n_trials - resumed_trials)

        # Run optimization for any remaining trials
        if n_trials > 0:
            study.optimize(
                func=self.objective,
                n_trials=n_trials,
                callbacks=[partial(trial_callback, study_dir=self.study_dir)],
                catch=CATCHABLE_EXCEPTIONS,  # Let Optuna handle these gracefully
            )

        # Save comprehensive study information
        save_study_info(study=study, study_dir=self.study_dir, score_metric=self.score_metric)

        return study

    def print_results(self, study: optuna.Study | None = None) -> None:
        """Print optimization results.

        Args:
            study: The study to print results for (loads existing if None).
        """
        if study is None:
            # Temporarily set resume=True to load existing study for results
            original_resume = self.resume
            self.resume = True
            study = self.create_study()
            self.resume = original_resume

        try:
            best_trial = study.best_trial
            print(f"Best trial: {best_trial.number}, value: {best_trial.value}")
            print(f"Best params: {best_trial.params}")
            print(f"See {self.study_dir}/{STUDY_INFO_FILE} and {self.study_dir}/trials.log for details.")
        except ValueError:
            print("No completed trials found")
