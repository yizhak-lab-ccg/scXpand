"""Utilities for Optuna study optimization and trial management."""

import json
import shutil

from datetime import datetime, timedelta
from pathlib import Path

import optuna

from scxpand.util.general_util import flatten_dict, format_float, get_utc_time
from scxpand.util.logger import get_logger


logger = get_logger()

# Constants for trial results detection (completed work)
from scxpand.util.model_constants import (
    BEST_CHECKPOINT_FILE,
    LAST_CHECKPOINT_FILE,
    RESULTS_DEV_FILE,
    RESULTS_TABLE_DEV_FILE,
    STUDY_INFO_FILE,
    SUMMARY_INFO_FILE,
)


RESULTS_INDICATORS = [
    LAST_CHECKPOINT_FILE,  # Training in progress (checkpoint saved)
    BEST_CHECKPOINT_FILE,  # Training completed at least one epoch
    SUMMARY_INFO_FILE,  # Results summary available
    RESULTS_DEV_FILE,  # Evaluation results
    RESULTS_TABLE_DEV_FILE,  # Detailed results table
]

DEFAULT_CLEANUP_AGE_HOURS = 0  # Immediate cleanup of incomplete trials


def save_study_info(study: optuna.Study, study_dir: Path, score_metric: str) -> None:
    """Save comprehensive study information to JSON file.

    Args:
        study: The Optuna study object.
        study_dir: Directory where study information will be saved.
        score_metric: The metric being optimized.
    """
    info = _build_study_info_dict(study, score_metric)
    _save_study_info_to_file(info, study_dir)


def _build_study_info_dict(study: optuna.Study, score_metric: str) -> dict:
    """Build the study information dictionary."""
    completed_trials = _get_trials_by_state(study, optuna.trial.TrialState.COMPLETE)
    pruned_trials = _get_trials_by_state(study, optuna.trial.TrialState.PRUNED)
    failed_trials = _get_trials_by_state(study, optuna.trial.TrialState.FAIL)

    info = {
        "timestamp": get_utc_time().isoformat(),
        "study_name": study.study_name,
        "metric_name": score_metric,
        "direction": study.direction.name,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "failed_trials": len(failed_trials),
        "best_value": None,
        "best_trial_number": None,
        "best_trial_params": None,
        "best_trial_results": None,
    }

    # Add best trial information if available
    if completed_trials:
        _add_best_trial_info(study, info)

    return info


def _get_trials_by_state(study: optuna.Study, state: optuna.trial.TrialState) -> list:
    """Get all trials with a specific state."""
    return [trial for trial in study.trials if trial.state == state]


def _add_best_trial_info(study: optuna.Study, info: dict) -> None:
    """Add best trial information to the study info dictionary."""
    try:
        best_trial = study.best_trial
        info.update(
            {
                "best_value": getattr(best_trial, "value", None),
                "best_trial_number": getattr(best_trial, "number", None),
                "best_trial_params": getattr(best_trial, "params", None),
                "best_trial_results": best_trial.user_attrs.get("all_results", None),
            }
        )
    except Exception:
        logger.exception("Error retrieving best trial information")


def _save_study_info_to_file(info: dict, study_dir: Path) -> None:
    """Save study information dictionary to JSON file."""
    info_path = study_dir / STUDY_INFO_FILE
    try:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, default=str)
    except Exception:
        logger.exception(f"Error saving study info to {info_path}")


def report_optuna_trial_result(trial: optuna.Trial | None, results: dict, score_metric: str) -> None:
    """Report a score to an Optuna trial and handle pruning.

    Args:
        trial: The Optuna trial object (or None).
        results: The nested results dictionary.
        score_metric: The metric key to report.
    """
    if trial is None:
        return

    results_flat = {f"{k1}/{k2}": v for k1, k2, v in flatten_dict(results)}

    if score_metric not in results_flat:
        logger.warning(f"Score metric {score_metric} not found in results: {list(results_flat.keys())}")
        return

    score_value = results_flat[score_metric]
    formatted_score = format_float(score_value)

    # Log and report the score
    _log_trial_score_report(trial, score_metric, formatted_score)
    trial.report(score_value, step=0)

    # Handle pruning if necessary
    if trial.should_prune():
        _handle_trial_pruning(trial, score_metric, formatted_score)


def _log_trial_score_report(trial: optuna.Trial, score_metric: str, formatted_score: str) -> None:
    """Log the trial score reporting."""
    message = f"Reporting score {score_metric}={formatted_score} to Optuna (trial {trial.number})"
    logger.info(message)
    print(f"[Optuna] Reporting score {score_metric}={formatted_score} for trial {trial.number}")


def _handle_trial_pruning(trial: optuna.Trial, score_metric: str, formatted_score: str) -> None:
    """Handle trial pruning if the trial should be pruned."""
    message = f"Trial {trial.number} would be pruned after reporting {score_metric}={formatted_score}"
    logger.info(message)
    print(f"[Optuna] {message}")


def log_trial_progress(trial: optuna.Trial, study_dir: Path) -> None:
    """Log trial progress to both logger and trials.log file.

    Args:
        trial: The Optuna trial object.
        study_dir: Path to the study directory for logging.
    """
    log_path = study_dir / "trials.log"

    # Build the log message
    state_name = optuna.trial.TrialState(trial.state).name
    trial_number = getattr(trial, "number", "?")
    trial_value = getattr(trial, "value", None)
    formatted_value = format_float(trial_value) if isinstance(trial_value, float) else trial_value

    base_message = f"Trial {trial_number}: state={state_name}, value={formatted_value}"
    detailed_message = f"{base_message}\nparams={trial.params}\nuser_attrs={trial.user_attrs}"

    # Log to logger and console
    logger.info(base_message)
    print("-" * 100)

    # Write to trials.log file
    _write_trial_log(log_path, detailed_message)


def _write_trial_log(log_path: Path, message: str) -> None:
    """Write trial information to log file."""
    try:
        with log_path.open("a") as f:
            f.write(f"{message}\n")
    except Exception as e:
        logger.warning(f"Could not write to {log_path}: {e}")


def cleanup_failed_trial(study_dir: Path, trial_number: int) -> None:
    """Clean up files and directories for a failed trial.

    This function only removes trial directories and files - it does not modify
    Optuna trial states. Trial state management should be handled by Optuna itself.

    Args:
        study_dir: The study directory path.
        trial_number: The trial number to clean up.
    """
    trial_dir = study_dir / f"trial_{trial_number}"

    if not trial_dir.exists():
        return

    try:
        shutil.rmtree(trial_dir)
        logger.info(f"Cleaned up failed trial directory: {trial_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up trial directory {trial_dir}: {e}")


def cleanup_incomplete_trials(
    study: optuna.Study, study_dir: Path, max_age_hours: int = DEFAULT_CLEANUP_AGE_HOURS
) -> int:
    """Clean up trial directories for RUNNING trials without results.

    Any RUNNING trial without results can be safely cleaned up since there are
    no concurrent processes that could be actively working on them.

    This function cleans up trial directories (not Optuna trial states) for trials that:
    1. Are in RUNNING state in Optuna database
    2. Have no valid results/checkpoint files (indicating incomplete execution)

    Note: This does not modify trial states in Optuna - it only cleans up orphaned directories.

    Args:
        study: The Optuna study object.
        study_dir: The study directory path.
        max_age_hours: Minimum age in hours before cleanup (0 = immediate cleanup).

    Returns:
        Number of trial directories cleaned up.
    """
    running_trials = _get_trials_by_state(study, optuna.trial.TrialState.RUNNING)
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

    cleaned_count = 0

    for trial in running_trials:
        trial_dir = study_dir / f"trial_{trial.number}"

        if _should_cleanup_trial_directory(trial_dir, cutoff_time) and _cleanup_single_trial_directory(
            study_dir, trial.number, trial_dir
        ):
            cleaned_count += 1

    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} incomplete RUNNING trial directory(ies)")

    return cleaned_count


def _should_cleanup_trial_directory(trial_dir: Path, cutoff_time: datetime) -> bool:
    """Check if a trial directory should be cleaned up."""
    # Only consider directories that exist
    if not trial_dir.exists():
        return False

    # Check minimum age if specified (0 = no age requirement)
    try:
        dir_mtime = datetime.fromtimestamp(trial_dir.stat().st_mtime)
        if dir_mtime > cutoff_time:
            return False  # Directory is too recent (respects min age)
    except OSError:
        return False  # Can't get directory stats, skip for safety

    # Clean up any RUNNING trial without results
    return not _has_results_indicators(trial_dir)


def _has_results_indicators(trial_dir: Path) -> bool:
    """Check if trial directory has any completed results."""
    return any((trial_dir / indicator).exists() for indicator in RESULTS_INDICATORS)


def _cleanup_single_trial_directory(study_dir: Path, trial_number: int, trial_dir: Path) -> bool:
    """Clean up a single trial directory and return True if successful."""
    try:
        age = datetime.now() - datetime.fromtimestamp(trial_dir.stat().st_mtime)
        logger.warning(f"Cleaning up incomplete RUNNING trial directory {trial_number} (age: {age}, no results found)")

        cleanup_failed_trial(study_dir, trial_number)
        return True

    except Exception as e:
        logger.warning(f"Failed to clean up trial directory {trial_number}: {e}")
        return False


def trial_callback(study: optuna.Study, trial: optuna.Trial, study_dir: Path) -> None:  # noqa: ARG001
    """Optuna callback for logging and handling different trial states.

    Args:
        study: The Optuna study object.
        trial: The Optuna trial object.
        study_dir: Path to the study directory for logging.
    """
    log_trial_progress(trial=trial, study_dir=study_dir)
    _handle_trial_completion(trial, study_dir)


def _handle_trial_completion(trial: optuna.Trial, study_dir: Path) -> None:
    """Handle different trial completion states."""
    state = trial.state
    trial_number = trial.number

    if state == optuna.trial.TrialState.PRUNED:
        formatted_value = format_float(trial.value)
        logger.info(f"Trial {trial_number} was pruned at step {trial.last_step} with value {formatted_value}")
        print(f"[PRUNED] Trial {trial_number} at step {trial.last_step} with value {formatted_value}")

    elif state == optuna.trial.TrialState.COMPLETE:
        formatted_value = format_float(trial.value)
        logger.info(f"Trial {trial_number} completed with value {formatted_value}")

    elif state == optuna.trial.TrialState.FAIL:
        logger.info(f"Trial {trial_number} failed")
        cleanup_failed_trial(study_dir, trial_number)

    else:
        logger.info(f"Trial {trial_number} ended with state {state}")
