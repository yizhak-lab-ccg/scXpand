from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from scxpand.util.general_util import metrics_dict_to_table
from scxpand.util.logger import get_logger


logger = get_logger()

# Constant for patient identifier separator
PATIENT_ID_SEPARATOR = ":"


def get_patient_identifiers(obs_df: pd.DataFrame) -> pd.Series:
    """Generate unique patient identifiers by combining study and patient columns.

    Creates composite identifiers in the format 'study:patient' to uniquely
    identify patients across different studies.

    Args:
        obs_df: DataFrame containing 'study' and 'patient' columns.

    Returns:
        Series of unique patient identifiers.

    Raises:
        ValueError: If study or patient identifiers contain the separator character.

    Example:
        >>> identifiers = get_patient_identifiers(adata.obs)
        >>> print(identifiers.head())  # ['study1:patient1', 'study1:patient2', ...]
    """
    study_strings = obs_df["study"].astype(str)
    patient_strings = obs_df["patient"].astype(str)

    # Check for separator character in study strings
    invalid_studies = study_strings[study_strings.str.contains(PATIENT_ID_SEPARATOR, na=False)]
    if not invalid_studies.empty:
        raise ValueError(
            f"Study identifiers cannot contain '{PATIENT_ID_SEPARATOR}' character. "
            f"Found invalid studies: {sorted(invalid_studies.unique())}"
        )

    # Check for separator character in patient strings
    invalid_patients = patient_strings[patient_strings.str.contains(PATIENT_ID_SEPARATOR, na=False)]
    if not invalid_patients.empty:
        raise ValueError(
            f"Patient identifiers cannot contain '{PATIENT_ID_SEPARATOR}' character. "
            f"Found invalid patients: {sorted(invalid_patients.unique())}"
        )

    return study_strings + PATIENT_ID_SEPARATOR + patient_strings


def log_dataset_overview(obs_df: pd.DataFrame, patient_identifiers: pd.Series) -> None:
    """Log overview statistics of the dataset.

    Args:
        obs_df: DataFrame containing observation data
        patient_identifiers: Series of unique patient identifiers
    """
    n_patients = len(patient_identifiers.unique())
    n_samples = len(obs_df["sample"].unique())
    cancer_types_options = np.unique(obs_df["cancer_type"].to_list())
    n_cancer_types = len(cancer_types_options)

    logger.info(
        f"Full dataset overview:\n"
        f"  Number of cells: {len(obs_df)}\n"
        f"  Number of patients: {n_patients}\n"
        f"  Number of samples: {n_samples}\n"
        f"  Number of cancer types: {n_cancer_types}"
    )


def log_split_results(
    n_train_patients: int,
    n_dev_patients: int,
    n_train_cells: int,
    n_dev_cells: int,
    n_total_cells: int,
) -> None:
    """Log the results of the data split.

    Args:
        n_train_patients: Number of patients in training set
        n_dev_patients: Number of patients in validation set
        n_train_cells: Number of cells in training set
        n_dev_cells: Number of cells in validation set
        n_total_cells: Total number of cells
    """
    logger.info(
        f"Data split results:\n"
        f"  Training set: {n_train_patients} patients, {n_train_cells} cells\n"
        f"  Validation set: {n_dev_patients} patients, {n_dev_cells} cells\n"
        f"  Total cells: {n_total_cells}"
    )


def calculate_and_log_cancer_distribution(
    train_patient_ids: list[str],
    dev_patient_ids: list[str],
    uniq_patient_ids: list[str],
    cancer_types_per_patient: list[str],
) -> None:
    """Calculate and log cancer type distribution across train/validation splits.

    Args:
        train_patient_ids: List of patient IDs in training set
        dev_patient_ids: List of patient IDs in validation set
        uniq_patient_ids: List of all unique patient IDs
        cancer_types_per_patient: List of cancer types per patient
    """
    train_cancers = [cancer_types_per_patient[uniq_patient_ids.index(pid)] for pid in train_patient_ids]
    dev_cancers = [cancer_types_per_patient[uniq_patient_ids.index(pid)] for pid in dev_patient_ids]

    train_cancer_distribution = pd.Series(train_cancers).value_counts(normalize=True) * 100
    dev_cancer_distribution = pd.Series(dev_cancers).value_counts(normalize=True) * 100

    # Create table format for cancer type distribution
    cancer_dist_data = {
        "Training": train_cancer_distribution.to_dict(),
        "Validation": dev_cancer_distribution.to_dict(),
    }
    cancer_table = metrics_dict_to_table(cancer_dist_data, title="Cancer Type Distribution (%)", precision=2)
    logger.info(cancer_table)


def calculate_and_log_category_distributions(train_obs_df: pd.DataFrame, dev_obs_df: pd.DataFrame) -> None:
    """Calculate and log category distributions (imputed_labels, tissue_type) across train/validation splits.

    Args:
        train_obs_df: Training set observation DataFrame
        dev_obs_df: Validation set observation DataFrame
    """
    for category_type in ["imputed_labels", "tissue_type"]:
        train_distribution = train_obs_df[category_type].value_counts(normalize=True) * 100
        dev_distribution = dev_obs_df[category_type].value_counts(normalize=True) * 100

        # Create table format for category distribution
        category_dist_data = {
            "Training": train_distribution.to_dict(),
            "Validation": dev_distribution.to_dict(),
        }
        category_table = metrics_dict_to_table(
            category_dist_data, title=f"{category_type.replace('_', ' ').title()} Distribution (%)", precision=2
        )
        logger.info(category_table)


def validate_patient_cancer_types(
    uniq_patient_ids: list[str], patient_identifiers: pd.Series, obs_df: pd.DataFrame
) -> list[str]:
    """Validate that each patient has exactly one cancer type and return cancer types per patient.

    Args:
        uniq_patient_ids: List of unique patient IDs
        patient_identifiers: Series of patient identifiers
        obs_df: DataFrame containing observation data

    Returns:
        List of cancer types per patient

    Raises:
        ValueError: If any patient has multiple cancer types
    """
    cancer_types_per_patient = []
    for patient_id in uniq_patient_ids:
        cancers_arr = obs_df[patient_identifiers == patient_id]["cancer_type"].unique()
        if len(cancers_arr) != 1:
            raise ValueError(
                f"Patient {patient_id} has multiple cancer types: {cancers_arr}. "
                "Each patient must have exactly one cancer type for stratified splitting."
            )
        cancer_types_per_patient.append(cancers_arr[0])
    return cancer_types_per_patient


def save_patient_ids(save_path: Path, train_patient_ids: list[str], dev_patient_ids: list[str]) -> None:
    """Save patient IDs to CSV files.

    Args:
        save_path: Directory path to save files
        train_patient_ids: List of training patient IDs
        dev_patient_ids: List of validation patient IDs
    """
    pd.Series(train_patient_ids).to_csv(save_path / "train_patient_ids.csv", index=False)
    pd.Series(dev_patient_ids).to_csv(save_path / "dev_patient_ids.csv", index=False)


def split_data(
    adata: ad.AnnData,
    dev_ratio: float,
    random_seed: int | None = None,
    save_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the data into training and validation sets using patient-level stratification.

    This function performs stratified splitting at the patient level to ensure that:
    1. No patient appears in both training and validation sets
    2. Cancer type distributions are preserved across splits
    3. Detailed logging provides transparency into the split process

    Args:
        adata: AnnData object with the data
        dev_ratio: The ratio of the validation set (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        save_path: Optional path to save the split patient IDs as CSV files

    Returns:
        tuple: (row_inds_train, row_inds_dev) - sorted indices for training and validation sets
    """
    # Prepare data with original row indices for later mapping
    obs_df = adata.obs.copy()
    obs_df["original_row_idx"] = np.arange(len(obs_df))
    obs_df = obs_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Create patient identifiers and get unique patients
    patient_identifiers = get_patient_identifiers(obs_df)
    uniq_patient_ids = list(patient_identifiers.unique())

    # Log dataset overview
    log_dataset_overview(obs_df, patient_identifiers)

    # Validate patient cancer types and get cancer types per patient
    cancer_types_per_patient = validate_patient_cancer_types(uniq_patient_ids, patient_identifiers, obs_df)

    # Perform stratified split by patient
    train_patient_ids, dev_patient_ids = train_test_split(
        uniq_patient_ids,
        test_size=dev_ratio,
        stratify=cancer_types_per_patient,
        random_state=random_seed,
    )

    # Create train/validation observation DataFrames
    train_obs_df = obs_df[patient_identifiers.isin(train_patient_ids)]
    dev_obs_df = obs_df[patient_identifiers.isin(dev_patient_ids)]

    # Log split results
    log_split_results(
        n_train_patients=len(train_patient_ids),
        n_dev_patients=len(dev_patient_ids),
        n_train_cells=len(train_obs_df),
        n_dev_cells=len(dev_obs_df),
        n_total_cells=len(obs_df),
    )

    # Log distribution statistics
    calculate_and_log_cancer_distribution(
        train_patient_ids, dev_patient_ids, uniq_patient_ids, cancer_types_per_patient
    )
    calculate_and_log_category_distributions(train_obs_df, dev_obs_df)

    # Save patient IDs if requested
    if save_path:
        save_patient_ids(save_path, train_patient_ids, dev_patient_ids)

    # Extract and sort row indices for efficient data loading
    row_inds_train = train_obs_df["original_row_idx"].to_numpy()
    row_inds_dev = dev_obs_df["original_row_idx"].to_numpy()

    row_inds_train.sort()
    row_inds_dev.sort()

    return row_inds_train, row_inds_dev
