import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.data_util.data_splitter import (
    PATIENT_ID_SEPARATOR,
    get_patient_identifiers,
    split_data,
)


@pytest.fixture
def mock_adata():
    # Create mock data with 100 cells, 10 genes
    X = np.random.rand(100, 10)

    # Create mock observation dataframe with 4 patients, 2 per cancer type
    # Use string indices from the start to avoid AnnData warnings
    obs = pd.DataFrame(
        {
            "study": np.repeat(
                ["study1", "study1", "study2", "study2"], [25, 25, 25, 25]
            ),
            "patient": np.repeat(["p1", "p2", "p3", "p4"], [25, 25, 25, 25]),
            "cancer_type": np.repeat(
                ["typeA", "typeB"], [50, 50]
            ),  # 2 patients per cancer type
            "tissue_type": np.tile(["normal", "tumor"], 50),
            "imputed_labels": np.tile(["label1", "label2"], 50),
            "sample": [f"s{i}" for i in range(100)],
        },
        index=pd.Index([str(i) for i in range(100)], dtype="string"),
    )

    return ad.AnnData(X=X, obs=obs)


def test_split_data_basic(mock_adata):
    dev_ratio = 0.3
    row_inds_train, row_inds_dev = split_data(mock_adata, dev_ratio, random_seed=42)

    # Test 1: Check if indices are valid
    assert all(0 <= idx < len(mock_adata) for idx in row_inds_train)
    assert all(0 <= idx < len(mock_adata) for idx in row_inds_dev)

    # Test 2: Check if there's no overlap between train and dev sets
    assert len(set(row_inds_train) & set(row_inds_dev)) == 0

    # Test 3: Check if all indices are used exactly once
    assert len(row_inds_train) + len(row_inds_dev) == len(mock_adata)
    assert len(set(row_inds_train) | set(row_inds_dev)) == len(mock_adata)


def test_split_data_patient_separation(mock_adata):
    dev_ratio = 0.3
    row_inds_train, row_inds_dev = split_data(mock_adata, dev_ratio, random_seed=42)

    # Get patient IDs for each set using the get_patient_identifiers function
    obs_df = mock_adata.obs
    train_patient_ids = get_patient_identifiers(obs_df.iloc[row_inds_train])
    dev_patient_ids = get_patient_identifiers(obs_df.iloc[row_inds_dev])

    train_patients = set(train_patient_ids)
    dev_patients = set(dev_patient_ids)

    # Test: No patient should appear in both sets
    assert len(train_patients & dev_patients) == 0


def test_split_data_cancer_distribution(mock_adata):
    dev_ratio = 0.3
    row_inds_train, row_inds_dev = split_data(mock_adata, dev_ratio, random_seed=42)

    # Get cancer type distributions
    train_cancers = mock_adata.obs.iloc[row_inds_train]["cancer_type"].value_counts(
        normalize=True
    )
    dev_cancers = mock_adata.obs.iloc[row_inds_dev]["cancer_type"].value_counts(
        normalize=True
    )

    # Test: Cancer type distributions should be roughly similar (within 15% difference)
    for cancer_type in train_cancers.index:
        assert abs(train_cancers[cancer_type] - dev_cancers[cancer_type]) < 0.15


def test_split_data_reproducibility(mock_adata):
    dev_ratio = 0.3
    seed = 42

    # Generate two splits with the same seed
    split1_train, split1_dev = split_data(mock_adata, dev_ratio, random_seed=seed)
    split2_train, split2_dev = split_data(mock_adata, dev_ratio, random_seed=seed)

    # Test: Same seed should produce identical splits
    assert np.array_equal(split1_train, split2_train)
    assert np.array_equal(split1_dev, split2_dev)


class TestGetPatientIdentifiers:
    """Test suite for get_patient_identifiers function and separator validation."""

    def test_separator_constant_exists(self):
        """Test that the separator constant is properly defined."""
        assert PATIENT_ID_SEPARATOR == ":"
        assert isinstance(PATIENT_ID_SEPARATOR, str)
        assert len(PATIENT_ID_SEPARATOR) == 1

    def test_valid_patient_identifiers(self):
        """Test generation of valid patient identifiers."""
        obs_df = pd.DataFrame(
            {"study": ["study1", "study2", "study1"], "patient": ["p1", "p2", "p3"]}
        )

        result = get_patient_identifiers(obs_df)
        expected = pd.Series(["study1:p1", "study2:p2", "study1:p3"])

        pd.testing.assert_series_equal(result, expected)

    def test_invalid_study_with_separator(self):
        """Test that study identifiers containing separator raise ValueError."""
        obs_df = pd.DataFrame({"study": ["study:1", "study2"], "patient": ["p1", "p2"]})

        with pytest.raises(
            ValueError, match="Study identifiers cannot contain ':' character"
        ):
            get_patient_identifiers(obs_df)

    def test_invalid_patient_with_separator(self):
        """Test that patient identifiers containing separator raise ValueError."""
        obs_df = pd.DataFrame({"study": ["study1", "study2"], "patient": ["p:1", "p2"]})

        with pytest.raises(
            ValueError, match="Patient identifiers cannot contain ':' character"
        ):
            get_patient_identifiers(obs_df)

    def test_multiple_invalid_studies(self):
        """Test error message includes all invalid study identifiers."""
        obs_df = pd.DataFrame(
            {
                "study": ["study:1", "study:2", "valid_study"],
                "patient": ["p1", "p2", "p3"],
            }
        )

        with pytest.raises(ValueError, match="Found invalid studies") as exc_info:
            get_patient_identifiers(obs_df)

        error_msg = str(exc_info.value)
        assert "study:1" in error_msg
        assert "study:2" in error_msg

    def test_multiple_invalid_patients(self):
        """Test error message includes all invalid patient identifiers."""
        obs_df = pd.DataFrame(
            {
                "study": ["study1", "study2", "study3"],
                "patient": ["p:1", "p:2", "valid_patient"],
            }
        )

        with pytest.raises(ValueError, match="Found invalid patients") as exc_info:
            get_patient_identifiers(obs_df)

        error_msg = str(exc_info.value)
        assert "p:1" in error_msg
        assert "p:2" in error_msg

    def test_numeric_identifiers_converted_to_string(self):
        """Test that numeric study/patient IDs are properly converted to strings."""
        obs_df = pd.DataFrame({"study": [1, 2, 3], "patient": [101, 102, 103]})

        result = get_patient_identifiers(obs_df)
        expected = pd.Series(["1:101", "2:102", "3:103"])

        pd.testing.assert_series_equal(result, expected)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        obs_df = pd.DataFrame({"study": [], "patient": []})

        result = get_patient_identifiers(obs_df)
        expected = pd.Series([], dtype=object)

        pd.testing.assert_series_equal(result, expected)

    def test_special_characters_allowed(self):
        """Test that other special characters (except separator) are allowed."""
        obs_df = pd.DataFrame(
            {
                "study": ["study-1", "study_2", "study.3"],
                "patient": ["p@1", "p#2", "p$3"],
            }
        )

        result = get_patient_identifiers(obs_df)
        expected = pd.Series(["study-1:p@1", "study_2:p#2", "study.3:p$3"])

        pd.testing.assert_series_equal(result, expected)

    def test_whitespace_handling(self):
        """Test handling of whitespace in identifiers."""
        obs_df = pd.DataFrame(
            {
                "study": ["study 1", " study2", "study3 "],
                "patient": ["p 1", " p2", "p3 "],
            }
        )

        result = get_patient_identifiers(obs_df)
        expected = pd.Series(["study 1:p 1", " study2: p2", "study3 :p3 "])

        pd.testing.assert_series_equal(result, expected)

    def test_na_values_converted_to_string(self):
        """Test that NaN values are converted to string representation."""
        obs_df = pd.DataFrame(
            {"study": ["study1", np.nan, "study3"], "patient": ["p1", "p2", np.nan]}
        )

        result = get_patient_identifiers(obs_df)
        # NaN becomes "nan" when converted to string
        expected = pd.Series(["study1:p1", "nan:p2", "study3:nan"])

        pd.testing.assert_series_equal(result, expected)
