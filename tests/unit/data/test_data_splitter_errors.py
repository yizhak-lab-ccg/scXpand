"""Tests for error handling in data_splitter.py."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scxpand.data_util.data_splitter import split_data


class TestDataSplitterErrorHandling:
    """Test error handling in data splitter functions."""

    def test_split_data_multiple_cancer_types_per_patient(self):
        """Test that ValueError is raised when a patient has multiple cancer types."""
        # Create test data with a patient having multiple cancer types
        n_cells = 100
        obs_data = {
            "study": ["study1"] * n_cells,
            "patient": ["patient1"] * 50 + ["patient2"] * 50,
            "sample": [f"sample_{i}" for i in range(n_cells)],
            "cancer_type": ["type1"] * 25
            + ["type2"] * 25
            + ["type1"] * 50,  # patient1 has both type1 and type2
            "imputed_labels": ["label1"] * n_cells,
            "tissue_type": ["tissue1"] * n_cells,
        }

        obs_df = pd.DataFrame(obs_data)
        X = np.random.rand(n_cells, 10)
        adata = AnnData(X=X, obs=obs_df)

        with pytest.raises(
            ValueError, match="Patient study1:patient1 has multiple cancer types"
        ):
            split_data(adata, dev_ratio=0.2, random_seed=42)

    def test_split_data_single_cancer_type_per_patient_success(self):
        """Test successful data splitting when each patient has exactly one cancer type."""
        # Create test data with multiple patients per cancer type (needed for stratification)
        n_cells = 200
        obs_data = {
            "study": ["study1"] * n_cells,
            "patient": (
                ["patient1"] * 25
                + ["patient2"] * 25
                + ["patient3"] * 25
                + ["patient4"] * 25
                + ["patient5"] * 25
                + ["patient6"] * 25
                + ["patient7"] * 25
                + ["patient8"] * 25
            ),
            "sample": [f"sample_{i}" for i in range(n_cells)],
            "cancer_type": (
                ["type1"] * 50 + ["type2"] * 50 + ["type1"] * 50 + ["type2"] * 50
            ),  # Multiple patients per cancer type
            "imputed_labels": ["label1"] * n_cells,
            "tissue_type": ["tissue1"] * n_cells,
        }

        obs_df = pd.DataFrame(obs_data)
        X = np.random.rand(n_cells, 10)
        adata = AnnData(X=X, obs=obs_df)

        # Should succeed without raising an exception
        train_indices, dev_indices = split_data(adata, dev_ratio=0.2, random_seed=42)

        # Verify the split
        assert len(train_indices) > 0
        assert len(dev_indices) > 0
        assert len(train_indices) + len(dev_indices) == n_cells
        assert len(set(train_indices) & set(dev_indices)) == 0  # No overlap

    def test_split_data_empty_cancer_type_array(self):
        """Test behavior when a patient has inconsistent cancer type data."""
        # Create test data where one patient has mixed cancer types including None
        n_cells = 100
        obs_data = {
            "study": ["study1"] * n_cells,
            "patient": ["patient1"] * 50 + ["patient2"] * 50,
            "sample": [f"sample_{i}" for i in range(n_cells)],
            "cancer_type": ["type1"] * 50
            + ["type1"] * 25
            + ["type2"] * 25,  # patient2 has mixed types
            "imputed_labels": ["label1"] * n_cells,
            "tissue_type": ["tissue1"] * n_cells,
        }

        obs_df = pd.DataFrame(obs_data)
        X = np.random.rand(n_cells, 10)
        adata = AnnData(X=X, obs=obs_df)

        # This should raise a ValueError because patient2 has multiple cancer types
        with pytest.raises(
            ValueError, match="Patient study1:patient2 has multiple cancer types"
        ):
            split_data(adata, dev_ratio=0.2, random_seed=42)

    def test_split_data_preserves_stratification(self):
        """Test that cancer type stratification is preserved in the split."""
        # Create test data with multiple patients per cancer type
        n_cells_per_patient = 20
        patients_per_type = 5
        cancer_types = ["type1", "type2"]

        obs_data = {
            "study": [],
            "patient": [],
            "sample": [],
            "cancer_type": [],
            "imputed_labels": [],
            "tissue_type": [],
        }

        cell_idx = 0
        for cancer_type in cancer_types:
            for patient_idx in range(patients_per_type):
                patient_id = f"patient_{cancer_type}_{patient_idx}"
                for _cell_idx_in_patient in range(n_cells_per_patient):
                    obs_data["study"].append("study1")
                    obs_data["patient"].append(patient_id)
                    obs_data["sample"].append(f"sample_{cell_idx}")
                    obs_data["cancer_type"].append(cancer_type)
                    obs_data["imputed_labels"].append("label1")
                    obs_data["tissue_type"].append("tissue1")
                    cell_idx += 1

        n_cells = len(obs_data["study"])
        obs_df = pd.DataFrame(obs_data)
        X = np.random.rand(n_cells, 10)
        adata = AnnData(X=X, obs=obs_df)

        train_indices, dev_indices = split_data(adata, dev_ratio=0.2, random_seed=42)

        # Check that both splits contain both cancer types
        train_cancer_types = set(obs_df.iloc[train_indices]["cancer_type"].unique())
        dev_cancer_types = set(obs_df.iloc[dev_indices]["cancer_type"].unique())

        assert train_cancer_types == set(cancer_types)
        assert dev_cancer_types == set(cancer_types)
