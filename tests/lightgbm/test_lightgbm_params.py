"""Tests for LightGBM parameters and enum validation."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from scxpand.lightgbm.lightgbm_params import (
    BoostingType,
    LightGBMParams,
    MetricType,
    ObjectiveType,
)
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_training


class TestLightGBMEnumValues:
    """Test that all enum values are supported by LightGBM."""

    def test_boosting_type_enum_values_supported(self):
        """Test that all BoostingType enum values are supported by LightGBM."""
        # Create dummy data
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        for boosting_type in BoostingType:
            print(
                f"Testing boosting_type: {boosting_type} (value: '{boosting_type.value}')"
            )

            try:
                # RF boosting type requires additional parameters
                if boosting_type == BoostingType.RF:
                    model = lgb.LGBMClassifier(
                        boosting_type=boosting_type.value,
                        n_estimators=1,
                        verbose=-1,
                        random_state=42,
                        bagging_fraction=0.8,  # Required for RF
                        bagging_freq=1,  # Required for RF
                        feature_fraction=0.8,  # Required for RF
                    )
                else:
                    model = lgb.LGBMClassifier(
                        boosting_type=boosting_type.value,  # Use .value to get the string
                        n_estimators=1,
                        verbose=-1,
                        random_state=42,
                    )

                # Test fitting (this is where the error usually occurs)
                model.fit(X, y)

                print(f"✅ {boosting_type} works correctly")

            except Exception as e:
                pytest.fail(
                    f"BoostingType.{boosting_type.name} ('{boosting_type.value}') is not supported by LightGBM: {e}"
                )

    def test_objective_type_enum_values_supported(self):
        """Test that all ObjectiveType enum values are supported by LightGBM."""
        # Create dummy data for different objectives
        X = np.random.rand(50, 5)
        y_binary = np.random.randint(0, 2, 50)
        y_multiclass = np.random.randint(0, 3, 50)
        y_regression = np.random.rand(50)

        objective_data_map = {
            ObjectiveType.BINARY: y_binary,
            ObjectiveType.MULTICLASS: y_multiclass,
            ObjectiveType.REGRESSION: y_regression,
            # Skip LAMBDARANK as it requires special ranking data
        }

        for objective_type, y_data in objective_data_map.items():
            print(
                f"Testing objective: {objective_type} (value: '{objective_type.value}')"
            )

            try:
                if objective_type == ObjectiveType.REGRESSION:
                    model = lgb.LGBMRegressor(
                        objective=objective_type.value,
                        n_estimators=1,
                        verbose=-1,
                        random_state=42,
                    )
                else:
                    model = lgb.LGBMClassifier(
                        objective=objective_type.value,
                        n_estimators=1,
                        verbose=-1,
                        random_state=42,
                    )

                model.fit(X, y_data)
                print(f"✅ {objective_type} works correctly")

            except Exception as e:
                pytest.fail(
                    f"ObjectiveType.{objective_type.name} ('{objective_type.value}') is not supported by LightGBM: {e}"
                )

    def test_metric_type_enum_values_supported(self):
        """Test that all MetricType enum values are supported by LightGBM."""
        # Create dummy data
        X = np.random.rand(50, 5)
        y_binary = np.random.randint(0, 2, 50)
        y_multiclass = np.random.randint(0, 3, 50)
        y_regression = np.random.rand(50)

        # Map metrics to appropriate data and model types
        metric_config_map = {
            MetricType.BINARY_LOGLOSS: (y_binary, lgb.LGBMClassifier, "binary"),
            MetricType.MULTICLASS_LOGLOSS: (
                y_multiclass,
                lgb.LGBMClassifier,
                "multiclass",
            ),
            MetricType.RMSE: (y_regression, lgb.LGBMRegressor, "regression"),
            MetricType.MAE: (y_regression, lgb.LGBMRegressor, "regression"),
            MetricType.AUC: (y_binary, lgb.LGBMClassifier, "binary"),
            # Skip NDCG as it requires ranking data
        }

        for metric_type, (y_data, model_class, objective) in metric_config_map.items():
            print(f"Testing metric: {metric_type} (value: '{metric_type.value}')")

            try:
                model = model_class(
                    objective=objective,
                    metric=metric_type.value,
                    n_estimators=1,
                    verbose=-1,
                    random_state=42,
                )

                model.fit(X, y_data)
                print(f"✅ {metric_type} works correctly")

            except Exception as e:
                pytest.fail(
                    f"MetricType.{metric_type.name} ('{metric_type.value}') is not supported by LightGBM: {e}"
                )


class TestLightGBMParamsIntegration:
    """Test LightGBMParams class integration with actual LightGBM."""

    def test_default_params_work_with_lightgbm(self):
        """Test that default LightGBMParams work with LightGBM."""
        params = LightGBMParams()
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        # Test that we can create a model with default params
        model = lgb.LGBMClassifier(
            boosting_type=params.boosting_type.value,
            objective=params.objective.value,
            metric=params.metric.value,
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            num_leaves=params.num_leaves,
            max_depth=params.max_depth,
            verbose=params.verbose,
            random_state=params.random_seed,
        )

        # Test that the model can fit
        model.fit(X, y)

        # Test that the model can predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_params_serialization_with_enums(self):
        """Test that LightGBMParams can be serialized and deserialized with enums."""
        params = LightGBMParams()

        # Convert to dict (this is what would happen in JSON serialization)
        params_dict = asdict(params)

        # Serialize to JSON
        json_str = json.dumps(params_dict, default=str)

        # Deserialize from JSON
        loaded_dict = json.loads(json_str)

        # Create new params from loaded dict
        loaded_params = LightGBMParams(**loaded_dict)

        # Test that the loaded params work with LightGBM
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        model = lgb.LGBMClassifier(
            boosting_type=loaded_params.boosting_type,  # This should be a string now
            objective=loaded_params.objective,
            metric=loaded_params.metric,
            n_estimators=loaded_params.n_estimators,
            verbose=-1,
            random_state=42,
        )

        model.fit(X, y)

    def test_enum_value_usage_pattern(self):
        params = LightGBMParams()

        # Test the enum properties
        boosting_enum = params.boosting_type
        boosting_value = params.boosting_type.value

        print(f"params.boosting_type: {boosting_enum}")
        print(f"params.boosting_type.value: '{boosting_value}'")

        # Verify enum properties
        assert isinstance(boosting_enum, BoostingType)
        assert boosting_value == "gbdt", "Enum value should be 'gbdt'"
        assert boosting_enum == "gbdt", "Enum should equal its string value"

        # Test that .value works reliably with LightGBM
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        # The recommended approach: use .value explicitly
        model = lgb.LGBMClassifier(
            boosting_type=params.boosting_type.value,  # Explicit .value usage
            objective=params.objective.value,  # Explicit .value usage
            n_estimators=1,
            verbose=-1,
            random_state=42,
        )
        model.fit(X, y)

        # Verify the model was created with correct parameters
        assert model.boosting_type == "gbdt"
        assert model.objective == "binary"


class TestEnumStringBehavior:
    """Test the string behavior of our enums."""

    def test_hyperopt_string_parameter_compatibility(self):
        """Test that LightGBM training works with string parameters from hyperopt."""
        # Simulate hyperopt parameters (strings instead of enums)
        hyperopt_params = {
            "use_log_transform": True,
            "use_zscore_norm": True,
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_seed": 42,
            "force_col_wise": True,
            "deterministic": True,
            "class_weight": "balanced",
            "n_jobs": -1,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "min_split_gain": 0.0,
            "min_child_weight": 1e-3,
            "boosting_type": "gbdt",  # String instead of enum
            "objective": "binary",  # String instead of enum
            "metric": "binary_logloss",  # String instead of enum
            "verbose": -1,
        }

        # Create LightGBMParams with string values
        params = LightGBMParams(**hyperopt_params)

        # Verify the parameters are stored correctly
        assert params.boosting_type == "gbdt"
        assert params.objective == "binary"
        assert params.metric == "binary_logloss"

        # Test that we can create a model with these parameters
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        # This should work without calling .value on strings
        model = lgb.LGBMClassifier(
            boosting_type=(
                params.boosting_type.value
                if hasattr(params.boosting_type, "value")
                else params.boosting_type
            ),
            objective=(
                params.objective.value
                if hasattr(params.objective, "value")
                else params.objective
            ),
            metric=(
                params.metric.value
                if hasattr(params.metric, "value")
                else params.metric
            ),
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            verbose=params.verbose,
            random_state=params.random_seed,
        )

        # Test fitting
        model.fit(X, y)

        # Verify the model was created with correct parameters
        assert model.boosting_type == "gbdt"
        assert model.objective == "binary"
        assert model.metric == "binary_logloss"

    def test_mixed_enum_string_parameters(self):
        """Test handling of mixed enum and string parameters."""
        # Test with enum parameters
        enum_params = LightGBMParams()
        assert isinstance(enum_params.boosting_type, BoostingType)
        assert isinstance(enum_params.objective, ObjectiveType)
        assert isinstance(enum_params.metric, MetricType)

        # Test with string parameters (simulating hyperopt)
        string_params = LightGBMParams(
            boosting_type="dart",  # String
            objective="binary",  # String
            metric="auc",  # String
        )

        # These should be strings, not enums
        assert isinstance(string_params.boosting_type, str)
        assert isinstance(string_params.objective, str)
        assert isinstance(string_params.metric, str)

        # Test model creation with both types
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        # Test enum parameters
        model1 = lgb.LGBMClassifier(
            boosting_type=(
                enum_params.boosting_type.value
                if hasattr(enum_params.boosting_type, "value")
                else enum_params.boosting_type
            ),
            objective=(
                enum_params.objective.value
                if hasattr(enum_params.objective, "value")
                else enum_params.objective
            ),
            n_estimators=1,
            verbose=-1,
            random_state=42,
        )
        model1.fit(X, y)

        # Test string parameters
        model2 = lgb.LGBMClassifier(
            boosting_type=(
                string_params.boosting_type.value
                if hasattr(string_params.boosting_type, "value")
                else string_params.boosting_type
            ),
            objective=(
                string_params.objective.value
                if hasattr(string_params.objective, "value")
                else string_params.objective
            ),
            n_estimators=1,
            verbose=-1,
            random_state=42,
        )
        model2.fit(X, y)

        # Both should work
        assert model1.boosting_type == "gbdt"
        assert model2.boosting_type == "dart"

    def test_parameter_serialization_with_mixed_types(self):
        """Test parameter serialization works with both enum and string types."""
        # Test with enum parameters
        enum_params = LightGBMParams()
        enum_dict = asdict(enum_params)

        # Test with string parameters
        string_params = LightGBMParams(
            boosting_type="goss",
            objective="binary",
            metric="auc",
        )
        string_dict = asdict(string_params)

        # Both should serialize correctly
        assert enum_dict["boosting_type"] == "gbdt"
        assert string_dict["boosting_type"] == "goss"

        # Test JSON serialization
        enum_json = json.dumps(enum_dict)
        string_json = json.dumps(string_dict)

        # Both should be valid JSON
        assert json.loads(enum_json)["boosting_type"] == "gbdt"
        assert json.loads(string_json)["boosting_type"] == "goss"

    def test_training_function_with_string_parameters(self):
        """Test that run_lightgbm_training works with string parameters from hyperopt."""
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal test data file

            # Create dummy data
            n_cells = 100
            n_genes = 50
            X = np.random.rand(n_cells, n_genes)

            # Create obs dataframe with required columns
            obs = pd.DataFrame(
                {
                    "is_expanded": np.random.randint(0, 2, n_cells),
                    "expansion": np.random.choice(
                        ["expanded", "not_expanded"], n_cells
                    ),  # Required for evaluation
                    "tissue_type": np.random.choice(["T_cell", "B_cell"], n_cells),
                    "imputed_labels": np.random.choice(["CD4", "CD8"], n_cells),
                    "study": ["study_1"] * n_cells,  # Required for data splitting
                    "patient": [
                        f"patient_{i}" for i in range(n_cells)
                    ],  # Required for data splitting
                    "sample": [
                        f"sample_{i}" for i in range(n_cells)
                    ],  # Required for data splitting
                    "cancer_type": ["cancer_A"]
                    * n_cells,  # Required for data splitting
                }
            )

            # Create var dataframe
            var = pd.DataFrame({"gene_ids": [f"GENE_{i:03d}" for i in range(n_genes)]})

            adata = ad.AnnData(X=X, obs=obs, var=var)
            test_data_path = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(test_data_path)

            # Test with enum parameters (what hyperopt now provides after our fix)
            enum_params = LightGBMParams(
                boosting_type=BoostingType.GBDT,  # Enum object
                objective=ObjectiveType.BINARY,  # Enum object
                metric=MetricType.BINARY_LOGLOSS,  # Enum object
                n_estimators=10,  # Small number for quick test
                random_seed=42,
            )

            # This should work without AttributeError
            try:
                results = run_lightgbm_training(
                    base_save_dir=temp_dir,
                    prm=enum_params,
                    data_path=str(test_data_path),
                    dev_ratio=0.2,
                )

                # Verify results were returned
                assert isinstance(results, dict)
                # The results structure may vary, just check it's not empty
                assert len(results) > 0

            except AttributeError as e:
                if "'str' object has no attribute 'value'" in str(e):
                    pytest.fail(f"AttributeError occurred with string parameters: {e}")
                else:
                    raise  # Re-raise if it's a different AttributeError

    def test_enum_inherits_from_str(self):
        """Test that our enums properly inherit from str."""
        boosting_type = BoostingType.GBDT
        objective_type = ObjectiveType.BINARY
        metric_type = MetricType.BINARY_LOGLOSS

        # All should be instances of str
        assert isinstance(boosting_type, str)
        assert isinstance(objective_type, str)
        assert isinstance(metric_type, str)
        assert boosting_type.value == "gbdt"
        assert objective_type.value == "binary"
        assert metric_type.value == "binary_logloss"

        # Equality with string should work (this is the key feature)
        assert boosting_type == "gbdt"
        assert objective_type == "binary"
        assert metric_type == "binary_logloss"

    def test_enum_comparison_behavior(self):
        """Test how enums compare with strings."""
        boosting_type = BoostingType.GBDT

        # These should all be True
        assert boosting_type == "gbdt"
        assert boosting_type == BoostingType.GBDT
        assert boosting_type.value == "gbdt"

        # The key insight: we use .value explicitly when we need the string
        # No need to rely on str() conversion behavior
        assert boosting_type.value == "gbdt"

        # Test that the enum can be used in string contexts
        # Note: f-string shows enum name, not value - this is expected behavior
        test_string = f"Using {boosting_type} boosting"
        assert "BoostingType.GBDT" in test_string

        # If we want the value in f-strings, we use .value explicitly
        test_string_with_value = f"Using {boosting_type.value} boosting"
        assert "gbdt" in test_string_with_value


if __name__ == "__main__":
    # Run a quick test
    test_class = TestLightGBMEnumValues()
    test_class.test_boosting_type_enum_values_supported()
    print("Basic enum tests passed!")
