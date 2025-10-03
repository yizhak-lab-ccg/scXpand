"""Tests for the inference and list_pretrained_models functions in main.py."""

import os
from unittest.mock import Mock, patch

import pytest

from scxpand.core.inference import DEFAULT_MODEL_NAME
from scxpand.main import inference
from scxpand.util.model_registry import list_pretrained_models


@pytest.fixture(autouse=True, scope="module")
def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


class TestInferenceCommand:
    """Tests for the unified inference command."""

    def test_inference_local_model_success(self):
        """Test inference with local model path."""
        with (
            patch("scxpand.main.run_inference") as mock_run_inference,
            patch("scxpand.main.load_eval_indices") as mock_load_indices,
        ):
            # Setup mocks
            mock_load_indices.return_value = [0, 1, 2, 3, 4]

            inference(
                data_path="data/test.h5ad",
                model_path="results/pan_cancer_mlp",
                eval_row_inds="eval_indices.txt",
            )

            # Verify unified inference function was called
            mock_run_inference.assert_called_once()
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["model_path"] == "results/pan_cancer_mlp"
            assert call_kwargs["data_path"] == "data/test.h5ad"
            assert call_kwargs["eval_row_inds"] == [0, 1, 2, 3, 4]

            mock_load_indices.assert_called_once_with("eval_indices.txt")

    def test_inference_registry_model_success(self):
        """Test inference with registry model name."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_name=DEFAULT_MODEL_NAME,
                batch_size=512,
            )

            # Verify unified inference function was called
            mock_run_inference.assert_called_once()
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["model_name"] == DEFAULT_MODEL_NAME
            assert call_kwargs["model_url"] is None
            assert call_kwargs["model_path"] is None
            assert call_kwargs["data_path"] == "data/test.h5ad"
            assert call_kwargs["batch_size"] == 512

    def test_inference_url_model_success(self):
        """Test inference with URL model."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_url="https://your-platform.com/model.zip",
                save_path="custom/save/path",
            )

            # Verify unified inference function was called
            mock_run_inference.assert_called_once()
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["model_name"] is None
            assert call_kwargs["model_url"] == "https://your-platform.com/model.zip"
            assert call_kwargs["model_path"] is None
            assert call_kwargs["data_path"] == "data/test.h5ad"
            assert call_kwargs["save_path"] == "custom/save/path"

    def test_inference_no_model_source_uses_default(self):
        """Test inference uses default model when no model source is provided."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(data_path="data/test.h5ad")

            # Verify unified inference function was called
            # The inference function passes None, but run_inference will internally use default
            mock_run_inference.assert_called_once()
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["model_name"] is None  # inference passes None
            assert call_kwargs["model_url"] is None
            assert call_kwargs["model_path"] is None
            assert call_kwargs["data_path"] == "data/test.h5ad"

    def test_inference_multiple_model_sources_error(self):
        """Test inference raises error when multiple model sources are provided."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            inference(
                data_path="data/test.h5ad",
                model_path="results/pan_cancer_mlp",
                model_name=DEFAULT_MODEL_NAME,
            )

    def test_inference_multiple_model_sources_all_three_error(self):
        """Test inference raises error when all three model sources are provided."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            inference(
                data_path="data/test.h5ad",
                model_path="results/pan_cancer_mlp",
                model_name=DEFAULT_MODEL_NAME,
                model_url="https://your-platform.com/model.zip",
            )

    def test_inference_registry_model_with_custom_save_path(self):
        """Test inference with registry model sets default save path correctly."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_name="pan_cancer_mlp",
            )

            # Verify default save path for registry model
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["save_path"] == "results/pan_cancer_mlp_predictions"

    def test_inference_url_model_with_no_default_save_path(self):
        """Test inference with URL model doesn't set default save path."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_url="https://your-platform.com/model.zip",
            )

            # Verify no default save path for URL model (user must specify)
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["save_path"] is None

    def test_inference_with_all_parameters(self):
        """Test inference with all parameters specified."""
        with (
            patch("scxpand.main.run_inference") as mock_run_inference,
            patch("scxpand.main.load_eval_indices") as mock_load_indices,
        ):
            mock_load_indices.return_value = [1, 2, 3]

            inference(
                data_path="data/test.h5ad",
                model_path="results/test_model",
                save_path="custom/predictions",
                batch_size=512,
                num_workers=8,
                eval_row_inds="eval.txt",
            )

            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["data_path"] == "data/test.h5ad"
            assert call_kwargs["model_path"] == "results/test_model"
            assert call_kwargs["save_path"] == "custom/predictions"
            assert call_kwargs["batch_size"] == 512
            assert call_kwargs["num_workers"] == 8
            assert call_kwargs["eval_row_inds"] == [1, 2, 3]

    def test_inference_empty_eval_indices_file(self):
        """Test inference with empty eval indices file."""
        with (
            patch("scxpand.main.run_inference") as mock_run_inference,
            patch("scxpand.main.load_eval_indices") as mock_load_indices,
        ):
            mock_load_indices.return_value = []

            inference(
                data_path="data/test.h5ad",
                model_path="results/test_model",
                eval_row_inds="empty_eval.txt",
            )

            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["eval_row_inds"] == []

    def test_inference_registry_model_with_version(self):
        """Test inference with registry model name that includes version."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_name="pan_cancer_mlp_v1.0",
            )

            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["model_name"] == "pan_cancer_mlp_v1.0"
            assert call_kwargs["save_path"] == "results/pan_cancer_mlp_v1.0_predictions"

    def test_inference_local_model_auto_detect_type(self):
        """Test inference with local model auto-detects model type."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_path="results/pan_cancer_mlp",
            )

            # Verify unified inference function was called (model type auto-detected)
            mock_run_inference.assert_called_once()
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["model_path"] == "results/pan_cancer_mlp"

    def test_inference_local_model_with_eval_indices_none(self):
        """Test inference with local model and no eval indices."""
        with patch("scxpand.main.run_inference") as mock_run_inference:
            inference(
                data_path="data/test.h5ad",
                model_path="results/pan_cancer_mlp",
                eval_row_inds=None,
            )

            # Verify eval_row_inds is None
            call_kwargs = mock_run_inference.call_args[1]
            assert call_kwargs["eval_row_inds"] is None


class TestListPretrainedModelsCommand:
    """Tests for the list_pretrained_models command."""

    def test_list_pretrained_models_success(self):
        """Test list_pretrained_models displays all models correctly."""
        mock_models = {
            "model1": Mock(
                version="1.0.0",
                url="https://example.com/model1.tar.gz",
            ),
            "model2": Mock(
                version="2.0.0",
                url="",
            ),
        }

        with (
            patch("scxpand.util.model_registry.PRETRAINED_MODELS", mock_models),
            patch("scxpand.util.model_registry.logger") as mock_logger,
        ):
            list_pretrained_models()

            # Verify logger calls
            assert mock_logger.info.call_count >= 2  # At least header and models

            # Check that model information was logged
            logged_messages = [call.args[0] for call in mock_logger.info.call_args_list]

            # Should contain header
            assert any(
                "Available pre-trained models:" in msg for msg in logged_messages
            )

            # Should contain model names
            assert any("model1" in msg for msg in logged_messages)
            assert any("model2" in msg for msg in logged_messages)

            # Should contain version info
            assert any("1.0.0" in msg for msg in logged_messages)
            assert any("2.0.0" in msg for msg in logged_messages)

            # Should contain URL status
            assert any("URL configured: Yes" in msg for msg in logged_messages)
            assert any("URL configured: No" in msg for msg in logged_messages)

            # Should contain model type auto-detection message
            assert any(
                "Auto-detected from model_type.txt" in msg for msg in logged_messages
            )

    def test_list_pretrained_models_empty_registry(self):
        """Test list_pretrained_models with empty model registry."""
        with (
            patch("scxpand.util.model_registry.PRETRAINED_MODELS", {}),
            patch("scxpand.util.model_registry.logger") as mock_logger,
        ):
            list_pretrained_models()

            # Should still show header
            logged_messages = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any(
                "Available pre-trained models:" in msg for msg in logged_messages
            )
