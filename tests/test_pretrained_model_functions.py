"""Tests for pretrained model management functions.

This module provides comprehensive tests for the public API functions
related to pretrained model management.
"""

from unittest.mock import Mock, patch

import pytest

from scxpand.pretrained import (
    PRETRAINED_MODELS,
    download_pretrained_model,
    get_pretrained_model_info,
)
from scxpand.pretrained.model_registry import PretrainedModelInfo


class TestGetPretrainedModelInfo:
    """Tests for the get_pretrained_model_info function."""

    def test_get_pretrained_model_info_success(self):
        """Test successful retrieval of pretrained model info."""
        # Test with a known model
        model_info = get_pretrained_model_info("pan_cancer_autoencoder")

        assert isinstance(model_info, PretrainedModelInfo)
        assert model_info.name == "pan_cancer_autoencoder"
        assert model_info.version == "1.0.0"
        assert "figshare.com" in model_info.url

    def test_get_pretrained_model_info_all_models(self):
        """Test retrieval of info for all available models."""
        for model_name in PRETRAINED_MODELS.keys():
            model_info = get_pretrained_model_info(model_name)

            assert isinstance(model_info, PretrainedModelInfo)
            assert model_info.name == model_name
            assert model_info.version == "1.0.0"
            assert model_info.url is not None

    def test_get_pretrained_model_info_invalid_model(self):
        """Test error handling for invalid model name."""
        with pytest.raises(
            ValueError, match="Pre-trained model 'invalid_model' not found"
        ):
            get_pretrained_model_info("invalid_model")

    def test_get_pretrained_model_info_empty_name(self):
        """Test error handling for empty model name."""
        with pytest.raises(ValueError, match="Pre-trained model '' not found"):
            get_pretrained_model_info("")

    def test_get_pretrained_model_info_none_name(self):
        """Test error handling for None model name."""
        with pytest.raises(ValueError, match="Pre-trained model 'None' not found"):
            get_pretrained_model_info(None)

    def test_get_pretrained_model_info_case_sensitivity(self):
        """Test that model names are case sensitive."""
        with pytest.raises(
            ValueError, match="Pre-trained model 'PAN_CANCER_AUTOENCODER' not found"
        ):
            get_pretrained_model_info("PAN_CANCER_AUTOENCODER")

    def test_get_pretrained_model_info_error_message_includes_available_models(self):
        """Test that error message includes list of available models."""
        try:
            get_pretrained_model_info("nonexistent_model")
        except ValueError as e:
            error_msg = str(e)
            # Check that all available models are mentioned in the error
            for model_name in PRETRAINED_MODELS.keys():
                assert model_name in error_msg


class TestDownloadPretrainedModel:
    """Tests for the download_pretrained_model function."""

    @pytest.fixture
    def mock_pooch_retrieve(self):
        """Mock pooch.retrieve function."""
        with patch(
            "scxpand.pretrained.download_manager.pooch.retrieve"
        ) as mock_retrieve:
            yield mock_retrieve

    @pytest.fixture
    def mock_get_pretrained_model_info(self):
        """Mock get_pretrained_model_info function."""
        with patch(
            "scxpand.pretrained.download_manager.get_pretrained_model_info"
        ) as mock_get_info:
            yield mock_get_info

    def test_download_pretrained_model_with_name_success(
        self, mock_pooch_retrieve, mock_get_pretrained_model_info, tmp_path
    ):
        """Test successful download with model name."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.name = "test_model"
        mock_model_info.url = "https://example.com/model.zip"
        mock_get_pretrained_model_info.return_value = mock_model_info

        # Create the expected directory structure
        extracted_model_dir = tmp_path / "extracted_model"
        extracted_model_dir.mkdir()
        mock_pooch_retrieve.return_value = str(extracted_model_dir)

        # Test download
        result_path = download_pretrained_model(
            model_name="test_model", cache_dir=tmp_path
        )

        # Verify calls
        mock_get_pretrained_model_info.assert_called_once_with("test_model")
        mock_pooch_retrieve.assert_called_once()

        # Verify pooch.retrieve arguments
        call_args = mock_pooch_retrieve.call_args
        assert call_args[1]["url"] == "https://example.com/model.zip"
        assert call_args[1]["path"] == str(tmp_path)
        assert call_args[1]["progressbar"] is True

        # Verify return value
        assert result_path == extracted_model_dir

    def test_download_pretrained_model_with_url_success(
        self, mock_pooch_retrieve, tmp_path
    ):
        """Test successful download with direct URL."""
        # Create the expected directory structure
        extracted_model_dir = tmp_path / "extracted_model"
        extracted_model_dir.mkdir()
        mock_pooch_retrieve.return_value = str(extracted_model_dir)

        # Test download
        result_path = download_pretrained_model(
            model_url="https://example.com/model.zip", cache_dir=tmp_path
        )

        # Verify pooch.retrieve was called with correct URL
        mock_pooch_retrieve.assert_called_once()
        call_args = mock_pooch_retrieve.call_args
        assert call_args[1]["url"] == "https://example.com/model.zip"
        assert call_args[1]["path"] == str(tmp_path)

        # Verify return value
        assert result_path == extracted_model_dir

    def test_download_pretrained_model_default_cache_dir(
        self, mock_pooch_retrieve, mock_get_pretrained_model_info, tmp_path
    ):
        """Test download with default cache directory."""
        # Mock current working directory
        with patch("scxpand.pretrained.download_manager.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path

            # Setup mocks
            mock_model_info = Mock()
            mock_model_info.url = "https://example.com/model.zip"
            mock_get_pretrained_model_info.return_value = mock_model_info

            # Create the expected directory structure
            extracted_model_dir = tmp_path / "extracted_model"
            extracted_model_dir.mkdir()
            mock_pooch_retrieve.return_value = str(extracted_model_dir)

            # Test download without cache_dir
            result_path = download_pretrained_model(model_name="test_model")

            # Verify default cache directory was used
            expected_cache_dir = tmp_path / ".scxpand_cache"
            mock_pooch_retrieve.assert_called_once()
            call_args = mock_pooch_retrieve.call_args
            assert call_args[1]["path"] == str(expected_cache_dir)

            # Verify cache directory was created
            assert expected_cache_dir.exists()

            # Verify return value
            assert result_path == extracted_model_dir

    def test_download_pretrained_model_no_model_source_error(self):
        """Test error when no model source is provided."""
        with pytest.raises(
            ValueError, match="Either model_name or model_url must be provided"
        ):
            download_pretrained_model()

    def test_download_pretrained_model_both_sources_error(self):
        """Test error when both model sources are provided."""
        with pytest.raises(
            ValueError, match="Cannot specify both model_name and model_url"
        ):
            download_pretrained_model(
                model_name="test_model", model_url="https://example.com/model.zip"
            )

    def test_download_pretrained_model_missing_url_error(
        self, mock_get_pretrained_model_info
    ):
        """Test error when model info has no URL."""
        # Setup mock with no URL
        mock_model_info = Mock()
        mock_model_info.name = "test_model"
        mock_model_info.url = None
        mock_get_pretrained_model_info.return_value = mock_model_info

        with pytest.raises(
            ValueError, match="Download URL not configured for model 'test_model'"
        ):
            download_pretrained_model(model_name="test_model")

    def test_download_pretrained_model_empty_url_error(
        self, mock_get_pretrained_model_info
    ):
        """Test error when model info has empty URL."""
        # Setup mock with empty URL
        mock_model_info = Mock()
        mock_model_info.name = "test_model"
        mock_model_info.url = ""
        mock_get_pretrained_model_info.return_value = mock_model_info

        with pytest.raises(
            ValueError, match="Download URL not configured for model 'test_model'"
        ):
            download_pretrained_model(model_name="test_model")

    def test_download_pretrained_model_pooch_exception(
        self, mock_pooch_retrieve, mock_get_pretrained_model_info, tmp_path
    ):
        """Test error handling when pooch.retrieve fails."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.url = "https://example.com/model.zip"
        mock_get_pretrained_model_info.return_value = mock_model_info

        # Make pooch.retrieve raise an exception
        mock_pooch_retrieve.side_effect = Exception("Download failed")

        with pytest.raises(FileNotFoundError, match="Failed to download model from"):
            download_pretrained_model(model_name="test_model", cache_dir=tmp_path)

    def test_download_pretrained_model_processor_selection(
        self, mock_pooch_retrieve, mock_get_pretrained_model_info, tmp_path
    ):
        """Test that correct processor is selected based on file extension."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.url = "https://example.com/model.tar.gz"
        mock_get_pretrained_model_info.return_value = mock_model_info

        # Create the expected directory structure
        extracted_model_dir = tmp_path / "extracted_model"
        extracted_model_dir.mkdir()
        mock_pooch_retrieve.return_value = str(extracted_model_dir)

        # Test download
        download_pretrained_model(model_name="test_model", cache_dir=tmp_path)

        # Verify processor selection
        call_args = mock_pooch_retrieve.call_args
        processor = call_args[1]["processor"]

        # Should use Untar for .tar.gz files
        assert processor.__class__.__name__ == "Untar"

    def test_download_pretrained_model_zip_processor_selection(
        self, mock_pooch_retrieve, tmp_path
    ):
        """Test that Unzip processor is selected for .zip files."""
        # Create the expected directory structure
        extracted_model_dir = tmp_path / "extracted_model"
        extracted_model_dir.mkdir()
        mock_pooch_retrieve.return_value = str(extracted_model_dir)

        # Test download with .zip URL
        download_pretrained_model(
            model_url="https://example.com/model.zip", cache_dir=tmp_path
        )

        # Verify processor selection
        call_args = mock_pooch_retrieve.call_args
        processor = call_args[1]["processor"]

        # Should use Unzip for .zip files
        assert processor.__class__.__name__ == "Unzip"

    def test_download_pretrained_model_returns_list_handling(
        self, mock_pooch_retrieve, mock_get_pretrained_model_info, tmp_path
    ):
        """Test handling when pooch.retrieve returns a list of files."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.url = "https://example.com/model.zip"
        mock_get_pretrained_model_info.return_value = mock_model_info

        # Create the expected directory structure
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "file1.txt").write_text("test1")
        (model_dir / "file2.txt").write_text("test2")

        # Make pooch.retrieve return a list
        extracted_files = [str(model_dir / "file1.txt"), str(model_dir / "file2.txt")]
        mock_pooch_retrieve.return_value = extracted_files

        # Test download
        result_path = download_pretrained_model(
            model_name="test_model", cache_dir=tmp_path
        )

        # Should return the directory containing the files
        assert result_path == model_dir

    def test_download_pretrained_model_normalize_filenames(
        self, mock_pooch_retrieve, mock_get_pretrained_model_info, tmp_path
    ):
        """Test that model filenames are normalized after download."""
        # Setup mocks
        mock_model_info = Mock()
        mock_model_info.url = "https://example.com/model.zip"
        mock_get_pretrained_model_info.return_value = mock_model_info

        # Create the expected directory structure
        extracted_model_dir = tmp_path / "extracted_model"
        extracted_model_dir.mkdir()
        mock_pooch_retrieve.return_value = str(extracted_model_dir)

        # Mock the normalization function
        with patch(
            "scxpand.pretrained.download_manager._normalize_model_filenames"
        ) as mock_normalize:
            # Test download
            download_pretrained_model(model_name="test_model", cache_dir=tmp_path)

            # Verify normalization was called
            mock_normalize.assert_called_once_with(extracted_model_dir)


class TestPretrainedModelsRegistry:
    """Tests for the PRETRAINED_MODELS registry."""

    def test_pretrained_models_registry_structure(self):
        """Test that the registry has the expected structure."""
        assert isinstance(PRETRAINED_MODELS, dict)
        assert len(PRETRAINED_MODELS) > 0

        # Check that all entries are PretrainedModelInfo objects
        for model_name, model_info in PRETRAINED_MODELS.items():
            assert isinstance(model_name, str)
            assert isinstance(model_info, PretrainedModelInfo)
            assert model_info.name == model_name
            assert model_info.version is not None
            assert model_info.url is not None

    def test_pretrained_models_registry_expected_models(self):
        """Test that the registry contains expected model types."""
        expected_models = [
            "pan_cancer_autoencoder",
            "pan_cancer_mlp",
            "pan_cancer_lightgbm",
            "pan_cancer_logistic",
            "pan_cancer_svm",
        ]

        for model_name in expected_models:
            assert model_name in PRETRAINED_MODELS
            model_info = PRETRAINED_MODELS[model_name]
            assert model_info.version == "1.0.0"
            assert "figshare.com" in model_info.url

    def test_pretrained_models_registry_urls_valid(self):
        """Test that all registry URLs are valid."""
        for _model_name, model_info in PRETRAINED_MODELS.items():
            assert model_info.url.startswith("http")
            assert "figshare.com" in model_info.url
            assert "ndownloader" in model_info.url
