"""Tests for autoencoder decoder functionality."""

import numpy as np
import pytest
import torch

from scxpand.autoencoders.ae_models import AutoencoderModel, ForkAutoencoder
from scxpand.data_util.data_format import DataFormat


class TestDecoderFunctionality:
    """Test decoder functionality for autoencoder models."""

    @pytest.fixture
    def data_format(self):
        """Create a simple DataFormat for testing."""
        return DataFormat(
            n_genes=100,
            genes_mu=np.zeros(100),
            genes_sigma=np.ones(100),
            eps=1e-4,
            use_log_transform=False,
            target_sum=1e4,
        )

    def test_autoencoder_model_decode_output(self, data_format):
        """Test AutoencoderModel decode output type and structure."""
        model = AutoencoderModel(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
        )
        x = torch.randn(5, 100)
        latent = model.encode(x)
        output = model.decode(latent)

        # Test output type and structure
        assert hasattr(output, "mu"), "Output should have mu attribute"
        assert hasattr(output, "pi"), "Output should have pi attribute"
        assert hasattr(output, "theta"), "Output should have theta attribute"
        assert output.mu.shape == (5, 100), "Output mu should have correct shape"
        assert output.pi is not None, "Output pi should not be None"
        assert output.theta is not None, "Output theta should not be None"

    def test_fork_autoencoder_decode_output(self, data_format):
        """Test ForkAutoencoder decode output type and structure."""
        model = ForkAutoencoder(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
        )
        x = torch.randn(5, 100)
        latent = model.encode(x)
        output = model.decode(latent)

        # Test output type and structure
        assert hasattr(output, "mu"), "Output should have mu attribute"
        assert hasattr(output, "pi"), "Output should have pi attribute"
        assert hasattr(output, "theta"), "Output should have theta attribute"
        assert output.mu.shape == (5, 100), "Output mu should have correct shape"
        assert output.pi is not None, "Output pi should not be None"
        assert output.theta is not None, "Output theta should not be None"
