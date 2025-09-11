import numpy as np
import pytest
import torch

from scxpand.autoencoders.ae_model_output import DecoderOutput
from scxpand.autoencoders.ae_models import AutoencoderModel, ForkAutoencoder
from scxpand.data_util.data_format import DataFormat


class TestDecoderOutput:
    """Test cases for DecoderOutput class."""

    def test_decoder_output_creation(self):
        """Test creating DecoderOutput with all fields."""
        mu = torch.randn(5, 100).abs() + 1e-5  # Ensure positive
        pi = torch.sigmoid(torch.randn(5, 100))
        theta = torch.randn(5, 100).abs() + 1e-4  # Ensure positive

        output = DecoderOutput(mu=mu, pi=pi, theta=theta)

        assert torch.equal(output.mu, mu)
        assert torch.equal(output.pi, pi)
        assert torch.equal(output.theta, theta)

    def test_decoder_output_optional_fields(self):
        """Test creating DecoderOutput with optional fields as None."""
        mu = torch.randn(5, 100).abs() + 1e-5  # Ensure positive

        output = DecoderOutput(mu=mu, pi=None, theta=None)

        assert torch.equal(output.mu, mu)
        assert output.pi is None
        assert output.theta is None

    def test_decoder_output_device_property(self):
        """Test device property returns correct device."""
        mu = torch.randn(5, 100).abs() + 1e-5
        output = DecoderOutput(mu=mu)

        assert output.device == mu.device

    def test_decoder_output_to_device(self):
        """Test moving DecoderOutput to different device."""
        mu = torch.randn(5, 100).abs() + 1e-5
        pi = torch.sigmoid(torch.randn(5, 100))
        theta = torch.randn(5, 100).abs() + 1e-4

        output = DecoderOutput(mu=mu, pi=pi, theta=theta)

        # Test moving to same device (should work)
        output_same = output.to("cpu")
        assert output_same.device.type == "cpu"

    def test_decoder_output_detach(self):
        """Test detaching DecoderOutput from computation graph."""
        mu = torch.randn(5, 100, requires_grad=True).abs() + 1e-5
        pi = torch.sigmoid(torch.randn(5, 100, requires_grad=True))
        theta = torch.randn(5, 100, requires_grad=True).abs() + 1e-4

        output = DecoderOutput(mu=mu, pi=pi, theta=theta)
        detached_output = output.detach()

        assert not detached_output.mu.requires_grad
        assert not detached_output.pi.requires_grad
        assert not detached_output.theta.requires_grad


class TestAutoencoderModelOutput:
    """Test cases for autoencoder models returning DecoderOutput."""

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

    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        return torch.randn(5, 100)

    def test_autoencoder_model_decode_output_type(self, data_format, sample_input):
        """Test AutoencoderModel.decode returns DecoderOutput."""
        model = AutoencoderModel(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
        )
        latent = model.encode(sample_input)
        output = model.decode(latent)

        assert isinstance(output, DecoderOutput)
        assert output.mu.shape == (5, 100)
        assert output.pi is not None  # Should have pi by default
        assert output.theta is not None  # Should have theta by default

        # Test that mu and theta are positive (mu is now inverse-transformed but should still be non-negative)
        assert torch.all(output.mu >= 0), "mu should be non-negative"
        assert torch.all(output.theta > 0), "theta should be positive"

        # Test that pi is in [0,1]
        assert torch.all(output.pi >= 0) and torch.all(output.pi <= 1), "pi should be in [0,1]"

    def test_fork_autoencoder_decode_output_type(self, data_format, sample_input):
        """Test ForkAutoencoder.decode returns DecoderOutput."""
        model = ForkAutoencoder(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
        )
        latent = model.encode(sample_input)
        output = model.decode(latent)

        assert isinstance(output, DecoderOutput)
        assert output.mu.shape == (5, 100)
        assert output.pi is not None  # Should have pi by default
        assert output.theta is not None  # Should have theta by default

        # Test that mu and theta are positive (mu is now inverse-transformed but should still be non-negative)
        assert torch.all(output.mu >= 0), "mu should be non-negative"
        assert torch.all(output.theta > 0), "theta should be positive"

        # Test that pi is in [0,1]
        assert torch.all(output.pi >= 0) and torch.all(output.pi <= 1), "pi should be in [0,1]"

    def test_autoencoder_model_without_pi_theta(self, data_format, sample_input):
        """Test AutoencoderModel.decode with pi and theta disabled."""
        model = AutoencoderModel(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
            needs_pi=False,
            needs_theta=False,
        )
        latent = model.encode(sample_input)
        output = model.decode(latent)

        assert isinstance(output, DecoderOutput)
        assert output.mu.shape == (5, 100)
        assert output.pi is None
        assert output.theta is None

        # Test that mu is non-negative
        assert torch.all(output.mu >= 0), "mu should be non-negative"

    def test_fork_autoencoder_without_pi_theta(self, data_format, sample_input):
        """Test ForkAutoencoder.decode with pi and theta disabled."""
        model = ForkAutoencoder(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
            needs_pi=False,
            needs_theta=False,
        )
        latent = model.encode(sample_input)
        output = model.decode(latent)

        assert isinstance(output, DecoderOutput)
        assert output.mu.shape == (5, 100)
        assert output.pi is None
        assert output.theta is None

        # Test that mu is non-negative
        assert torch.all(output.mu >= 0), "mu should be non-negative"

    def test_model_forward_integration(self, data_format, sample_input):
        """Test that the full forward pass works with DecoderOutput."""
        model = AutoencoderModel(
            data_format=data_format,
            latent_dim=10,
            encoder_hidden_dims=(64,),
            decoder_hidden_dims=(64,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
        )
        model_output = model.forward(sample_input)

        # Check that ModelOutput contains the right fields from DecoderOutput
        assert model_output.mu.shape == (5, 100)
        assert model_output.pi is not None
        assert model_output.theta is not None
        assert model_output.latent_vec.shape == (5, 10)
        assert model_output.class_logit.shape == (5,)

        # Test that activation constraints are satisfied
        assert torch.all(model_output.mu >= 0), "mu should be non-negative"
        assert torch.all(model_output.theta > 0), "theta should be positive"
        assert torch.all(model_output.pi >= 0) and torch.all(model_output.pi <= 1), "pi should be in [0,1]"
