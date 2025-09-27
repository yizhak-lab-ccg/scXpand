import numpy as np
import pytest
import torch

from scxpand.autoencoders.ae_models import AutoencoderModel, ForkAutoencoder
from scxpand.autoencoders.ae_modules import Decoder
from scxpand.data_util.data_format import DataFormat


class TestDecoderTransforms:
    """Test cases for decoder's inverse transform functionality."""

    @pytest.fixture
    def simple_data_format(self):
        """Create a simple DataFormat for testing."""
        return DataFormat(
            n_genes=50,
            genes_mu=np.random.randn(50),
            genes_sigma=np.random.rand(50) + 0.1,  # Ensure positive
            eps=1e-4,
            use_log_transform=False,
            target_sum=1e4,
        )

    @pytest.fixture
    def log_transform_data_format(self):
        """Create a DataFormat with log transform enabled."""
        return DataFormat(
            n_genes=50,
            genes_mu=np.random.randn(50),
            genes_sigma=np.random.rand(50) + 0.1,
            eps=1e-4,
            use_log_transform=True,
            target_sum=1e4,
        )

    @pytest.fixture
    def sample_latent(self):
        """Create sample latent vectors."""
        return torch.randn(3, 16)

    def test_decoder_applies_mean_activation(self, simple_data_format, sample_latent):
        """Test that decoder applies mean_activation and inverse transforms."""
        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=False,
            needs_theta=False,
            data_format=simple_data_format,
        )

        output = decoder(sample_latent)

        # mu should be non-negative (result of mean_activation + inverse transforms)
        assert torch.all(output.mu >= 0), (
            "mu should be non-negative after inverse transforms"
        )
        assert torch.all(torch.isfinite(output.mu)), "mu should be finite"

    def test_decoder_applies_theta_activation(self, simple_data_format, sample_latent):
        """Test that decoder applies theta_activation for positive dispersion."""
        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=False,
            needs_theta=True,
            data_format=simple_data_format,
        )

        output = decoder(sample_latent)

        # theta should be positive (result of theta_activation)
        assert output.theta is not None
        assert torch.all(output.theta > 0), (
            "theta should be positive after theta_activation"
        )
        assert torch.all(output.theta >= 1e-4), "theta should respect minimum bound"
        assert torch.all(output.theta <= 1e4), "theta should respect maximum bound"

    def test_decoder_applies_dropout_activation(
        self, simple_data_format, sample_latent
    ):
        """Test that decoder applies dropout_activation for pi probabilities."""
        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=True,
            needs_theta=False,
            data_format=simple_data_format,
        )

        output = decoder(sample_latent)

        # pi should be in [0,1] (result of dropout_activation)
        assert output.pi is not None
        assert torch.all(output.pi >= 0), "pi should be >= 0"
        assert torch.all(output.pi <= 1), "pi should be <= 1"

    def test_decoder_inverse_transforms_applied(
        self, simple_data_format, sample_latent
    ):
        """Test that decoder applies inverse transforms."""
        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=False,
            needs_theta=False,
            data_format=simple_data_format,
        )

        output = decoder(sample_latent)

        # mu should exist and be non-negative (after inverse transforms)
        assert output.mu is not None
        assert torch.all(output.mu >= 0), (
            "mu should be non-negative after inverse transforms"
        )

    def test_decoder_with_log_transform(self, log_transform_data_format, sample_latent):
        """Test decoder with log transform in data format."""
        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=False,
            needs_theta=False,
            data_format=log_transform_data_format,
        )

        output = decoder(sample_latent)

        # Should still produce valid outputs
        assert torch.all(output.mu >= 0), (
            "mu should be non-negative after inverse transforms"
        )

    def test_decoder_conditional_heads(self, simple_data_format, sample_latent):
        """Test decoder with different combinations of pi and theta heads."""
        test_cases = [
            (True, True),  # Both pi and theta
            (True, False),  # Only pi
            (False, True),  # Only theta
            (False, False),  # Neither
        ]

        for needs_pi, needs_theta in test_cases:
            decoder = Decoder(
                latent_dim=16,
                hidden_dims=(32,),
                n_genes=50,
                needs_pi=needs_pi,
                needs_theta=needs_theta,
                data_format=simple_data_format,
            )

            output = decoder(sample_latent)

            # Always should have mu
            assert output.mu is not None
            assert torch.all(output.mu >= 0)

            # Check conditional outputs
            if needs_pi:
                assert output.pi is not None
                assert torch.all(output.pi >= 0) and torch.all(output.pi <= 1)
            else:
                assert output.pi is None

            if needs_theta:
                assert output.theta is not None
                assert torch.all(output.theta > 0)
            else:
                assert output.theta is None

    def test_autoencoder_model_integration(self, simple_data_format):
        """Test that AutoencoderModel properly integrates decoder transforms."""
        model = AutoencoderModel(
            data_format=simple_data_format,
            latent_dim=16,
            encoder_hidden_dims=(32,),
            decoder_hidden_dims=(32,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
            needs_pi=True,
            needs_theta=True,
        )

        sample_input = torch.randn(3, 50).abs() + 0.1  # Positive input
        model_output = model.forward(sample_input)

        # Test all outputs are valid
        assert torch.all(model_output.mu >= 0), "mu should be non-negative"
        assert torch.all(model_output.theta > 0), "theta should be positive"
        assert torch.all(model_output.pi >= 0) and torch.all(model_output.pi <= 1), (
            "pi should be in [0,1]"
        )

    def test_fork_autoencoder_integration(self, simple_data_format):
        """Test that ForkAutoencoder properly integrates decoder transforms."""
        model = ForkAutoencoder(
            data_format=simple_data_format,
            latent_dim=16,
            encoder_hidden_dims=(32,),
            decoder_hidden_dims=(32,),
            classifier_hidden_dims=(16,),
            dropout_rate=0.3,
            needs_pi=True,
            needs_theta=True,
        )

        sample_input = torch.randn(3, 50).abs() + 0.1  # Positive input
        model_output = model.forward(sample_input)

        # Test all outputs are valid
        assert torch.all(model_output.mu >= 0), "mu should be non-negative"
        assert torch.all(model_output.theta > 0), "theta should be positive"
        assert torch.all(model_output.pi >= 0) and torch.all(model_output.pi <= 1), (
            "pi should be in [0,1]"
        )

    def test_transforms_preserve_batch_and_gene_dimensions(self, simple_data_format):
        """Test that transforms preserve tensor dimensions."""
        batch_size, n_genes = 4, 50
        sample_latent = torch.randn(batch_size, 16)

        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=n_genes,
            needs_pi=True,
            needs_theta=True,
            data_format=simple_data_format,
        )

        output = decoder(sample_latent)

        # All outputs should have correct shape
        expected_shape = (batch_size, n_genes)
        assert output.mu.shape == expected_shape
        assert output.pi.shape == expected_shape
        assert output.theta.shape == expected_shape

    def test_gradient_flow_through_transforms(self, simple_data_format):
        """Test that gradients flow through decoder transforms."""
        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=True,
            needs_theta=True,
            data_format=simple_data_format,
        )

        sample_latent = torch.randn(3, 16, requires_grad=True)
        output = decoder(sample_latent)

        # Compute loss from all outputs
        loss = output.mu.sum() + output.pi.sum() + output.theta.sum()

        loss.backward()

        # Gradients should flow back to input
        assert sample_latent.grad is not None
        assert not torch.allclose(
            sample_latent.grad, torch.zeros_like(sample_latent.grad)
        )

    def test_numerical_stability_with_extreme_inputs(self, simple_data_format):
        """Test decoder stability with extreme input values."""
        extreme_inputs = [
            torch.ones(2, 16) * 100,  # Large positive
            torch.ones(2, 16) * -100,  # Large negative
            torch.zeros(2, 16),  # Zero
            torch.ones(2, 16) * 1e-8,  # Very small positive
        ]

        decoder = Decoder(
            latent_dim=16,
            hidden_dims=(32,),
            n_genes=50,
            needs_pi=True,
            needs_theta=True,
            data_format=simple_data_format,
        )

        for latent_input in extreme_inputs:
            output = decoder(latent_input)

            # All outputs should be finite
            assert torch.all(torch.isfinite(output.mu)), "mu should be finite"
            assert torch.all(torch.isfinite(output.pi)), "pi should be finite"
            assert torch.all(torch.isfinite(output.theta)), "theta should be finite"

            # Outputs should satisfy constraints
            assert torch.all(output.mu >= 0), "mu should be non-negative"
            assert torch.all(output.theta > 0), "theta should be positive"
            assert torch.all(output.pi >= 0) and torch.all(output.pi <= 1), (
                "pi should be in [0,1]"
            )
