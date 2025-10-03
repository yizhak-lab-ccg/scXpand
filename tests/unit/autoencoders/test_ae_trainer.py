"""Tests for autoencoder trainer functionality."""

from scxpand.autoencoders.ae_params import AutoEncoderParams


class TestAutoEncoderTrainer:
    """Test autoencoder trainer functionality."""

    def test_autoencoder_params_initialization(self) -> None:
        """Test that AutoEncoderParams can be initialized."""
        params = AutoEncoderParams()

        # Test that required attributes exist
        assert hasattr(params, "latent_dim")
        assert hasattr(params, "encoder_hidden_dims")
        assert hasattr(params, "decoder_hidden_dims")
        assert hasattr(params, "classifier_hidden_dims")
        assert hasattr(params, "dropout_rate")
        assert hasattr(params, "n_epochs")
        assert hasattr(params, "early_stopping_patience")

    def test_autoencoder_params_default_values(self) -> None:
        """Test default values for AutoEncoderParams."""
        params = AutoEncoderParams()

        # Test default values
        assert params.latent_dim > 0, "latent_dim should be positive"
        assert isinstance(params.encoder_hidden_dims, tuple), (
            "encoder_hidden_dims should be tuple"
        )
        assert isinstance(params.decoder_hidden_dims, tuple), (
            "decoder_hidden_dims should be tuple"
        )
        assert isinstance(params.classifier_hidden_dims, tuple), (
            "classifier_hidden_dims should be tuple"
        )
        assert 0 <= params.dropout_rate <= 1, "dropout_rate should be between 0 and 1"
        assert params.n_epochs > 0, "n_epochs should be positive"
        assert params.early_stopping_patience > 0, (
            "early_stopping_patience should be positive"
        )

    def test_autoencoder_params_custom_values(self) -> None:
        """Test custom values for AutoEncoderParams."""
        custom_params = AutoEncoderParams(
            latent_dim=32,
            encoder_hidden_dims=(128, 64),
            decoder_hidden_dims=(64, 128),
            classifier_hidden_dims=(32,),
            dropout_rate=0.5,
            n_epochs=50,
            early_stopping_patience=10,
        )

        assert custom_params.latent_dim == 32
        assert custom_params.encoder_hidden_dims == (128, 64)
        assert custom_params.decoder_hidden_dims == (64, 128)
        assert custom_params.classifier_hidden_dims == (32,)
        assert custom_params.dropout_rate == 0.5
        assert custom_params.n_epochs == 50
        assert custom_params.early_stopping_patience == 10
