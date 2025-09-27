import pytest
import torch
from scipy.stats import nbinom

from scxpand.autoencoders.ae_losses import NB, MSELoss, ZINBLoss


class TestMSELoss:
    """Test the MSE loss implementation."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing MSE loss."""
        torch.manual_seed(42)
        batch_size, n_genes = 10, 5
        x_true = torch.rand(batch_size, n_genes) * 100
        x_pred = torch.rand(batch_size, n_genes) * 100
        return x_true, x_pred

    def test_mse_loss_basic(self, mock_data):
        """Test basic MSE loss computation."""
        x_true, x_pred = mock_data

        mse_loss_fn = MSELoss(eps=1e-8)
        loss = mse_loss_fn(x_genes_true=x_true, x_pred=x_pred)

        # Check that loss is finite and positive
        assert torch.isfinite(loss), "MSE loss should be finite"
        assert loss.item() >= 0, "MSE loss should be non-negative"

    def test_mse_loss_with_nans(self):
        """Test MSE loss with NaN values in input."""
        x_true = torch.tensor([[1.0, 2.0, float("nan")], [4.0, float("nan"), 6.0]])
        x_pred = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])

        mse_loss_fn = MSELoss(eps=1e-8)
        loss = mse_loss_fn(x_genes_true=x_true, x_pred=x_pred)

        # Should handle NaNs gracefully
        assert torch.isfinite(loss), "MSE loss should handle NaNs"

    def test_mse_loss_identical_inputs(self):
        """Test MSE loss when inputs are identical."""
        x = torch.rand(5, 3)

        mse_loss_fn = MSELoss(eps=1e-8)
        loss = mse_loss_fn(x_genes_true=x, x_pred=x)

        # Loss should be zero (or very close to zero) for identical inputs
        assert loss.item() < 1e-6, "MSE loss should be ~0 for identical inputs"

    def test_mse_better_reconstruction_lower_loss(self):
        """Test that better reconstruction gives lower MSE loss."""
        torch.manual_seed(42)
        x_true = torch.tensor([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]])

        # Good prediction (close to true values)
        x_pred_good = x_true + torch.randn_like(x_true) * 0.1  # Small noise

        # Bad prediction (far from true values)
        x_pred_bad = x_true + torch.randn_like(x_true) * 10.0  # Large noise

        mse_loss_fn = MSELoss(eps=1e-8)
        loss_good = mse_loss_fn(x_genes_true=x_true, x_pred=x_pred_good)
        loss_bad = mse_loss_fn(x_genes_true=x_true, x_pred=x_pred_bad)

        assert (
            loss_good.item() < loss_bad.item()
        ), "Better reconstruction should have lower MSE loss"


class TestNBLoss:
    """Test the Negative Binomial loss implementation."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing NB loss."""
        torch.manual_seed(42)
        batch_size, n_genes = 10, 5
        x_true = torch.randint(0, 100, (batch_size, n_genes)).float()
        x_pred = torch.rand(batch_size, n_genes) * 50 + 1  # means between 1-51
        theta = (
            torch.rand(batch_size, n_genes) * 10 + 0.1
        )  # dispersion between 0.1-10.1
        return x_true, x_pred, theta

    def test_nb_loss_basic(self, mock_data):
        """Test basic NB loss computation."""
        x_true, mu, theta = mock_data

        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        loss = nb_loss_fn(x_genes_true=x_true, mu=mu, theta=theta)

        assert torch.isfinite(loss), "NB loss should be finite"
        assert loss.item() > 0, "NB loss should be positive"

    def test_nb_loss_with_masking(self, mock_data):
        """Test NB loss with masking enabled."""
        x_true, mu, theta = mock_data
        # Add some NaN values
        x_true[0, 0] = float("nan")
        x_true[1, 2] = float("nan")

        nb_loss_fn = NB(use_masking=True, eps=1e-8)
        loss = nb_loss_fn(x_genes_true=x_true, mu=mu, theta=theta)

        assert torch.isfinite(loss), "NB loss should handle NaNs with masking"

    def test_nb_loss_theta_required(self):
        """Test that NB loss requires theta parameter."""
        x_true = torch.rand(5, 3)
        mu = torch.rand(5, 3)

        nb_loss_fn = NB(use_masking=False, eps=1e-8)

        with pytest.raises(TypeError):  # theta is now required parameter
            nb_loss_fn(x_genes_true=x_true, mu=mu)

    def test_nb_better_reconstruction_lower_loss(self):
        """Test that better reconstruction gives lower NB loss."""
        torch.manual_seed(42)
        # True counts (integers)
        x_true = torch.tensor([[5, 10, 15], [0, 8, 20]]).float()

        # Good prediction (close to true values)
        mu_good = torch.tensor([[5.1, 10.2, 14.8], [0.5, 8.1, 19.9]])

        # Bad prediction (far from true values)
        mu_bad = torch.tensor([[15.0, 2.0, 5.0], [25.0, 1.0, 5.0]])

        # Fixed theta for consistent comparison
        theta = torch.ones_like(x_true) * 2.0

        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        loss_good = nb_loss_fn(x_genes_true=x_true, mu=mu_good, theta=theta)
        loss_bad = nb_loss_fn(x_genes_true=x_true, mu=mu_bad, theta=theta)

        assert (
            loss_good.item() < loss_bad.item()
        ), "Better reconstruction should have lower NB loss"

    def test_nb_loss_against_scipy(self):
        """Test NB loss against scipy implementation if available."""
        try:
            # Single value test
            k = 5  # observed count
            mu = 3.0  # mean
            theta = 2.0  # dispersion

            # Convert to scipy parameterization: n=theta, p=theta/(theta+mu)
            n = theta
            p = theta / (theta + mu)

            scipy_logpmf = nbinom.logpmf(k, n, p)

            # Our implementation
            x_true = torch.tensor([[k]], dtype=torch.float32)
            mu = torch.tensor([[mu]], dtype=torch.float32)
            theta_tensor = torch.tensor([[theta]], dtype=torch.float32)

            nb_loss_fn = NB(use_masking=False, eps=1e-8)
            our_nll = nb_loss_fn(x_genes_true=x_true, mu=mu, theta=theta_tensor).item()

            # They should be close (within numerical precision)
            expected_nll = -scipy_logpmf
            assert (
                abs(our_nll - expected_nll) < 1e-4
            ), f"Mismatch: {our_nll} vs {expected_nll}"

        except ImportError:
            pytest.skip("Scipy not available for comparison")


class TestZINBLoss:
    """Test the Zero-Inflated Negative Binomial loss implementation."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing ZINB loss."""
        torch.manual_seed(42)
        batch_size, n_genes = 10, 5
        x_true = torch.randint(0, 100, (batch_size, n_genes)).float()
        # Add some zeros to test zero-inflation
        x_true[x_true < 20] = 0

        x_pred = torch.rand(batch_size, n_genes) * 50 + 1  # means between 1-51
        theta = (
            torch.rand(batch_size, n_genes) * 10 + 0.1
        )  # dispersion between 0.1-10.1
        pi = torch.rand(batch_size, n_genes) * 0.5  # zero-inflation probability 0-0.5
        return x_true, x_pred, theta, pi

    def test_zinb_loss_basic(self, mock_data):
        """Test basic ZINB loss computation."""
        x_true, mu, theta, pi = mock_data

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss = zinb_loss_fn(x_genes_true=x_true, mu=mu, pi=pi, theta=theta)

        assert torch.isfinite(loss), "ZINB loss should be finite"
        assert loss.item() > 0, "ZINB loss should be positive"

    def test_zinb_loss_all_zeros(self):
        """Test ZINB loss with all zero observations."""
        batch_size, n_genes = 5, 3
        x_true_zeros = torch.zeros(batch_size, n_genes)
        mu = torch.ones(batch_size, n_genes)
        theta = torch.ones(batch_size, n_genes)
        pi = torch.ones(batch_size, n_genes) * 0.5

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss = zinb_loss_fn(x_genes_true=x_true_zeros, mu=mu, pi=pi, theta=theta)

        assert torch.isfinite(loss), "ZINB loss with all zeros should be finite"

    def test_zinb_loss_high_counts(self):
        """Test ZINB loss with high count observations."""
        batch_size, n_genes = 5, 3
        x_true_high = torch.ones(batch_size, n_genes) * 1000
        mu = torch.ones(batch_size, n_genes) * 10
        theta = torch.ones(batch_size, n_genes)
        pi = torch.ones(batch_size, n_genes) * 0.1

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss = zinb_loss_fn(x_genes_true=x_true_high, mu=mu, pi=pi, theta=theta)

        assert torch.isfinite(loss), "ZINB loss with high counts should be finite"

    def test_zinb_loss_parameter_clamping(self, mock_data):
        """Test that ZINB loss properly clamps parameters."""
        x_true, x_pred, theta, pi = mock_data

        # Test with extreme values
        theta_extreme = torch.tensor([[1e-10, 1e10]])  # Very small and very large
        pi_extreme = torch.tensor([[-0.1, 1.1]])  # Outside [0, 1] range
        x_true_small = torch.tensor([[0, 5]])
        mu_small = torch.tensor([[1.0, 2.0]])

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss = zinb_loss_fn(
            x_genes_true=x_true_small, mu=mu_small, pi=pi_extreme, theta=theta_extreme
        )

        assert torch.isfinite(loss), "ZINB loss should handle extreme parameter values"

    def test_zinb_loss_with_nans(self):
        """Test ZINB loss with NaN values in observations."""
        x_true = torch.tensor([[1.0, float("nan"), 3.0], [0.0, 5.0, float("nan")]])
        mu = torch.tensor([[1.1, 2.1, 3.1], [0.1, 5.1, 6.1]])
        theta = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        pi = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss = zinb_loss_fn(x_genes_true=x_true, mu=mu, pi=pi, theta=theta)

        assert torch.isfinite(loss), "ZINB loss should handle NaNs gracefully"

    def test_zinb_loss_reduces_to_nb_when_pi_zero(self, mock_data):
        """Test that ZINB loss reduces to NB loss when pi=0."""
        x_true, x_pred, theta, _ = mock_data
        pi_zero = torch.zeros_like(theta)

        # ZINB loss with pi=0
        zinb_loss_fn = ZINBLoss(eps=1e-8)
        zinb_loss = zinb_loss_fn(
            x_genes_true=x_true, mu=x_pred, pi=pi_zero, theta=theta
        )

        # Regular NB loss
        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        nb_loss = nb_loss_fn(x_genes_true=x_true, mu=x_pred, theta=theta)

        # They should be very close (accounting for numerical precision)
        assert (
            abs(zinb_loss.item() - nb_loss.item()) < 1e-3
        ), "ZINB with pi=0 should equal NB loss"

    def test_zinb_better_reconstruction_lower_loss(self):
        """Test that better reconstruction gives lower ZINB loss."""
        torch.manual_seed(42)
        # True counts with zeros and non-zeros
        x_true = torch.tensor([[0, 5, 0], [10, 0, 15]]).float()

        # Good prediction (close to true values)
        x_pred_good = torch.tensor([[0.1, 5.2, 0.1], [9.8, 0.1, 15.1]])

        # Bad prediction (far from true values)
        x_pred_bad = torch.tensor([[20.0, 1.0, 25.0], [2.0, 30.0, 3.0]])

        # Fixed parameters for consistent comparison
        theta = torch.ones_like(x_true) * 2.0
        pi = torch.ones_like(x_true) * 0.1  # Low zero-inflation

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss_good = zinb_loss_fn(
            x_genes_true=x_true, mu=x_pred_good, pi=pi, theta=theta
        )
        loss_bad = zinb_loss_fn(x_genes_true=x_true, mu=x_pred_bad, pi=pi, theta=theta)

        assert (
            loss_good.item() < loss_bad.item()
        ), "Better reconstruction should have lower ZINB loss"

    def test_zinb_better_zero_inflation_parameter_lower_loss(self):
        """Test that better zero-inflation parameter gives lower ZINB loss."""
        torch.manual_seed(42)
        # True counts with many zeros
        x_true = torch.tensor([[0, 0, 5], [0, 8, 0]]).float()

        # Fixed predictions and theta
        x_pred = torch.tensor([[1.0, 1.0, 5.0], [1.0, 8.0, 1.0]])
        theta = torch.ones_like(x_true) * 2.0

        # Good pi (high zero-inflation where there are zeros)
        pi_good = torch.tensor([[0.8, 0.8, 0.1], [0.8, 0.1, 0.8]])

        # Bad pi (low zero-inflation where there are zeros)
        pi_bad = torch.tensor([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        loss_good = zinb_loss_fn(
            x_genes_true=x_true, mu=x_pred, pi=pi_good, theta=theta
        )
        loss_bad = zinb_loss_fn(x_genes_true=x_true, mu=x_pred, pi=pi_bad, theta=theta)

        assert (
            loss_good.item() < loss_bad.item()
        ), "Better zero-inflation parameter should have lower ZINB loss"


class TestLossComparisonEdgeCases:
    """Test edge cases and comparisons between different loss functions."""

    def test_loss_functions_handle_zero_predictions(self):
        """Test that all loss functions handle zero predictions gracefully."""
        x_true = torch.tensor([[0, 1, 5]])
        x_pred_zero = torch.tensor([[0.0, 0.0, 0.0]])  # Zero predictions
        theta = torch.tensor([[1.0, 1.0, 1.0]])
        pi = torch.tensor([[0.1, 0.1, 0.1]])

        # MSE should handle zero predictions
        mse_loss_fn = MSELoss(eps=1e-8)
        mse_loss = mse_loss_fn(x_genes_true=x_true, x_pred=x_pred_zero)
        assert torch.isfinite(mse_loss)

        # NB should handle zero predictions (with small epsilon)
        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        x_pred_small = torch.tensor([[1e-6, 1e-6, 1e-6]])  # Very small but non-zero
        nb_loss = nb_loss_fn(x_genes_true=x_true, mu=x_pred_small, theta=theta)
        assert torch.isfinite(nb_loss)

        # ZINB should handle zero predictions (with small epsilon)
        zinb_loss_fn = ZINBLoss(eps=1e-8)
        zinb_loss = zinb_loss_fn(
            x_genes_true=x_true, mu=x_pred_small, pi=pi, theta=theta
        )
        assert torch.isfinite(zinb_loss)

    def test_loss_functions_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create data with extreme count values
        x_true = torch.tensor([[0, 1, 10000]])  # Including very high count
        mu = torch.tensor([[1e-3, 1.0, 1000.0]])  # Wide range of predictions
        theta = torch.tensor([[0.01, 1.0, 100.0]])  # Wide range of dispersions
        pi = torch.tensor([[0.99, 0.5, 0.01]])  # Wide range of zero-inflation

        # All should handle extreme values
        mse_loss_fn = MSELoss(eps=1e-8)
        mse_loss = mse_loss_fn(x_genes_true=x_true.float(), x_pred=mu)
        assert torch.isfinite(mse_loss), "MSE should handle extreme values"

        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        nb_loss = nb_loss_fn(x_genes_true=x_true, mu=mu, theta=theta)
        assert torch.isfinite(nb_loss), "NB should handle extreme values"

        zinb_loss_fn = ZINBLoss(eps=1e-8)
        zinb_loss = zinb_loss_fn(x_genes_true=x_true, mu=mu, pi=pi, theta=theta)
        assert torch.isfinite(zinb_loss), "ZINB should handle extreme values"

    def test_loss_functions_gradient_flow(self):
        """Test that loss functions allow gradient flow."""
        torch.manual_seed(42)
        x_true = torch.randint(0, 10, (5, 3)).float()

        # Test MSE gradients
        x_pred_mse = torch.rand(5, 3, requires_grad=True)
        mse_loss_fn = MSELoss(eps=1e-8)
        mse_loss = mse_loss_fn(x_genes_true=x_true, x_pred=x_pred_mse)
        mse_loss.backward()
        assert x_pred_mse.grad is not None, "MSE should compute gradients"

        # Test NB gradients - create fresh tensors
        torch.manual_seed(42)  # Reset seed for reproducibility
        x_pred_nb = torch.rand(5, 3, requires_grad=True)
        theta_nb = (
            torch.rand(5, 3) * 2.0 + 0.5
        )  # Range 0.5-2.5, safely within clamp bounds
        theta_nb.requires_grad_(True)  # Make it a leaf tensor
        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        nb_loss = nb_loss_fn(x_genes_true=x_true, mu=x_pred_nb, theta=theta_nb)
        nb_loss.backward()
        assert x_pred_nb.grad is not None, "NB should compute gradients for x_pred"
        assert theta_nb.grad is not None, "NB should compute gradients for theta"

        # Test ZINB gradients - create fresh tensors
        torch.manual_seed(42)  # Reset seed for reproducibility
        x_pred_zinb = torch.rand(5, 3, requires_grad=True)
        theta_zinb = (
            torch.rand(5, 3) * 2.0 + 0.5
        )  # Range 0.5-2.5, safely within clamp bounds
        theta_zinb.requires_grad_(True)  # Make it a leaf tensor
        pi_zinb = torch.rand(5, 3) * 0.5
        pi_zinb.requires_grad_(True)  # Make it a leaf tensor
        zinb_loss_fn = ZINBLoss(eps=1e-8)
        zinb_loss = zinb_loss_fn(
            x_genes_true=x_true, mu=x_pred_zinb, pi=pi_zinb, theta=theta_zinb
        )
        zinb_loss.backward()
        assert x_pred_zinb.grad is not None, "ZINB should compute gradients for x_pred"
        assert theta_zinb.grad is not None, "ZINB should compute gradients for theta"
        assert pi_zinb.grad is not None, "ZINB should compute gradients for pi"

    def test_all_losses_monotonic_in_reconstruction_quality(self):
        """Test that all loss functions are monotonic in reconstruction quality."""
        torch.manual_seed(42)

        # Create a range of predictions from very good to very bad
        x_true = torch.tensor([[10.0, 0.0, 25.0]])

        # Create predictions with systematically increasing error
        predictions = [
            x_true + 0.1,  # Very close (best)
            x_true + 1.0,  # Close
            x_true + 5.0,  # Far
            x_true + 10.0,  # Very far (worst)
        ]

        # Ensure all predictions are positive for NB/ZINB
        predictions = [torch.clamp(pred, min=0.001) for pred in predictions]

        # Fixed parameters
        theta = torch.ones_like(x_true) * 2.0
        pi = torch.ones_like(x_true) * 0.1

        # Test MSE - should be perfectly monotonic since MSE is just squared distance
        mse_loss_fn = MSELoss(eps=1e-8)
        mse_losses = [
            mse_loss_fn(x_genes_true=x_true, x_pred=pred).item() for pred in predictions
        ]

        # MSE should increase monotonically
        for i in range(len(mse_losses) - 1):
            assert (
                mse_losses[i] < mse_losses[i + 1]
            ), f"MSE should increase monotonically: {mse_losses}"

        # Test NB - compare best vs worst
        nb_loss_fn = NB(use_masking=False, eps=1e-8)
        nb_loss_best = nb_loss_fn(
            x_genes_true=x_true, mu=predictions[0], theta=theta
        ).item()
        nb_loss_worst = nb_loss_fn(
            x_genes_true=x_true, mu=predictions[-1], theta=theta
        ).item()

        assert (
            nb_loss_best < nb_loss_worst
        ), f"NB: best ({nb_loss_best}) should be < worst ({nb_loss_worst})"

        # Test ZINB - compare best vs worst
        zinb_loss_fn = ZINBLoss(eps=1e-8)
        zinb_loss_best = zinb_loss_fn(
            x_genes_true=x_true, mu=predictions[0], pi=pi, theta=theta
        ).item()
        zinb_loss_worst = zinb_loss_fn(
            x_genes_true=x_true, mu=predictions[-1], pi=pi, theta=theta
        ).item()

        assert (
            zinb_loss_best < zinb_loss_worst
        ), f"ZINB: best ({zinb_loss_best}) should be < worst ({zinb_loss_worst})"
