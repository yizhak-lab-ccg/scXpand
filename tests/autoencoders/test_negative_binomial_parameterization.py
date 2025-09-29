import pytest
import torch
from torch.distributions import NegativeBinomial

from scxpand.autoencoders.ae_modules import theta_activation


class TestNegativeBinomialParameterization:
    """Test cases for negative binomial distribution parameterization."""

    def test_theta_activation_produces_positive_dispersion(self):
        """Test that theta_activation produces positive dispersion parameters."""
        # Test various input ranges
        test_inputs = [
            torch.randn(100, 50),  # Random values
            torch.ones(10, 20) * -10,  # Large negative values
            torch.ones(10, 20) * 10,  # Large positive values
            torch.zeros(5, 10),  # Zero values
            torch.linspace(-5, 5, 100).unsqueeze(0).repeat(10, 1),  # Linear range
        ]

        for raw_theta in test_inputs:
            theta = theta_activation(raw_theta)

            # All theta values should be positive
            assert torch.all(theta > 0), "All theta values should be positive"
            assert torch.all(theta >= 1e-4), "Theta should respect minimum bound"
            assert torch.all(theta <= 1e4), "Theta should respect maximum bound"

            # Should be finite
            assert torch.all(torch.isfinite(theta)), "All theta values should be finite"

    def test_negative_binomial_distribution_compatibility(self):
        """Test that our theta values work with PyTorch's NegativeBinomial distribution."""
        # Create sample mean and theta values
        mean_vals = torch.rand(5, 10) * 100 + 1  # Positive means
        raw_theta = torch.randn(5, 10)
        theta_vals = theta_activation(raw_theta)

        # Convert to PyTorch NegativeBinomial parameterization
        # PyTorch uses total_count (r) and logits parameterization is more stable
        # For mean m and dispersion r: logits = log(m/r)
        total_count = theta_vals
        logits = torch.log(mean_vals / theta_vals)

        # Create distribution (should not raise errors)
        try:
            dist = NegativeBinomial(total_count=total_count, logits=logits)

            # Verify mean calculation
            computed_mean = dist.mean
            assert torch.allclose(computed_mean, mean_vals, atol=1e-5), (
                "Distribution mean should match input mean"
            )

        except Exception as e:
            pytest.fail(f"NegativeBinomial distribution creation failed: {e}")

    def test_negative_binomial_variance_formula(self):
        """Test that variance follows negative binomial formula."""
        mean_vals = torch.rand(3, 5) * 50 + 5  # Positive means
        raw_theta = torch.randn(3, 5)
        theta_vals = theta_activation(raw_theta)

        # In negative binomial: variance = mean + mean^2/r
        # where r is the dispersion parameter (our theta)
        expected_variance = mean_vals + (mean_vals**2) / theta_vals

        # Create PyTorch distribution
        total_count = theta_vals
        logits = torch.log(mean_vals / theta_vals)
        dist = NegativeBinomial(total_count=total_count, logits=logits)

        computed_variance = dist.variance

        assert torch.allclose(computed_variance, expected_variance, atol=1e-5), (
            "Variance should follow negative binomial formula"
        )

    def test_overdispersion_behavior(self):
        """Test that smaller theta leads to higher overdispersion."""
        mean_val = 10.0

        # Test different theta values
        small_theta = theta_activation(torch.tensor([-5.0]))  # Small positive theta
        large_theta = theta_activation(torch.tensor([5.0]))  # Large positive theta

        # Calculate variances for same mean but different theta
        var_small = mean_val + (mean_val**2) / small_theta
        var_large = mean_val + (mean_val**2) / large_theta

        # Smaller theta should lead to larger variance (more overdispersion)
        assert var_small > var_large, (
            "Smaller theta should lead to higher variance (overdispersion)"
        )

        # Both should be greater than Poisson variance (mean)
        assert var_small > mean_val, "NB variance should exceed Poisson variance"
        assert var_large > mean_val, "NB variance should exceed Poisson variance"

    def test_theta_activation_monotonicity(self):
        """Test that theta_activation is monotonically increasing."""
        x_vals = torch.linspace(-10, 10, 1000)
        theta_vals = theta_activation(x_vals)

        # Check monotonicity
        diffs = torch.diff(theta_vals)
        assert torch.all(diffs >= 0), (
            "theta_activation should be monotonically increasing"
        )

    def test_theta_activation_asymptotic_behavior(self):
        """Test asymptotic behavior of theta_activation."""
        # For large positive x, softplus(x) ≈ x (but we clamp to max 1e4)
        large_x = torch.tensor([5.0, 8.0, 10.0])  # Use smaller values due to clamping
        large_theta = theta_activation(large_x)

        # Should approximate input for moderately large positive values
        assert torch.allclose(large_theta, large_x, atol=0.01), (
            "theta_activation should approximate input for moderately large positive values"
        )

        # For large negative x, softplus(x) ≈ 0 (but we clamp to minimum)
        small_x = torch.tensor([-20.0, -15.0, -10.0])
        small_theta = theta_activation(small_x)

        # Should be close to minimum bound
        assert torch.all(small_theta <= 0.01), (
            "theta_activation should approach minimum for large negative values"
        )
        assert torch.all(small_theta >= 1e-4), (
            "theta_activation should respect minimum bound"
        )

    def test_realistic_parameter_ranges(self):
        """Test with realistic parameter ranges for single-cell data."""
        # Realistic mean expression ranges (log-scale normalized counts)
        mean_ranges = [
            torch.logspace(-2, 2, 50),  # 0.01 to 100
            torch.linspace(0.1, 50, 100),  # Linear scale
        ]

        # Realistic theta ranges (dispersion parameters)
        theta_raw = torch.linspace(-3, 3, 50)  # Will be transformed to positive

        for mean_vals in mean_ranges:
            theta_vals = theta_activation(theta_raw)

            # Test with subset of combinations to avoid memory issues
            n_samples = min(10, len(mean_vals), len(theta_vals))
            mean_sample = mean_vals[:n_samples]
            theta_sample = theta_vals[:n_samples]

            # Test each combination
            total_count = theta_sample
            logits = torch.log(mean_sample / theta_sample)

            # All parameters should be valid
            assert torch.all(total_count > 0), (
                "All total_count parameters should be positive"
            )

            # Test that we can create distributions
            try:
                dist = NegativeBinomial(total_count=total_count, logits=logits)
                computed_mean = dist.mean

                # Means should match (within numerical precision)
                assert torch.allclose(computed_mean, mean_sample, atol=1e-4), (
                    "Distribution means should match input means"
                )

            except Exception as e:
                pytest.fail(f"Failed with realistic parameters: {e}")

    def test_gradient_computation_through_theta(self):
        """Test that gradients can flow through theta_activation."""
        # Create input that requires gradients
        raw_theta = torch.randn(5, 10, requires_grad=True)
        mean_vals = torch.rand(5, 10) * 10 + 1

        # Apply activation
        theta_vals = theta_activation(raw_theta)

        # Compute negative binomial log probability
        total_count = theta_vals
        logits = torch.log(mean_vals / theta_vals)

        # Create distribution and compute log prob for some sample data
        dist = NegativeBinomial(total_count=total_count, logits=logits)
        sample_data = torch.randint(0, 20, (5, 10)).float()

        log_prob = dist.log_prob(sample_data)
        loss = -log_prob.sum()  # Negative log likelihood

        # Backpropagate
        loss.backward()

        # Check that gradients were computed
        assert raw_theta.grad is not None, "Gradients should be computed"
        assert not torch.allclose(raw_theta.grad, torch.zeros_like(raw_theta.grad)), (
            "Gradients should be non-zero"
        )

    def test_comparison_with_poisson_baseline(self):
        """Test that NB reduces to Poisson as theta approaches large values."""
        mean_val = 5.0

        # Use maximum allowable theta from our activation function
        large_theta = theta_activation(
            torch.tensor([15.0])
        )  # Large theta within bounds

        # NB variance: mean + mean^2/theta
        nb_variance = mean_val + (mean_val**2) / large_theta
        poisson_variance = mean_val

        # With large theta, NB variance should be closer to Poisson variance
        # But due to our clamping, allow for reasonable tolerance
        relative_error = torch.abs(nb_variance - poisson_variance) / poisson_variance
        assert relative_error < 0.5, (
            f"Large theta should reduce overdispersion, got relative error: {relative_error}"
        )
