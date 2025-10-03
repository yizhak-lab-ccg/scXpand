import torch

from scxpand.autoencoders.ae_modules import dropout_activation, theta_activation


class TestActivationFunctions:
    """Test cases for autoencoder activation functions."""

    def test_theta_activation_positive_output(self) -> None:
        """Test that theta_activation outputs positive values using softplus."""
        # Test with various input ranges
        inputs = [
            torch.randn(10, 20),  # Random values
            torch.ones(5, 10) * -10,  # Large negative values
            torch.ones(5, 10) * 10,  # Large positive values
            torch.zeros(3, 5),  # Zero values
        ]

        for x in inputs:
            result = theta_activation(x)
            assert torch.all(result > 0), (
                "theta_activation should always output positive values"
            )
            assert torch.all(result >= 1e-4), (
                "theta_activation should respect minimum bound"
            )
            assert torch.all(result <= 1e4), (
                "theta_activation should respect maximum bound"
            )

    def test_theta_activation_softplus_behavior(self) -> None:
        """Test that theta_activation behaves like clamped softplus."""
        x = torch.linspace(-10, 10, 100)
        result = theta_activation(x)

        # Should be monotonically increasing
        diffs = torch.diff(result)
        assert torch.all(diffs >= 0), (
            "theta_activation should be monotonically increasing"
        )

        # For large positive inputs, should approach x (softplus behavior)
        large_x = torch.tensor([5.0, 10.0])
        large_result = theta_activation(large_x)
        # softplus(x) ≈ x for large x
        assert torch.allclose(large_result, large_x, atol=0.1), (
            "Should approximate input for large positive values"
        )

    def test_theta_activation_numerical_stability(self) -> None:
        """Test theta_activation with extreme values."""
        # Test with very large values (should be clamped)
        extreme_pos = torch.ones(5, 10) * 100
        result_pos = theta_activation(extreme_pos)
        assert torch.all(torch.isfinite(result_pos)), (
            "Should handle extreme positive values"
        )
        assert torch.all(result_pos <= 1e4), "Should clamp large values"

        # Test with very negative values
        extreme_neg = torch.ones(5, 10) * -100
        result_neg = theta_activation(extreme_neg)
        assert torch.all(torch.isfinite(result_neg)), (
            "Should handle extreme negative values"
        )
        assert torch.all(result_neg >= 1e-4), "Should respect minimum bound"

    def test_dropout_activation_probability_output(self) -> None:
        """Test that dropout_activation outputs values in [0,1]."""
        # Test with various input ranges
        inputs = [
            torch.randn(10, 20),  # Random values
            torch.ones(5, 10) * -10,  # Large negative values
            torch.ones(5, 10) * 10,  # Large positive values
            torch.zeros(3, 5),  # Zero values
        ]

        for x in inputs:
            result = dropout_activation(x)
            assert torch.all(result >= 0), (
                "dropout_activation should output values >= 0"
            )
            assert torch.all(result <= 1), (
                "dropout_activation should output values <= 1"
            )

    def test_dropout_activation_sigmoid_behavior(self) -> None:
        """Test that dropout_activation behaves like sigmoid."""
        x = torch.linspace(-10, 10, 100)
        result = dropout_activation(x)

        # Should be monotonically increasing
        diffs = torch.diff(result)
        assert torch.all(diffs >= 0), (
            "dropout_activation should be monotonically increasing"
        )

        # Should approach 0 for large negative inputs
        neg_x = torch.tensor([-10.0, -5.0])
        neg_result = dropout_activation(neg_x)
        assert torch.all(neg_result < 0.1), (
            "Should approach 0 for large negative inputs"
        )

        # Should approach 1 for large positive inputs
        pos_x = torch.tensor([5.0, 10.0])
        pos_result = dropout_activation(pos_x)
        assert torch.all(pos_result > 0.9), (
            "Should approach 1 for large positive inputs"
        )

    def test_dropout_activation_numerical_stability(self) -> None:
        """Test dropout_activation with extreme values."""
        # Test with very large positive values
        extreme_pos = torch.ones(5, 10) * 100
        result_pos = dropout_activation(extreme_pos)
        assert torch.all(torch.isfinite(result_pos)), (
            "Should handle extreme positive values"
        )
        assert torch.all(result_pos <= 1), "Should not exceed 1"

        # Test with very large negative values
        extreme_neg = torch.ones(5, 10) * -100
        result_neg = dropout_activation(extreme_neg)
        assert torch.all(torch.isfinite(result_neg)), (
            "Should handle extreme negative values"
        )
        assert torch.all(result_neg >= 0), "Should not go below 0"

    def test_activation_functions_gradient_flow(self) -> None:
        """Test that activation functions allow gradient flow."""
        # Test with requires_grad=True
        x = torch.randn(5, 10, requires_grad=True)
        # Reset gradients
        x.grad = None

        # Theta activation
        theta_out = theta_activation(x)
        loss = theta_out.sum()
        loss.backward()
        assert x.grad is not None, "theta_activation should allow gradient flow"

        # Reset gradients
        x.grad = None

        # Dropout activation
        dropout_out = dropout_activation(x)
        loss = dropout_out.sum()
        loss.backward()
        assert x.grad is not None, "dropout_activation should allow gradient flow"

    def test_activation_functions_device_compatibility(self) -> None:
        """Test that activation functions work on different devices."""
        x_cpu = torch.randn(5, 10)

        # Test on CPU
        theta_cpu = theta_activation(x_cpu)
        dropout_cpu = dropout_activation(x_cpu)

        assert theta_cpu.device.type == "cpu"
        assert dropout_cpu.device.type == "cpu"

        # Test shapes are preserved
        assert theta_cpu.shape == x_cpu.shape
        assert dropout_cpu.shape == x_cpu.shape

    def test_activation_functions_dtype_preservation(self) -> None:
        """Test that activation functions preserve input dtype."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(5, 10, dtype=dtype)

            theta_out = theta_activation(x)
            dropout_out = dropout_activation(x)

            assert theta_out.dtype == dtype, f"theta_activation should preserve {dtype}"
            assert dropout_out.dtype == dtype, (
                f"dropout_activation should preserve {dtype}"
            )
