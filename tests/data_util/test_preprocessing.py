import numpy as np
import pytest
import torch

from scxpand.data_util.transforms import (
    apply_inverse_zscore_normalization,
    apply_row_normalization,
    apply_zscore_normalization,
)


class TestZScoreNormalizationDense:
    def test_dense_normalization_in_place(self):
        X_dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        X_original = X_dense.copy()
        mu = np.array([1.0, 2.0], dtype=np.float32)
        sigma = np.array([1.0, 1.0], dtype=np.float32)

        # Should modify in-place and return the result
        res = apply_zscore_normalization(X_dense, mu, sigma, in_place=True)
        assert res is X_dense  # Should return the same array when in_place=True

        # Check that the operation was performed correctly
        expected = (X_original - mu) / (sigma + 1e-6)
        np.testing.assert_allclose(X_dense, expected, rtol=1e-6, atol=1e-6)

    def test_round_trip_inverse(self):
        rng = np.random.default_rng(0)
        X_original = rng.random((5, 3)).astype(np.float32)
        X_dense = X_original.copy()
        mu = rng.random(3).astype(np.float32)
        sigma = rng.random(3).astype(np.float32) + 0.5

        # Apply z-score normalization
        apply_zscore_normalization(X_dense, mu, sigma, in_place=True)

        # Convert to tensor and invert
        z_tensor = torch.from_numpy(X_dense)
        inv = apply_inverse_zscore_normalization(X=z_tensor, genes_mu=mu, genes_sigma=sigma)
        np.testing.assert_allclose(inv.numpy(), X_original, rtol=1e-5, atol=1e-5)


class TestRowNormalization:
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_row_normalization(self, backend):
        X = np.array([[1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        if backend == "torch":
            X_in = torch.from_numpy(X.copy())
        else:
            X_in = X.copy()

        target_sum = 10.0
        out = apply_row_normalization(X_in, target_sum=target_sum)

        if backend == "torch":
            out_np = out.numpy()
        else:
            out_np = out

        row_sums = out_np.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.full_like(row_sums, target_sum), atol=1e-6)
