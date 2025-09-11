"""Tests for linear classifier parameters."""

from scxpand.linear.linear_params import LinearClassifierParam


class TestLinearClassifierParam:
    """Test the LinearClassifierParam dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        prm = LinearClassifierParam()

        assert prm.use_log_transform is True
        assert prm.model_type == "logistic"
        assert prm.alpha == 0.0001  # scikit-learn SGDClassifier default
        assert prm.penalty == "l2"
        assert prm.n_epochs == 1000
        assert prm.class_weight is None  # scikit-learn SGDClassifier default
        assert prm.tol == 1e-3  # scikit-learn SGDClassifier default
        assert prm.l1_ratio == 0.15
        assert prm.random_seed == 42
        assert prm.batch_size == 2048
        assert prm.early_stopping_patience == 5
        assert prm.eval_interval == 1

    def test_custom_parameters(self):
        """Test custom parameter values."""
        prm = LinearClassifierParam(
            model_type="svm",
            alpha=0.5,
            penalty="l1",
            n_epochs=500,
            class_weight=None,
            random_seed=123,
            batch_size=512,
        )

        assert prm.model_type == "svm"
        assert prm.alpha == 0.5
        assert prm.penalty == "l1"
        assert prm.n_epochs == 500
        assert prm.class_weight is None
        assert prm.random_seed == 123
        assert prm.batch_size == 512

    def test_parameter_types(self):
        """Test parameter type validation."""
        prm = LinearClassifierParam()

        assert isinstance(prm.use_log_transform, bool)
        assert isinstance(prm.model_type, str)
        assert isinstance(prm.alpha, float)
        assert isinstance(prm.penalty, str)
        assert isinstance(prm.n_epochs, int)
        assert isinstance(prm.tol, float)
        assert isinstance(prm.l1_ratio, float)
        assert isinstance(prm.random_seed, int)
        assert isinstance(prm.batch_size, int)
        assert isinstance(prm.early_stopping_patience, int)
        assert isinstance(prm.eval_interval, int)
