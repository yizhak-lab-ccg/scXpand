"""Test error handling in hyperparameter optimization."""

from scxpand.hyperopt.hyperopt_optimizer import CATCHABLE_EXCEPTIONS, CRITICAL_ERRORS


class TestErrorHandling:
    """Test error handling in hyperparameter optimization."""

    def test_error_categorization(self):
        """Test that errors are categorized correctly for trial failure vs continuation."""
        # Verify specific error types are classified correctly
        assert MemoryError in CRITICAL_ERRORS, "MemoryError should cause trial failure"
        assert (
            RuntimeError in CRITICAL_ERRORS
        ), "RuntimeError should cause trial failure"
        assert ValueError in CRITICAL_ERRORS, "ValueError should cause trial failure"
        assert (
            FileNotFoundError in CRITICAL_ERRORS
        ), "FileNotFoundError should cause trial failure"

        assert (
            ConnectionError in CATCHABLE_EXCEPTIONS
        ), "ConnectionError should be recoverable"
        assert (
            TimeoutError in CATCHABLE_EXCEPTIONS
        ), "TimeoutError should be recoverable"
        assert ImportError in CATCHABLE_EXCEPTIONS, "ImportError should be recoverable"
        assert OSError in CATCHABLE_EXCEPTIONS, "OSError should be recoverable"

    def test_error_handling_design(self):
        """Test the design principles of error handling."""
        # Verify that critical scenarios are properly categorized
        critical_error_types = [
            MemoryError,
            RuntimeError,
            ValueError,
            FileNotFoundError,
        ]
        for error_type in critical_error_types:
            assert (
                error_type in CRITICAL_ERRORS
            ), f"{error_type.__name__} should be critical"

        # Verify that recoverable scenarios are properly categorized
        recoverable_error_types = [ConnectionError, TimeoutError, ImportError, OSError]
        for error_type in recoverable_error_types:
            assert (
                error_type in CATCHABLE_EXCEPTIONS
            ), f"{error_type.__name__} should be recoverable"

    def test_out_of_memory_handling(self):
        """Test specific handling of out-of-memory scenarios."""
        # These are the types of memory errors that should cause trial failure
        memory_error_scenarios = [
            "CUDA out of memory",
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "torch.cuda.OutOfMemoryError",
            "RuntimeError: [enforce fail at alloc_cpu.cpp:39]",
        ]

        # These should all be caught by MemoryError or RuntimeError
        for scenario in memory_error_scenarios:
            if "CUDA out of memory" in scenario or "OutOfMemoryError" in scenario:
                # Usually manifests as RuntimeError in PyTorch
                assert RuntimeError in CRITICAL_ERRORS
            elif "alloc_cpu.cpp" in scenario:
                # CPU memory allocation failures
                assert RuntimeError in CRITICAL_ERRORS

    def test_error_types_are_exceptions(self):
        """Test that all error types are actually exception classes."""
        for error_type in CRITICAL_ERRORS:
            assert issubclass(
                error_type, Exception
            ), f"{error_type} should be an Exception subclass"

        for error_type in CATCHABLE_EXCEPTIONS:
            assert issubclass(
                error_type, Exception
            ), f"{error_type} should be an Exception subclass"

    def test_no_overlap_between_error_categories(self):
        """Test that critical and catchable errors don't overlap."""
        overlap = set(CRITICAL_ERRORS) & set(CATCHABLE_EXCEPTIONS)
        assert len(overlap) == 0, f"Error categories should not overlap: {overlap}"
