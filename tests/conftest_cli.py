"""Pytest configuration for CLI tests."""

import pytest


def pytest_configure(config):
    """Configure pytest markers for CLI tests."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Mark integration tests as slow
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.integration)

        # Mark subprocess tests as slow
        if "subprocess" in str(item.function.__code__.co_names):
            item.add_marker(pytest.mark.slow)
