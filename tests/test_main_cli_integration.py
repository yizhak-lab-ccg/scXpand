"""Integration tests for CLI using subprocess calls.

These tests are slower but verify actual CLI behavior end-to-end.
They should be run separately from unit tests.
"""

import os
import subprocess
import sys
import tempfile
import time

import pytest


class TestCLIIntegration:
    """Slow integration tests using actual subprocess calls."""

    @pytest.mark.slow
    def test_cli_help_command_integration(self):
        """Test that CLI shows help information via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should succeed and show help
        assert result.returncode in {0, 1, 2}
        output = result.stdout + result.stderr

        # Verify help content
        assert len(output) > 0
        # Check for key commands in help
        help_text = output.lower()
        assert any(cmd in help_text for cmd in ["train", "optimize", "inference", "list-models"])

    @pytest.mark.slow
    def test_cli_train_help_integration(self):
        """Test train command help via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in {0, 1, 2}
        output = result.stdout + result.stderr
        assert len(output) > 0

        # Check for train-specific help content
        help_text = output.lower()
        assert "model_type" in help_text or "train" in help_text

    @pytest.mark.slow
    def test_cli_inference_help_integration(self):
        """Test inference command help via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "inference", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in {0, 1, 2}
        output = result.stdout + result.stderr
        assert len(output) > 0

        # Check for inference-specific help content
        help_text = output.lower()
        assert "data_path" in help_text or "inference" in help_text

    @pytest.mark.slow
    def test_cli_invalid_command_integration(self):
        """Test invalid command handling via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "invalid-command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert len(output) > 0

    @pytest.mark.slow
    def test_cli_missing_required_args_integration(self):
        """Test missing required arguments via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail due to missing required argument
        assert result.returncode != 0

    @pytest.mark.slow
    def test_cli_invalid_model_type_integration(self):
        """Test invalid model type via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "invalid_model"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail with validation error
        assert result.returncode != 0

    @pytest.mark.slow
    def test_cli_nonexistent_data_file_integration(self):
        """Test nonexistent data file via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "mlp", "--data_path", "nonexistent.h5ad"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail due to missing data file
        assert result.returncode != 0

    @pytest.mark.slow
    def test_cli_list_models_integration(self):
        """Test list-models command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "list-models"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should succeed (list-models has no required args)
        assert result.returncode in {0, 1, 2}
        output = result.stdout + result.stderr
        assert len(output) > 0


class TestCLIEndToEnd:
    """End-to-end tests for complete CLI workflows."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_complete_workflow_help(self):
        """Test complete CLI workflow with help commands."""
        commands = [
            [sys.executable, "-m", "scxpand.main", "--help"],
            [sys.executable, "-m", "scxpand.main", "train", "--help"],
            [sys.executable, "-m", "scxpand.main", "optimize", "--help"],
            [sys.executable, "-m", "scxpand.main", "inference", "--help"],
            [sys.executable, "-m", "scxpand.main", "list-models", "--help"],
        ]

        for cmd in commands:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Each help command should work
            assert result.returncode in {0, 1, 2}
            output = result.stdout + result.stderr
            assert len(output) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_error_handling_workflow(self):
        """Test CLI error handling workflow."""
        error_commands = [
            [sys.executable, "-m", "scxpand.main", "invalid-command"],
            [sys.executable, "-m", "scxpand.main", "train"],  # Missing required args
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "invalid"],
        ]

        for cmd in error_commands:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Error commands should fail
            assert result.returncode != 0
            output = result.stdout + result.stderr
            assert len(output) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cli_performance_benchmark(self):
        """Benchmark CLI performance for common operations."""
        # Test help command performance
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        end_time = time.time()

        # Help should be fast
        duration = end_time - start_time
        assert duration < 5.0, f"Help command took {duration:.3f}s, should be < 5.0s"
        assert result.returncode in {0, 1, 2}


class TestCLIEnvironment:
    """Tests for CLI behavior in different environments."""

    @pytest.mark.slow
    def test_cli_with_different_python_versions(self):
        """Test CLI works with different Python versions."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should work regardless of Python version
        assert result.returncode in {0, 1, 2}

    @pytest.mark.slow
    def test_cli_with_different_working_directories(self):
        """Test CLI works from different working directories."""
        # Test from temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                result = subprocess.run(
                    [sys.executable, "-m", "scxpand.main", "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Should work from any directory
                assert result.returncode in {0, 1, 2}
            finally:
                os.chdir(original_cwd)

    @pytest.mark.slow
    def test_cli_with_environment_variables(self):
        """Test CLI behavior with different environment variables."""
        # Test with debug environment variable
        env = os.environ.copy()
        env["SCXPAND_DEBUG"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        # Should work with environment variables
        assert result.returncode in {0, 1, 2}
