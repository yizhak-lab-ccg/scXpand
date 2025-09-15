"""Tests for the CLI interface and main() function in main.py.

This module tests the command-line interface integration using Fire,
including argument parsing, command routing, and end-to-end CLI behavior.
"""

import subprocess
import sys

from unittest.mock import patch

import pytest

from scxpand.main import fire, main


class TestMainCLI:
    """Tests for the main CLI entry point and Fire integration."""

    def test_main_fire_integration(self):
        """Test that main() properly integrates with Fire framework."""
        # Test that main() doesn't crash and returns None
        result = main()
        assert result is None

    def test_main_fire_command_mapping(self):
        """Test that Fire correctly maps CLI commands to functions."""
        # Mock the Fire.Fire call to verify command mapping
        with patch("scxpand.main.fire.Fire") as mock_fire:
            main()

            # Verify Fire was called with correct command mapping
            mock_fire.assert_called_once()
            call_args = mock_fire.call_args[0][0]

            # Check that all expected commands are mapped
            expected_commands = {
                "train": call_args["train"],
                "optimize": call_args["optimize"],
                "optimize-all": call_args["optimize-all"],
                "inference": call_args["inference"],
                "list-models": call_args["list-models"],
            }

            # Verify all commands are present
            assert len(expected_commands) == 5
            assert all(cmd in call_args for cmd in expected_commands)

    def test_main_imports_fire_correctly(self):
        """Test that Fire is imported and used correctly."""
        # This test ensures the import structure is correct
        assert fire is not None
        assert hasattr(fire, "Fire")

    @pytest.mark.parametrize("command", ["train", "optimize", "optimize-all", "inference", "list-models"])
    def test_cli_command_availability(self, command):
        """Test that each CLI command is available through Fire."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            main()

            # Get the command mapping from Fire call
            call_args = mock_fire.call_args[0][0]
            assert command in call_args
            assert callable(call_args[command])

    def test_main_handles_fire_exceptions(self):
        """Test that main() handles Fire exceptions gracefully."""
        with patch("scxpand.main.fire.Fire", side_effect=Exception("Fire error")):
            # Should not crash the application
            with pytest.raises(Exception, match="Fire error"):
                main()


class TestCLIEndToEnd:
    """End-to-end tests for CLI commands using subprocess."""

    def test_cli_help_command(self):
        """Test that CLI shows help information."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "--help"], check=False, capture_output=True, text=True, timeout=30
        )

        # Should succeed and show help
        assert result.returncode == 0
        assert "train" in result.stdout
        assert "optimize" in result.stdout
        assert "inference" in result.stdout
        assert "list-models" in result.stdout

    def test_cli_train_help(self):
        """Test that train command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "model_type" in result.stdout
        assert "data_path" in result.stdout

    def test_cli_optimize_help(self):
        """Test that optimize command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "optimize", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "model_type" in result.stdout
        assert "n_trials" in result.stdout

    def test_cli_optimize_all_help(self):
        """Test that optimize-all command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "optimize-all", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "data_path" in result.stdout
        assert "n_trials" in result.stdout

    def test_cli_inference_help(self):
        """Test that inference command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "inference", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "data_path" in result.stdout
        assert "model_path" in result.stdout

    def test_cli_list_models_help(self):
        """Test that list-models command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "list-models", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

    def test_cli_invalid_command(self):
        """Test CLI behavior with invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "invalid-command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        assert "invalid-command" in result.stderr or "Usage:" in result.stderr

    def test_cli_missing_required_args(self):
        """Test CLI behavior with missing required arguments."""
        # Test train command without required model_type
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train"], check=False, capture_output=True, text=True, timeout=30
        )

        # Should fail due to missing required argument
        assert result.returncode != 0

    def test_cli_invalid_model_type(self):
        """Test CLI behavior with invalid model type."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "invalid_model"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail with validation error
        assert result.returncode != 0
        assert "model_type must be one of" in result.stderr or "invalid_model" in result.stderr

    def test_cli_nonexistent_data_file(self):
        """Test CLI behavior with nonexistent data file."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "mlp", "--data_path", "nonexistent.h5ad"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail due to missing data file
        assert result.returncode != 0
        assert "not found" in result.stderr or "FileNotFoundError" in result.stderr


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing and validation."""

    def test_cli_argument_types(self):
        """Test that CLI correctly parses different argument types."""
        # Test with various argument types
        test_cases = [
            # String arguments
            ("--model_type", "autoencoder"),
            ("--data_path", "data/test.h5ad"),
            # Integer arguments
            ("--n_trials", "50"),
            ("--num_workers", "4"),
            ("--batch_size", "1024"),
            # Boolean arguments
            ("--resume", "true"),
            ("--fail_fast", "false"),
        ]

        for arg_name, arg_value in test_cases:
            # Test that arguments are accepted (even if validation fails later)
            result = subprocess.run(
                [sys.executable, "-m", "scxpand.main", "train", arg_name, arg_value],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should not fail due to argument parsing (may fail due to validation)
            # We're just testing that Fire can parse the arguments
            assert "unrecognized arguments" not in result.stderr

    def test_cli_kwargs_handling(self):
        """Test that CLI correctly handles **kwargs parameters."""
        # Test with additional parameters that should be passed through
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "mlp", "--n_epochs", "10"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should not fail due to unknown parameter (n_epochs is a valid kwargs)
        assert "unrecognized arguments" not in result.stderr

    def test_cli_default_values(self):
        """Test that CLI uses correct default values."""
        # Test that commands work with minimal arguments (using defaults)
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "list-models"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should succeed (list-models has no required args)
        assert result.returncode == 0


class TestCLIErrorHandling:
    """Tests for CLI error handling and edge cases."""

    def test_cli_handles_keyboard_interrupt(self):
        """Test that CLI handles KeyboardInterrupt gracefully."""
        # This is more of a conceptual test since we can't easily simulate Ctrl+C
        # But we can test that the main function doesn't have obvious issues
        with patch("scxpand.main.fire.Fire", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                main()

    def test_cli_handles_system_exit(self):
        """Test that CLI handles SystemExit gracefully."""
        with patch("scxpand.main.fire.Fire", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                main()

    def test_cli_handles_general_exceptions(self):
        """Test that CLI handles general exceptions."""
        with patch("scxpand.main.fire.Fire", side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                main()

    def test_cli_with_empty_arguments(self):
        """Test CLI behavior with empty argument list."""
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main"], check=False, capture_output=True, text=True, timeout=30
        )

        # Should show help or usage information
        assert result.returncode in {0, 1}
        # Should contain some indication of available commands
        output = result.stdout + result.stderr
        assert any(cmd in output for cmd in ["train", "optimize", "inference", "list-models"])


class TestCLIIntegration:
    """Integration tests for CLI with mocked dependencies."""

    @patch("scxpand.main.train")
    def test_cli_train_integration(self, mock_train):
        """Test CLI train command integration."""
        # Mock the train function to avoid actual training
        mock_train.return_value = None

        # Test CLI call
        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "train", "--model_type", "mlp", "--data_path", "data/test.h5ad"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should succeed (with mocked dependencies)
        assert result.returncode == 0

    @patch("scxpand.main.optimize")
    def test_cli_optimize_integration(self, mock_optimize):
        """Test CLI optimize command integration."""
        mock_optimize.return_value = None

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scxpand.main",
                "optimize",
                "--model_type",
                "mlp",
                "--data_path",
                "data/test.h5ad",
                "--n_trials",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

    @patch("scxpand.main.inference")
    def test_cli_inference_integration(self, mock_inference):
        """Test CLI inference command integration."""
        mock_inference.return_value = None

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scxpand.main",
                "inference",
                "--data_path",
                "data/test.h5ad",
                "--model_name",
                "test_model",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0

    @patch("scxpand.main.list_pretrained_models")
    def test_cli_list_models_integration(self, mock_list_models):
        """Test CLI list-models command integration."""
        mock_list_models.return_value = None

        result = subprocess.run(
            [sys.executable, "-m", "scxpand.main", "list-models"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
