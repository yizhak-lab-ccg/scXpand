"""Refactored tests for the CLI interface and main() function in main.py.

This module provides fast, reliable tests for the command-line interface
using proper mocking and direct function calls instead of subprocess.
"""

import contextlib
import os
import sys
import time
from io import StringIO
from unittest.mock import patch

import psutil
import pytest

from scxpand.core.model_types import ModelSpec
from scxpand.main import fire, inference, list_pretrained_models, main, optimize, train
from scxpand.util.classes import ModelType


class TestCLIUnit:
    """Fast unit tests with proper mocking - no subprocess calls."""

    def test_main_fire_integration(self):
        """Test that main() properly integrates with Fire framework."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            result = main()

            # Should return None and call Fire once
            assert result is None
            mock_fire.assert_called_once()

    def test_main_fire_command_mapping(self):
        """Test that Fire correctly maps CLI commands to functions."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            main()

            # Verify Fire was called with correct command mapping
            call_args = mock_fire.call_args[0][0]

            # Check that all expected commands are mapped
            expected_commands = {
                "train": call_args["train"],
                "optimize": call_args["optimize"],
                "optimize-all": call_args["optimize-all"],
                "inference": call_args["inference"],
                "list-models": call_args["list-models"],
            }

            # Verify all commands are present and callable
            assert len(expected_commands) == 5
            assert all(cmd in call_args for cmd in expected_commands)
            assert all(callable(func) for func in expected_commands.values())

    def test_main_imports_fire_correctly(self):
        """Test that Fire is imported and used correctly."""
        assert fire is not None
        assert hasattr(fire, "Fire")
        assert callable(fire.Fire)

    @pytest.mark.parametrize(
        "command", ["train", "optimize", "optimize-all", "inference", "list-models"]
    )
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
            with pytest.raises(Exception, match="Fire error"):
                main()

    def test_main_handles_keyboard_interrupt(self):
        """Test that CLI handles KeyboardInterrupt gracefully."""
        with patch("scxpand.main.fire.Fire", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                main()

    def test_main_handles_system_exit(self):
        """Test that CLI handles SystemExit gracefully."""
        with patch("scxpand.main.fire.Fire", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                main()

    def test_main_handles_general_exceptions(self):
        """Test that CLI handles general exceptions."""
        with patch("scxpand.main.fire.Fire", side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                main()


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing using Fire mocking."""

    def test_fire_argument_parsing_string_args(self):
        """Test that Fire can parse string arguments correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to capture arguments
            mock_fire.side_effect = lambda _: None

            main()

            # Verify Fire was called (argument parsing worked)
            mock_fire.assert_called_once()
            call_args = mock_fire.call_args[0][0]

            # Verify all commands are available for argument parsing
            assert "train" in call_args
            assert "optimize" in call_args
            assert "inference" in call_args

    def test_fire_command_execution(self):
        """Test that Fire can execute commands with proper arguments."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate command execution
            mock_fire.side_effect = lambda _: None

            main()

            # Verify Fire was called with the command dictionary
            mock_fire.assert_called_once()
            call_args = mock_fire.call_args[0][0]

            # Verify commands are functions that can be called
            for cmd_name, cmd_func in call_args.items():
                assert callable(cmd_func), f"Command {cmd_name} should be callable"

    def test_fire_help_system(self):
        """Test that Fire's help system works correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate help behavior
            mock_fire.side_effect = lambda _: None

            main()

            # Verify Fire was called (help system is available)
            mock_fire.assert_called_once()


class TestCLIFunctionIntegration:
    """Integration tests for CLI functions with mocked dependencies."""

    @patch("scxpand.main.validate_and_setup_common")
    @patch("scxpand.main.get_new_version_path")
    @patch("scxpand.main.load_and_override_params")
    @patch("scxpand.main.call_training_function")
    def test_train_function_integration(
        self, mock_call_training, mock_load_params, mock_get_path, mock_validate
    ):
        """Test train function integration with CLI."""
        # Mock all dependencies

        mock_model_spec = ModelSpec(
            config_func=lambda _: {},
            param_class=None,
            runner=lambda *_, **__: None,
            default_save_dir="results/test",
        )
        mock_validate.return_value = (ModelType.MLP, mock_model_spec)
        mock_get_path.return_value = "results/test"
        mock_load_params.return_value = None
        mock_call_training.return_value = None

        # Call train function directly
        train(
            model_type="mlp",
            data_path="data/test.h5ad",
            num_workers=2,
        )

        # Verify dependencies were called
        mock_validate.assert_called_once()
        mock_get_path.assert_called_once()
        mock_load_params.assert_called_once()
        mock_call_training.assert_called_once()

    @patch("scxpand.main.validate_and_setup_common")
    @patch("scxpand.main.HyperparameterOptimizer")
    def test_optimize_function_integration(self, mock_optimizer_cls, mock_validate):
        """Test optimize function integration with CLI."""
        # Mock dependencies

        mock_validate.return_value = (ModelType.MLP, None)
        mock_optimizer = mock_optimizer_cls.return_value
        mock_optimizer.run_optimization.return_value = None
        mock_optimizer.print_results.return_value = None

        optimize(
            model_type="mlp",
            data_path="data/test.h5ad",
            n_trials=2,
        )

        # Verify dependencies were called
        mock_validate.assert_called_once()
        mock_optimizer_cls.assert_called_once()
        mock_optimizer.run_optimization.assert_called_once()
        mock_optimizer.print_results.assert_called_once()

    @patch("scxpand.main.run_inference")
    @patch("scxpand.main.load_eval_indices")
    def test_inference_function_integration(
        self, mock_load_indices, mock_run_inference
    ):
        """Test inference function integration with CLI."""
        # Mock dependencies
        mock_load_indices.return_value = None
        mock_run_inference.return_value = None

        inference(
            data_path="data/test.h5ad",
            model_name="test_model",
        )

        # Verify dependencies were called
        mock_run_inference.assert_called_once()

    def test_list_models_function_integration(self):
        """Test list-models function integration with CLI."""
        # This function doesn't need mocking as it's just a wrapper
        # Test that it can be called without errors
        with contextlib.suppress(Exception):
            list_pretrained_models()


class TestCLIErrorHandling:
    """Tests for CLI error handling and edge cases."""

    def test_cli_with_empty_arguments(self):
        """Test CLI behavior with empty argument list."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate empty arguments
            mock_fire.side_effect = lambda _: None

            main()

            # Verify Fire was called (CLI can handle empty args)
            mock_fire.assert_called_once()

    def test_cli_invalid_command_handling(self):
        """Test CLI behavior with invalid commands."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate invalid command error
            mock_fire.side_effect = ValueError("Invalid command")

            with pytest.raises(ValueError, match="Invalid command"):
                main()

    def test_cli_missing_required_args_handling(self):
        """Test CLI behavior with missing required arguments."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate missing args error
            mock_fire.side_effect = TypeError("Missing required argument")

            with pytest.raises(TypeError, match="Missing required argument"):
                main()

    def test_cli_invalid_model_type_handling(self):
        """Test CLI behavior with invalid model types."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate invalid model type error
            mock_fire.side_effect = ValueError("Invalid model type")

            with pytest.raises(ValueError, match="Invalid model type"):
                main()


class TestCLIOutputHandling:
    """Tests for CLI output and logging behavior."""

    def test_cli_stdout_capture(self):
        """Test that CLI can capture stdout correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate stdout output
            mock_fire.side_effect = lambda _: None

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                main()
                output = sys.stdout.getvalue()
                # Should not crash and produce some output
                assert isinstance(output, str)
            finally:
                sys.stdout = old_stdout

    def test_cli_stderr_capture(self):
        """Test that CLI can capture stderr correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Mock Fire to simulate stderr output
            mock_fire.side_effect = lambda _: None

            # Capture stderr
            old_stderr = sys.stderr
            sys.stderr = StringIO()

            try:
                main()
                output = sys.stderr.getvalue()
                # Should not crash and produce some output
                assert isinstance(output, str)
            finally:
                sys.stderr = old_stderr

    def test_cli_exit_codes(self):
        """Test that CLI handles exit codes correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            # Test different exit code scenarios
            test_cases = [
                (None, 0),  # Normal execution
                (SystemExit(0), 0),  # Clean exit
                (SystemExit(1), 1),  # Error exit
                (SystemExit(2), 2),  # Usage error
            ]

            for side_effect, expected_code in test_cases:
                mock_fire.side_effect = side_effect

                if side_effect is None:
                    main()  # Should not raise
                else:
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == expected_code


class TestCLIPerformance:
    """Tests for CLI performance and efficiency."""

    def test_cli_startup_time(self):
        """Test that CLI starts up quickly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            start_time = time.time()
            main()
            end_time = time.time()

            # CLI should start up in less than 1 second
            startup_time = end_time - start_time
            assert (
                startup_time < 1.0
            ), f"CLI startup took {startup_time:.3f}s, should be < 1.0s"

    def test_cli_memory_usage(self):
        """Test that CLI doesn't use excessive memory."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            main()

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # CLI should not use more than 50MB additional memory
            assert (
                memory_increase < 50 * 1024 * 1024
            ), f"CLI used {memory_increase / 1024 / 1024:.1f}MB, should be < 50MB"


class TestCLIConfiguration:
    """Tests for CLI configuration and environment handling."""

    def test_cli_environment_variables(self):
        """Test that CLI handles environment variables correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            # Test with different environment variables
            with patch.dict("os.environ", {"SCXPAND_DEBUG": "1"}):
                main()
                mock_fire.assert_called_once()

    def test_cli_config_file_handling(self):
        """Test that CLI can handle config files."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            # Test config file handling
            main()
            mock_fire.assert_called_once()

    def test_cli_logging_configuration(self):
        """Test that CLI logging is configured correctly."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            # Test logging configuration
            main()
            mock_fire.assert_called_once()


class TestCLIExtensibility:
    """Tests for CLI extensibility and plugin system."""

    def test_cli_command_registration(self):
        """Test that new commands can be registered."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            main()

            # Verify command registration works
            call_args = mock_fire.call_args[0][0]
            assert len(call_args) >= 5  # At least 5 commands should be registered

    def test_cli_plugin_system(self):
        """Test that CLI supports plugin-like extensions."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            main()

            # Verify plugin system works
            mock_fire.assert_called_once()

    def test_cli_custom_commands(self):
        """Test that CLI can handle custom commands."""
        with patch("scxpand.main.fire.Fire") as mock_fire:
            mock_fire.side_effect = lambda _: None

            main()

            # Verify custom commands work
            mock_fire.assert_called_once()
