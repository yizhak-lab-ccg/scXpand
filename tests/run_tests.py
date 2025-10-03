#!/usr/bin/env python3
"""
Test runner script for the reorganized test structure.

This script provides convenient commands to run different categories of tests
following the new organization structure.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <test_type>")
        print("\nAvailable test types:")
        print("  unit        - Run all unit tests (fast)")
        print("  integration - Run all integration tests (medium)")
        print("  e2e         - Run all end-to-end tests (slow)")
        print("  fast        - Run unit + some integration tests")
        print("  all         - Run all tests")
        print("  discovery   - Test discovery only (verify structure)")
        print("  auto        - Run autoencoder tests")
        print("  data        - Run data processing tests")
        print("  models      - Run model tests")
        print("  cli         - Run CLI tests")
        print("  coverage    - Run tests with coverage report")
        return 1

    test_type = sys.argv[1].lower()
    base_dir = Path(__file__).parent.parent

    commands = {
        "unit": ["python", "-m", "pytest", "tests/unit/", "--tb=short"],
        "integration": ["python", "-m", "pytest", "tests/integration/", "--tb=short"],
        "e2e": ["python", "-m", "pytest", "tests/e2e/", "--tb=short"],
        "fast": [
            "python",
            "-m",
            "pytest",
            "tests/unit/",
            "tests/integration/data_processing/test_*consistency*.py",
            "tests/integration/data_processing/test_*basic*.py",
        ],
        "all": ["python", "-m", "pytest", "tests/", "--tb=short"],
        "discovery": ["python", "-m", "pytest", "tests/", "--collect-only", "--quiet"],
        "auto": ["python", "-m", "pytest", "tests/unit/autoencoders/", "--tb=short"],
        "data": ["python", "-m", "pytest", "tests/unit/data/", "--tb=short"],
        "models": ["python", "-m", "pytest", "tests/unit/models/", "--tb=short"],
        "cli": ["python", "-m", "pytest", "tests/e2e/cli/", "--tb=short"],
        "coverage": [
            "python",
            "-m",
            "pytest",
            "tests/",
            "--cov=src/scxpand",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ],
    }

    if test_type not in commands:
        print(f"Unknown test type: {test_type}")
        return 1

    cmd = commands[test_type]

    # Change to project root directory
    try:
        import os

        os.chdir(base_dir)

        # Run the selected test command
        success = run_command(cmd, f"{test_type.upper()} tests")

        if test_type == "coverage" and success:
            print(f"\n📊 Coverage report generated in: {base_dir}/htmlcov/index.html")

        return 0 if success else 1

    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
