Contributing to scXpand
=======================

We welcome contributions to scXpand! This document provides guidelines for contributing to the project.

Quick Start
-----------

1. **Fork and clone the repository**
2. **Install in development mode:**
   .. code-block:: bash
      pip install -e ".[dev]"
3. **Install pre-commit hooks:**
   .. code-block:: bash
      pre-commit install
4. **Create a feature branch and make your changes**
5. **Run tests and linting:**
   .. code-block:: bash
      # Run tests with coverage (parallel execution for faster results)
      pytest --cov=src/scxpand --cov-report=term-missing -n auto

      # Run linting and formatting
      pre-commit run --all-files
6. **Submit a pull request**

Coding Standards
----------------

We use several tools to maintain code quality:

- **Ruff** for code formatting and linting
- **Pre-commit hooks** for automated checks
- **Type hints** for better code documentation
- **Comprehensive testing** with pytest

All tools are configured in `pyproject.toml` and run automatically via pre-commit hooks.

Pull Request Process
--------------------

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

Reporting Issues
----------------

When reporting issues, please include:

- Python version
- scXpand version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.

Testing
-------

**Running Tests:**

The project uses pytest with coverage reporting and parallel execution for efficient testing:

.. code-block:: bash

    # Run all tests with coverage and parallel execution (recommended)
    pytest --cov=src/scxpand --cov-report=term-missing -n auto

    # Run tests without parallel execution (if you encounter issues)
    pytest --cov=src/scxpand --cov-report=term-missing

    # Run specific test files or modules
    pytest tests/data_util/test_data_format.py

    # Run tests with verbose output
    pytest --cov=src/scxpand --cov-report=term-missing -n auto -v

    # Run tests and generate HTML coverage report
    pytest --cov=src/scxpand --cov-report=html --cov-report=term-missing -n auto

**Test Coverage:**

The project maintains a minimum test coverage of 80%. Coverage reports are generated in multiple formats:
- Terminal output with missing lines
- HTML report in the `htmlcov/` directory
- XML report for CI/CD integration

**Parallel Execution:**

The `-n auto` flag automatically detects the optimal number of parallel workers based on your CPU cores. This significantly speeds up test execution, especially for the comprehensive test suite with 900+ tests.

**Pre-commit Integration:**

Tests are also run automatically via pre-commit hooks to ensure code quality before commits.

Release Process
===============

**For Maintainers Only**

We use an automated release script that handles the entire publishing process to PyPI.
For detailed instructions, see :doc:`../scripts/PUBLISHING`.

Dev Releases
------------

For testing releases before official announcement, use dev releases:

.. code-block:: bash

    # Create a dev release (no GitHub announcement)
    ./scripts/release.sh --dev

    # Dry run for dev release
    ./scripts/release.sh --dev --dry-run

Dev releases:
- Publish packages to PyPI with dev version suffix (e.g., 0.3.6.dev1)
- Skip GitHub release creation and announcement
- Skip ReadTheDocs documentation build
- Useful for testing releases on other machines before official release

Version Management
------------------

We use `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible functionality additions (1.0.0 → 1.1.0)
- **PATCH**: Backward-compatible bug fixes (1.0.0 → 1.0.1)

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
