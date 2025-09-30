Contributing to scXpand
=======================

We welcome contributions to scXpand! This document provides guidelines for contributing to the project.

Reporting Issues
----------------

When reporting issues, please include:

- Python version
- scXpand version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

Contributing Process
--------------------

**Quick Start:**

1. **Fork and clone the repository**
2. **Set up development environment:**
   Follow the development setup instructions in :doc:`installation`
3. **Create a feature branch and make your changes**

**Before submitting your pull request:**

4. **Ensure all tests pass** and meet coverage requirements:

   .. code-block:: bash

      # Install in editable mode and run tests with coverage
      pip install -e ".[dev]" && pytest --cov=src/scxpand -n auto

5. **Add tests** for new functionality
6. **Update documentation** if your changes affect the API or user-facing functionality
7. **Update CHANGELOG.md** with your changes
8. **Run pre-commit hooks** to ensure code quality:

   .. code-block:: bash

      # Run linting and formatting
      pre-commit run --all-files

**Submit your pull request:**

9. **Create a pull request** with:

   - Clear, descriptive title
   - Detailed description of changes
   - Reference to any related issues
   - Screenshots or examples if applicable

10. **Request review** from maintainers

Your pull request will be reviewed and may require changes before being merged.



Template Updates
----------------

This project uses the `scverse cookiecutter template <https://github.com/scverse/cookiecutter-scverse>`_ and can be updated when new template versions are released.

To update the project with the latest template changes:

.. code-block:: bash

   # Check if updates are available
   cruft check

   # Update to the latest template version
   cruft update

   # Review and commit the changes
   git add .
   git commit -m "Update from cookiecutter template"

Testing
-------

**Running Tests:**

The project uses pytest with coverage reporting and parallel execution for efficient testing:

.. code-block:: bash

    # Install in editable mode and run all tests with coverage (recommended)
    pip install -e ".[dev]" && pytest --cov=src/scxpand -n auto

    # Run tests without parallel execution (if you encounter issues)
    pip install -e ".[dev]" && pytest --cov=src/scxpand

    # Run specific test files or modules
    pip install -e ".[dev]" && pytest tests/data_util/test_data_format.py

    # Run tests with verbose output
    pip install -e ".[dev]" && pytest --cov=src/scxpand -n auto -v

    # Run tests and generate HTML coverage report
    pip install -e ".[dev]" && pytest --cov=src/scxpand --cov-report=html -n auto

**Test Coverage:**

The project maintains a minimum test coverage of 80%. Coverage reports are generated in multiple formats:
- Terminal output with missing lines
- HTML report in the `htmlcov/` directory
- XML report for CI/CD integration

**Parallel Execution:**

The `-n auto` flag automatically detects the optimal number of parallel workers based on your CPU cores.

**Pre-commit Integration:**

Tests are also run automatically via pre-commit hooks to ensure code quality before commits.

Release Process
===============

**For Maintainers Only**

We use an automated release script that handles the entire publishing process to PyPI.
For detailed instructions, see :doc:`scripts/PUBLISHING`.

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
