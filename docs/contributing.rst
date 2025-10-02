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
   Follow the "Development Setup (from Source)" instructions in :doc:`installation`
3. **Create a feature branch and make your changes**

**Before submitting your pull request:**

4. **Ensure all tests pass** and meet coverage requirements:

   .. code-block:: bash

      # Install in editable mode and run tests with coverage
      pytest --cov=scxpand -n auto

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

The project uses pytest with coverage reporting and parallel execution. After following the development setup in :doc:`installation`.
Tests also run automatically via pre-commit hooks before push, but you can run them manually with the following commands:

.. code-block:: bash

    # Install in editable mode and run all tests with coverage (recommended)
   pytest --cov=scxpand -n auto

    # Run tests without parallel execution (if you encounter issues)
    pytest --cov=scxpand

    # Run specific test files
    pytest tests/data_util/test_data_format.py

    # Generate HTML coverage report
    pytest --cov=scxpand --cov-report=html -n auto

**Test Coverage:**

The project maintains a minimum test coverage of 80%. Coverage reports are generated in:
- Terminal output with missing lines
- HTML report in the `htmlcov/` directory
- XML report for CI/CD integration

**Note:** The `-n auto` flag automatically detects the optimal number of parallel workers.

Release Process
===============

**For Maintainers Only**

We use GitHub Actions for automated dual package releases (standard and CUDA versions).
The release process is fully integrated with CI/CD and includes changelog validation.

Creating Releases
-----------------

**Method 1: GitHub Actions UI (Recommended)**

1. Go to the `Actions tab <https://github.com/yizhak-lab-ccg/scXpand/actions/workflows/release.yml>`_ in GitHub
2. Click "Run workflow"
3. Select the version bump type (patch/minor/major)
4. Optionally check "Create dev release" for testing
5. Click "Run workflow"

**Method 2: Manual Tag Push**

.. code-block:: bash

    # Create and push a version tag
    git tag v0.4.6
    git push origin v0.4.6

Release Types
-------------

**Regular Releases:**
- Create GitHub releases with auto-generated notes
- Publish both ``scxpand`` (CPU/MPS) and ``scxpand-cuda`` (CUDA) packages
- Trigger ReadTheDocs documentation builds
- add changelog entries to CHANGELOG.md

**Dev Releases:**
- No GitHub release creation
- Publish packages with ``.dev0`` suffix (e.g., ``0.4.6.dev0``)
- Useful for testing before official releases




Version Management
------------------

We use `Semantic Versioning <https://semver.org/>`_ with VCS-based versioning:

- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible functionality additions (1.0.0 → 1.1.0)
- **PATCH**: Backward-compatible bug fixes (1.0.0 → 1.0.1)
- **Version source**: Git tags (e.g., ``v0.4.6``, ``v0.4.7.dev0``)
- **No manual version bumping** required in ``pyproject.toml``

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
