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

We use GitHub Actions for automated dual package releases (standard and CUDA versions).
The release process is fully integrated with CI/CD and includes changelog validation.

Creating Releases
-----------------

**Prerequisites:**
- Ensure you have a changelog entry for the target version in CHANGELOG.md
- All tests and checks must be passing on the main branch
- You must have maintainer permissions on the repository

**Method 1: GitHub Actions UI (Recommended)**

This is the easiest and most reliable method:

1. **Navigate to the Release workflow:**
   - Go to `Actions tab <https://github.com/yizhak-lab-ccg/scXpand/actions/workflows/release.yml>`_
   - Click on the "Release" workflow in the left sidebar

2. **Trigger the release:**
   - Click the "Run workflow" button (top right)
   - Select the version bump type:
     - **patch**: Bug fixes (0.4.4 → 0.4.5)
     - **minor**: New features (0.4.4 → 0.5.0)
     - **major**: Breaking changes (0.4.4 → 1.0.0)
   - Optionally check "Create dev release" for testing (skips changelog validation)
   - Click "Run workflow"

3. **Monitor the release:**
   - The workflow will automatically:
     - Validate the changelog entry
     - Create and push a git tag
     - Build both standard and CUDA packages
     - Publish to PyPI
     - Create a GitHub release

**Method 2: Manual Tag Push**

For advanced users who prefer manual control:

.. code-block:: bash

    # Ensure you're on the main branch with latest changes
    git checkout main
    git pull origin main

    # Create and push a version tag (replace with actual version)
    git tag v0.4.6
    git push origin v0.4.6

    # This will automatically trigger the release workflow

Release Types
-------------

**Regular Releases:**
- Require changelog entry validation
- Create GitHub releases with auto-generated notes
- Publish both ``scxpand`` (CPU/MPS) and ``scxpand-cuda`` (CUDA) packages
- Trigger ReadTheDocs documentation builds

**Dev Releases:**
- Skip changelog validation
- No GitHub release creation
- Publish packages with ``.dev0`` suffix (e.g., ``0.4.6.dev0``)
- Useful for testing before official releases

Changelog Requirements
----------------------

**IMPORTANT:** Before creating a regular release, you **must** add a changelog entry.

1. **Add your changes to CHANGELOG.md:**

   .. code-block:: markdown

      ## [0.4.6] - 2025-01-15

      - New hyperparameter optimization for MLP models
      - Support for custom loss functions
      - Improved memory efficiency in data loading
      - Fixed CUDA memory leak in autoencoder training

2. **Changelog validation** runs automatically and will fail if:
   - No entry exists for the version
   - Entry has no bullet points
   - Entry contains only placeholder dashes
   - Date format is incorrect

3. **Best practices:**
   - Use descriptive bullet points for each change
   - Group related changes together
   - Use present tense ("Add feature" not "Added feature")
   - Include breaking changes prominently
   - Reference issue numbers when relevant

**Example of a complete release process:**

1. Make your code changes and tests
2. Add changelog entry for the next version
3. Create and merge a pull request
4. Go to GitHub Actions → Release workflow
5. Click "Run workflow" → Select version type → Run
6. Monitor the automated release process

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
