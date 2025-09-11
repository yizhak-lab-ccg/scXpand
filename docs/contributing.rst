Contributing to scXpand
=======================

We welcome contributions to scXpand! This guide will help you contribute effectively to the project.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.

**Python Version Support**

We support Python 3.11, 3.12, and 3.13. Our CI tests run on all these versions to ensure compatibility.

**Running Tests**
Tests run automatically on every push to any branch, but you can also run them manually:

.. code-block:: bash

   # Run all tests in parallel (recommended)
   uv run pytest -n auto

   # Run specific test file
   uv run pytest tests/test_specific.py

   # Run with coverage
   uv run pytest -n auto --cov=scxpand


Commit Guidelines
-----------------

Write clear, descriptive commit messages that explain what your changes do. Examples:

- ``Add support for batch prediction``
- ``Fix memory leak in data preprocessing``
- ``Update installation instructions``
- ``Add tests for autoencoder training``

Keep commits focused on a single change when possible.


Pull Request Process
--------------------

1. **Create Branch**: ``git checkout -b your-feature-name``
2. **Make Changes**: Implement your changes with appropriate tests
3. **Test**: Run tests and ensure they pass
4. **Push**: ``git push origin your-feature-name``
5. **Create PR**: Open a pull request targeting ``main``
6. **Wait for CI**: All status checks must pass
7. **Code Review**: PR requires approval before merging
8. **Merge**: Use "Squash and merge" for clean history

**Pull Request Checklist**

- [ ] Tests added for new functionality
- [ ] All tests pass locally and in CI
- [ ] Documentation updated if needed
- [ ] PR description clearly explains changes
- [ ] Branch is up-to-date with main

.. note::
   Direct pushes to ``main`` are blocked when branch protection is enabled.
   All changes to ``main`` must go through pull requests.


Version Management
------------------

**Semantic Versioning**

We follow `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible functionality additions (1.0.0 → 1.1.0)
- **PATCH**: Backward-compatible bug fixes (1.0.0 → 1.0.1)

**Version Bumping**

Use uv's built-in version management:

.. code-block:: bash

   # Check current version
   uv version

   # Bump version with uv
   uv version --bump patch   # for bug fixes
   uv version --bump minor   # for new features
   uv version --bump major   # for breaking changes


Trusted Publishing Setup
=======================

**One-Time Setup for Maintainers**

Before you can publish releases, you need to configure PyPI Trusted Publishing:

**Step 1: Configure PyPI Trusted Publisher**

1. Go to https://pypi.org/manage/account/publishing/
2. Fill in the form:
   - **PyPI Project Name**: ``scxpand``
   - **Owner**: ``yizhak-lab-ccg`` (or your GitHub username/org)
   - **Repository name**: ``scXpand``
   - **Workflow filename**: ``release.yml``
   - **Environment name**: ``pypi``
3. Click "Add"

**Step 2: Configure TestPyPI Trusted Publisher**

1. Go to https://test.pypi.org/manage/account/publishing/
2. Repeat step 1, but use environment name: ``testpypi``

**Step 3: Set Up GitHub Environments**

1. Go to your GitHub repository → Settings → Environments
2. Create environment ``pypi``:
   - **Required reviewers**: Add yourself (for manual approval)
   - **Deployment branches**: Only protected branches
3. Create environment ``testpypi`` (no special settings needed)

.. note::
   If you don't have a TestPyPI account, create one at https://test.pypi.org/account/register/


Publishing Workflow Details
============================
The project uses GitHub Actions for automated publishing with the following behavior:

**TestPyPI Publishing (Automatic)**
- **Trigger**: Every push to any branch (including feature branches)
- **Purpose**: Development testing and validation
- **Approval**: None required
- **URL**: https://test.pypi.org/project/scxpand/

**PyPI Publishing (Manual Approval)**
- **Trigger**: Only when pushing git tags (e.g., ``git push origin v1.0.0``)
- **Purpose**: Official releases
- **Approval**: Manual approval required via GitHub ``pypi`` environment
- **URL**: https://pypi.org/project/scxpand/

.. note::
   If you're working on feature branches and don't want to publish development
   versions to TestPyPI, consider working in a fork instead of pushing directly
   to the main repository.


Release Process
===============

**For Maintainers Only**

Following the `uv packaging guide <https://docs.astral.sh/uv/guides/package/>`_, we use uv's built-in tools for building and publishing.

.. note::
   Due to branch protection rules, all changes to ``main`` must go through pull requests.

**Step 1: Prepare Release**

.. code-block:: bash

   # Create release branch from main
   git checkout main
   git pull origin main
   git checkout -b release/vX.X.X

   # Run tests to ensure everything works
   uv run pytest -n auto

   # Update version with uv
   uv version --bump patch   # or minor/major as needed

**Step 2: Update Changelog**

Update ``CHANGELOG.md`` with:

- Version number and date
- New features, bug fixes, changes
- Breaking changes (if any)

**Step 3: Build Package**

.. code-block:: bash

   # Build the package
   uv build

   # Verify the build artifacts in dist/
   ls -la dist/

**Step 4: Commit and Create PR**

.. code-block:: bash

   # Commit version changes
   git add -A
   git commit -m "Bump version to $(uv version)"

   # Push release branch
   git push origin release/vX.X.X

   # Create PR and get approval, then merge to main

**Step 5: Tag and Publish**

.. code-block:: bash

   # Switch to main after PR merge
   git checkout main
   git pull origin main

   # Create and push tag (triggers automated publishing)
   git tag v$(uv version)
   git push origin --tags

**Step 6: Verify Release**

The existing GitHub Actions workflow will automatically:

1. Build the package with ``uv build``
2. Publish to PyPI (with manual approval)
3. Create GitHub release

Verify the release at:
- **PyPI**: https://pypi.org/project/scxpand/
- **GitHub**: https://github.com/yizhak-lab-ccg/scXpand/releases

**Quick Release Checklist**

.. code-block:: bash

   # 1. Create release branch and update version
   git checkout main && git pull origin main
   git checkout -b release/vX.X.X
   uv run pytest -n auto
   uv version --bump patch  # or minor/major

   # 2. Build and commit
   uv build
   git add -A && git commit -m "Bump version to $(uv version)"
   git push origin release/vX.X.X

   # 3. Create PR, get approval, merge to main

   # 4. Tag and publish
   git checkout main && git pull origin main
   git tag v$(uv version) && git push origin --tags

   # 5. Approve deployment in GitHub Actions

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
