Contributing to scXpand
=======================

We welcome contributions to scXpand! This guide will help you contribute effectively to the project.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.

**Python Version Support**

We support Python 3.11, 3.12, and 3.13. Our CI tests run on all these versions to ensure compatibility.

**Running Tests**

.. code-block:: bash

   # Run all tests
   uv run pytest

   # Run specific test file
   uv run pytest tests/test_specific.py

   # Run with coverage
   uv run pytest --cov=scxpand


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

Update versions manually or use UV:

.. code-block:: bash

   # Check current version
   grep 'version =' pyproject.toml

   # Bump version with UV (updates pyproject.toml only)
   uv version --bump patch   # for bug fixes
   uv version --bump minor   # for new features
   uv version --bump major   # for breaking changes

   # Then manually update scxpand/__init__.py to match


Trusted Publishing Setup
=======================

**One-Time Setup for Maintainers**

Before you can publish releases, you need to configure PyPI Trusted Publishing:

**Step 1: Configure PyPI Trusted Publisher**

1. Go to https://pypi.org/manage/account/publishing/
2. Fill in the form:
   - **PyPI Project Name**: ``scxpand``
   - **Owner**: ``ronamit`` (or your GitHub username/org)
   - **Repository name**: ``scxpand``
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

**Step 1: Prepare Release**

.. code-block:: bash

   # Ensure you're on main and up to date
   git checkout main
   git pull origin main

   # Run tests locally to ensure everything works
   uv run pytest

   # Update version
   uv version --bump <patch|minor|major>
   # Then manually update scxpand/__init__.py to match

**Step 2: Update Changelog**

Update ``CHANGELOG.md`` with:

- Version number and date
- New features, bug fixes, changes
- Breaking changes (if any)

**Step 3: Commit and Tag**

.. code-block:: bash

   # Create release branch
   git checkout -b release/vX.X.X

   # Commit version changes
   git add -A
   git commit -m "Bump version to X.X.X"

   # Push release branch and create PR
   git push origin release/vX.X.X
   # Open PR to main, get approval, and merge

   # After PR is merged, tag the main branch
   git checkout main
   git pull origin main
   git tag vX.X.X
   git push origin --tags

**Step 4: Automated Publishing**

The GitHub Actions workflow automatically:

1. **On every push to any branch**: Publishes to TestPyPI (no approval needed)
2. **On tagged pushes only**: Publishes to PyPI (requires manual approval via ``pypi`` environment)
3. **After PyPI publish**: Creates GitHub release with signed artifacts

.. warning::
   Every push to any branch triggers a TestPyPI publication. This includes feature branches,
   so be mindful that development versions will be published to TestPyPI automatically.

**Step 5: Manual Approval (PyPI only)**

1. Go to your repository's Actions tab
2. Find the running workflow for your tag
3. Click on the "pypi" environment deployment
4. Click "Review deployments" and approve

**Step 6: Verify Release**

- Check PyPI: https://pypi.org/project/scxpand/
- Check TestPyPI: https://test.pypi.org/project/scxpand/
- Test installation: ``pip install scxpand==X.X.X``


Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
