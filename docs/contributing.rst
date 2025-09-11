Contributing to scXpand
=======================

We welcome contributions to scXpand! This guide will help you contribute effectively to the project.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.

**Python Version Support**

We support Python 3.11, 3.12, and 3.13. All tests should pass on these versions before submitting PRs.

**Running Tests**

Run tests manually with:

.. code-block:: bash

   # Run all tests (parallel)
   uv run pytest -n auto

   # Run fast tests only (skip slow tests)
   uv run pytest -n auto -m "not slow"

   # Run specific test file
   uv run pytest tests/your_test_file.py

Contributing Workflow
---------------------

**1. Create Branch**

.. code-block:: bash

   git checkout -b your-feature-name

**2. Make Changes**

Implement your changes with appropriate tests and documentation updates.

**3. Test Your Changes**

.. code-block:: bash

   # Run all tests
   uv run pytest -n auto

   # Run specific tests for your changes
   uv run pytest tests/your_test_file.py

**4. Commit Guidelines**

Write clear, descriptive commit messages.
Keep commits focused on a single change when possible.

**5. Create Pull Request**

.. code-block:: bash

   git push origin your-feature-name
   # Then create PR targeting main branch

**Pull Request Checklist**

- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated if needed
- [ ] PR description clearly explains changes
- [ ] Branch is up-to-date with main

Release Process
===============

**For Maintainers Only**

We follow the `uv packaging guide <https://docs.astral.sh/uv/guides/package/>`_ for manual publishing.

Version Management
------------------

We use `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible functionality additions (1.0.0 → 1.1.0)
- **PATCH**: Backward-compatible bug fixes (1.0.0 → 1.0.1)

One-Time Setup
--------------

**1. PyPI Account Setup**

- Create account at https://pypi.org/account/register/
- Generate API token at https://pypi.org/manage/account/token/
- Set environment variable: ``export UV_PUBLISH_TOKEN=your_token_here``

**2. TestPyPI Account Setup (Optional)**

- Create account at https://test.pypi.org/account/register/
- Generate API token at https://test.pypi.org/manage/account/token/
- Set environment variable: ``export UV_PUBLISH_TOKEN_TEST=your_test_token_here``

Release Steps
-------------

**Step 1: Update Version**

.. code-block:: bash

   git checkout main && git pull origin main

   # Choose one of these version bump commands:
   uv version --bump patch  # for patch release (0.1.2 -> 0.1.3)
   uv version --bump minor  # for minor release (0.1.2 -> 0.2.0)
   uv version --bump major  # for major release (0.1.2 -> 1.0.0)

   # Get the new version number
   VERSION=$(uv version | cut -d' ' -f2)

**Step 2: Test the Build**

.. code-block:: bash

   # Build the package
   uv build

   # Test installation
   uv run --with scxpand --no-project -- python -c "import scxpand; print('Import successful')"

**Step 3: Commit and Push Changes**

.. code-block:: bash

   git add -A && git commit -m "Bump version to $VERSION"
   git push origin main

**Step 4: Create and Push Tag**

.. code-block:: bash

   git tag v$VERSION && git push origin v$VERSION

**Step 5: Publish to PyPI**

.. code-block:: bash

   # Publish to PyPI
   uv publish

   # Optional: Publish to TestPyPI first for testing
   # UV_PUBLISH_TOKEN=$UV_PUBLISH_TOKEN_TEST uv publish --index testpypi

**Step 6: Verify Release**

- Check that your package appears at https://pypi.org/project/scxpand/
- Test installation: ``pip install scxpand==$VERSION``

**Release Branch Process (For Major Releases):**

.. code-block:: bash

   # 1. Create release branch
   git checkout main && git pull origin main
   git checkout -b release/v$VERSION

   # 2. Update version and changelog
   # Manually edit pyproject.toml to update version number
   # Edit CHANGELOG.md with new version details
   git add -A && git commit -m "Bump version to $VERSION"
   git push --set-upstream origin release/v$VERSION

   # 3. Create PR, get approval, merge to main

   # 4. Tag and publish (after PR merge)
   git checkout main && git pull origin main
   git tag v$VERSION && git push origin v$VERSION
   uv publish

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
