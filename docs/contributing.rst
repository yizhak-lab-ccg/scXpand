Contributing to scXpand
=======================

We welcome contributions to scXpand! This guide will help you contribute effectively to the project.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.

**Python Version Support**

We support Python 3.11, 3.12, and 3.13. Our CI tests run on all these versions to ensure compatibility.

**Running Tests**

Tests run automatically on every push to any branch, but you can also run them manually with ``uv run pytest``


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
   uv run pytest

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
- [ ] All tests pass locally and in CI
- [ ] Documentation updated if needed
- [ ] PR description clearly explains changes
- [ ] Branch is up-to-date with main

.. note::
   All changes to ``main`` must go through pull requests. Direct pushes are blocked.

Release Process
===============

**For Maintainers Only**

We follow the `uv packaging guide <https://docs.astral.sh/uv/guides/package/>`_ for releases.

Release Process
===============

**For Maintainers Only**

Version Management
------------------

We use `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible functionality additions (1.0.0 → 1.1.0)
- **PATCH**: Backward-compatible bug fixes (1.0.0 → 1.0.1)

One-Time Setup
--------------

Configure PyPI Trusted Publishing:

1. **PyPI**: https://pypi.org/manage/account/publishing/
   - Project: ``scxpand``, Owner: ``yizhak-lab-ccg``, Repository: ``scXpand``
   - Workflow: ``release.yml``, Environment: ``pypi``

2. **TestPyPI**: https://test.pypi.org/manage/account/publishing/
   - Same settings, Environment: ``testpypi``

3. **GitHub Environments**: Settings → Environments
   - Create ``pypi`` environment with required reviewers
   - Create ``testpypi`` environment (no special settings)

Release Steps
-------------

**Simple Release (Recommended):**

.. code-block:: bash

   # 1. Check current version and update manually
   git checkout main && git pull origin main
   uv version  # Check current version
   # Manually edit pyproject.toml to update version number
   # Example: change version = "0.1.2" to version = "0.1.3"

   # 2. Commit version change and tag
   git add pyproject.toml && git commit -m "Bump version to 0.1.3"
   git push origin main

   # 3. Create and push tag (triggers release)
   git tag v0.1.3 && git push origin v0.1.3

   # 4. Approve deployment in GitHub Actions (PyPI environment)

**Release Branch Process (For Major Releases):**

.. code-block:: bash

   # 1. Create release branch
   git checkout main && git pull origin main
   git checkout -b release/v0.1.3

   # 2. Update version and changelog
   # Manually edit pyproject.toml to update version number
   # Edit CHANGELOG.md with new version details
   git add -A && git commit -m "Bump version to 0.1.3"
   git push --set-upstream origin release/v0.1.3

   # 3. Create PR, get approval, merge to main

   # 4. Tag and publish (after PR merge)
   git checkout main && git pull origin main
   git tag v0.1.3 && git push origin v0.1.3

   # 5. Approve deployment in GitHub Actions

Automated Workflows
-------------------

**What Happens Automatically:**

**Every Push to Main:**
- Runs tests
- Publishes to TestPyPI (for testing)

**Tag Push (e.g., v0.1.3):**
- Runs tests
- Publishes to TestPyPI
- Publishes to PyPI (with approval)
- Creates GitHub Release

**Pull Requests:**
- Runs full test matrix
- Pre-commit checks



Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
