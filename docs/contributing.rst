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

**Version Management**

We use `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible functionality additions (1.0.0 → 1.1.0)
- **PATCH**: Backward-compatible bug fixes (1.0.0 → 1.0.1)

**Release Steps**

.. code-block:: bash

   # 1. Switch to main and update version
   git checkout main && git pull origin main

   # Bump version
   uv version --bump patch  # or minor/major

   # Create release branch
   git checkout -b release/v$(uv version)

   # 2. Update CHANGELOG.md and build package
   uv build
   git add -A && git commit -m "Bump version to $(uv version)"
   git push --set-upstream origin release/v$(uv version)

   # 3. Create PR, get approval, merge to main

   # 4. Tag and publish
   git checkout main && git pull origin main
   git tag v$(uv version) && git push origin --tags

   # 5. Approve deployment in GitHub Actions

**Publishing Workflow**

- **TestPyPI**: Automatic on every push (for development testing)
- **PyPI**: Manual approval required when pushing git tags (for official releases)

**One-Time Setup for Maintainers**

Configure PyPI Trusted Publishing:

1. **PyPI**: https://pypi.org/manage/account/publishing/
   - Project: ``scxpand``, Owner: ``yizhak-lab-ccg``, Repository: ``scXpand``
   - Workflow: ``release.yml``, Environment: ``pypi``

2. **TestPyPI**: https://test.pypi.org/manage/account/publishing/
   - Same settings, Environment: ``testpypi``

3. **GitHub Environments**: Settings → Environments
   - Create ``pypi`` environment with required reviewers
   - Create ``testpypi`` environment (no special settings)

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs && python build_docs.py --serve
