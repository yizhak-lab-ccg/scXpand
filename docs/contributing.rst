Contributing to scXpand
=======================

We welcome contributions to scXpand! This guide will help you contribute effectively to the project.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.


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

We use an automated release script that handles the entire publishing process to PyPI.
For detailed instructions, see :doc:`../scripts/PUBLISHING`.

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
