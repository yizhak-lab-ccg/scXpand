Contributing to scXpand
=======================

We welcome contributions to scXpand! This guide will help you contribute effectively to the project.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.


Contributing Workflow
---------------------

External Contributor Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Fork the Repository**

Fork the scXpand repository on GitHub by clicking the "Fork" button on the main repository page.

**2. Clone Your Fork**

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/scXpand.git
   cd scXpand

**3. Add Upstream Remote**

This allows you to sync your fork with the main repository. Replace ORIGINAL_OWNER with the main scXpand repo owner (e.g., scXpandTeam):

.. code-block:: bash

   git remote add upstream https://github.com/ORIGINAL_OWNER/scXpand.git

**4. Create Branch**

.. code-block:: bash

   git checkout -b your-feature-name

**5. Make Changes**

Implement your changes with appropriate tests and documentation updates.

**6. Test Your Changes**

.. code-block:: bash

   # Run all tests
   uv run pytest -n auto

   # Run specific tests for your changes
   uv run pytest tests/your_test_file.py

**7. Commit Guidelines**

Write clear, descriptive commit messages.
Keep commits focused on a single change when possible.

**8. Push to Your Fork**

.. code-block:: bash

   git push origin your-feature-name

**9. Create Pull Request**

On GitHub, create a pull request from your fork's branch to the main repository's main branch.

Keeping Your Fork Up to Date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For external contributors, regularly sync your fork with the upstream repository to avoid merge conflicts:

.. code-block:: bash

   # Fetch latest changes from upstream
   git fetch upstream

   # Switch to main branch
   git checkout main

   # Merge upstream changes
   git merge upstream/main

   # Push updated main to your fork
   git push origin main

Maintainer Workflow
~~~~~~~~~~~~~~~~~~~

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

**5. Push Branch**

.. code-block:: bash

   git push origin your-feature-name

**6. Create Pull Request**

On GitHub, create a pull request targeting the main branch.

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
