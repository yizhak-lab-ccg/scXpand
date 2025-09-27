Contributing to scXpand
=======================

We welcome contributions to scXpand! For detailed contributing guidelines, please see our `CONTRIBUTING.md <https://github.com/yizhak-lab-ccg/scXpand/blob/main/CONTRIBUTING.md>`_ file.

Quick Start
-----------

1. **Fork and clone the repository**
2. **Install in development mode:**
   .. code-block:: bash
      pip install -e ".[dev]"
3. **Install pre-commit hooks:**
   .. code-block:: bash
      pre-commit install
4. **Create a feature branch and make your changes**
5. **Run tests and linting:**
   .. code-block:: bash
      pytest
      pre-commit run --all-files
6. **Submit a pull request**

Template Updates
----------------

scXpand follows the `scverse cookiecutter template <https://github.com/scverse/cookiecutter-scverse>`_ for consistent ecosystem standards.

**To update from the latest scverse template:**

.. code-block:: bash

    pip install cruft
    cruft update

This will:

* Pull the latest improvements from the scverse template
* Update CI/CD workflows, documentation structure, and tool configurations
* Preserve your custom CUDA PyTorch installation solution
* Keep your project-specific files intact

**Note:** The CUDA PyTorch installation solution is protected from template updates and will continue working as before.

For complete instructions, coding standards, and workflow details, please refer to the `CONTRIBUTING.md <https://github.com/yizhak-lab-ccg/scXpand/blob/main/CONTRIBUTING.md>`_ file.

Development Setup
-----------------

For development setup instructions, see :doc:`installation`.

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
