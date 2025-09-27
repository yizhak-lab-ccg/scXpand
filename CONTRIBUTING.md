# Contributing to scXpand

We welcome contributions to scXpand! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11 or newer
- Git

### Installation

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/scXpand.git
   cd scXpand
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests:**
   ```bash
   pytest
   ```

4. **Run linting:**
   ```bash
   pre-commit run --all-files
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

6. **Push and create a pull request**

## Coding Standards

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **ruff** for additional linting

All tools are configured in `pyproject.toml` and run automatically via pre-commit hooks.

### Type Hints

Please add type hints to all functions and methods:

```python
def example_function(param1: str, param2: int) -> bool:
    """Example function with type hints."""
    return True
```

### Documentation

- Use Google-style docstrings
- Document all public functions and classes
- Include examples in docstrings when helpful

### Testing

- Write tests for new functionality
- Aim for high test coverage
- Use descriptive test names
- Test edge cases and error conditions

## Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

## Reporting Issues

When reporting issues, please include:

- Python version
- scXpand version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

## Getting Help

- Check existing [issues](https://github.com/yizhak-lab-ccg/scXpand/issues)
- Join the [scverse discourse](https://discourse.scverse.org/)
- Create a new issue for bugs or feature requests

## License

By contributing to scXpand, you agree that your contributions will be licensed under the same license as the project (MIT License).
