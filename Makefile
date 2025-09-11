# scXpand Makefile

.PHONY: help test test-coverage install-dev clean

# Default target
help:
	@echo "scXpand Commands:"
	@echo ""
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run all tests in parallel"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "For custom parallel testing:"
	@echo "  uv run pytest -n auto        # Auto-detect CPU cores"
	@echo "  uv run pytest -n 4           # Use 4 workers"
	@echo "  uv run pytest -n auto --cov=scxpand  # With coverage"

# Install development dependencies
install-dev:
	uv sync --dev

# Run all tests in parallel (auto-detect workers)
test:
	uv run pytest -n auto

# Run tests with coverage
test-coverage:
	uv run pytest -n auto --cov=scxpand --cov-report=html --cov-report=term

# Clean artifacts
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
