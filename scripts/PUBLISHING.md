# Publishing Guide

This guide explains how to publish new versions of scXpand to PyPI.

**scXpand publishes TWO packages:**
- `scxpand` - Standard version (CPU/MPS support)
- `scxpand-cuda` - CUDA version (NVIDIA GPU support)

## Quick Start

```bash
# Patch release (default)
./scripts/release.sh

# Minor release
./scripts/release.sh --minor

# Major release
./scripts/release.sh --major

# Dry run
./scripts/release.sh --dry-run
```

## Prerequisites

### 1. PyPI Token Setup

**CRITICAL**: Use "Entire account" scope (not project-specific) because scXpand publishes TWO packages.

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. **Set scope to "Entire account"**
4. Copy the token immediately

**Configure the token:**
```bash
# Option 1: Edit scripts/pypi_token.txt and replace the dummy token
# Option 2: export UV_PUBLISH_TOKEN=pypi-your_actual_token_here
```

### 2. GitHub CLI (Optional)
```bash
brew install gh  # macOS
gh auth login
```

### 3. Repository State
- Must be on `main` branch
- Working directory clean (no uncommitted changes)
- Up to date with remote origin/main
- No unpushed commits

## Release Process

The `release.sh` script automates the complete release pipeline:

1. **Prerequisites Check**: Verifies git repo, `uv` installation, PyPI token, branch state
2. **Version Management**: Bumps version using `uv version --bump [patch|minor|major]`
3. **Changelog Management**:
   - Checks for existing changelog entry for new version
   - If missing/empty, creates template and exits
   - Automatically moves content from `[Unreleased]` section
   - You edit CHANGELOG.md manually, then re-run script
4. **Package Building**: Builds both `scxpand` (CPU/MPS) and `scxpand-cuda` (CUDA 12.8)
5. **Git Operations**: Commits changes, pushes to main, creates version tag
6. **GitHub Release**: Creates release with auto-generated notes (if GitHub CLI available)
7. **Publishing**: Publishes both packages to PyPI
8. **Verification**: Tests installation from PyPI

## Changelog Workflow

1. **First run**: Script checks for changelog entry, creates template if missing
2. **Edit**: Manually edit CHANGELOG.md with your changes
3. **Second run**: Script validates entry and continues with release

**Example changelog entry:**
```markdown
## [0.3.5] - 2025-01-15

- New hyperparameter optimization for MLP models
- Support for custom loss functions
- Improved memory efficiency in data loading
- Fixed CUDA memory leak in autoencoder training
```

## CUDA Package Configuration

The `scxpand-cuda` package uses:
- **CUDA Version**: CUDA 12.8 (`cu128`)
- **PyTorch Index**: `https://download.pytorch.org/whl/cu128`
- **Configuration**: Uses `[tool.uv.sources]` and `[[tool.uv.index]]` in pyproject.toml
- **Platform Support**: Windows and Linux (macOS falls back to CPU PyTorch)

To change CUDA version:
```bash
python scripts/create_cuda_pyproject.py -c cu124  # example for CUDA 12.4
```

## Troubleshooting

**"403 Forbidden" or "project-scoped token is not valid"**:
- Your token is project-scoped but scXpand publishes TWO packages
- Create new token with "Entire account" scope

**"Invalid token" or "Authentication failed"**:
- Check token starts with `pypi-`
- Verify token is complete and copied correctly

**"Not on main branch"**: `git checkout main`

**"Working directory is not clean"**: `git add . && git commit -m "Your changes"`

**"Local branch is not up to date"**: `git pull origin main`

**"There are unpushed commits"**: `git push origin main`

**"CUDA PyTorch not installing correctly"**:
- Verify generated CUDA configuration in `temp/pyproject-cuda.toml`
- Check `torch.__version__` includes `+cu128` after installation

## Key Points

- **Dual Package Publishing**: Always publishes both `scxpand` and `scxpand-cuda`
- **Account-Scoped Token Required**: Project-scoped tokens fail for second package
- **Automated Changelog**: Script handles version bumping and template creation
- **CUDA Configuration**: Automatically generated with proper PyTorch index setup
- **Complete Automation**: From version bump to PyPI publishing in one command
