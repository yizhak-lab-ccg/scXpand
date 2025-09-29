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

# Dev release (automated versioning)
./scripts/release.sh --dev

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
- Must be on `main` branch (for regular releases)
- Dev releases can be run from any branch
- Working directory clean (no uncommitted changes)
- Up to date with remote origin/main
- No unpushed commits

## Release Process

The `release.sh` script automates the complete release pipeline with **VCS-based versioning**:

### Regular Releases (main branch)
1. **Prerequisites Check**: Verifies git repo, `uv` installation, PyPI token, branch state
2. **Version Management**: Uses VCS versioning with `hatch-vcs` - version determined from Git tags
3. **Changelog Management**:
   - Checks for existing changelog entry for new version
   - If missing/empty, creates template and exits
   - Automatically moves content from `[Unreleased]` section
   - You edit CHANGELOG.md manually, then re-run script
4. **VCS Operations**: For VCS versioning, commits only if changes exist, creates and pushes version tag
5. **Package Building**:
   - Builds `scxpand` (CPU/MPS) with correct version from Git tag
   - Builds `scxpand-cuda` (CUDA 12.8) using git staging to maintain clean versioning
6. **Publishing**: Publishes both packages to PyPI (with confirmation prompt)
7. **GitHub Release**: Creates release with auto-generated notes **after** PyPI confirmation
8. **Documentation**: Triggers ReadTheDocs build
9. **Verification**: Tests installation from PyPI

### Dev Releases (any branch)
1. **Automated Versioning**: Automatically increments patch version and adds `.dev0` suffix
2. **Tag Management**: Handles tag conflicts automatically (deletes and recreates if needed)
3. **VCS Compatibility**: Uses `.dev0` format required by `hatch-vcs`
4. **Package Building**: Builds both packages with dev version
5. **Git Operations**: Pushes branch and creates dev tag
6. **Publishing**: Publishes dev packages to PyPI (no GitHub release)

## VCS Versioning

scXpand uses **VCS-based versioning** with `hatch-vcs`:

- **Version Source**: Git tags (e.g., `v0.3.7`, `v0.3.8.dev0`)
- **Automatic Detection**: Version determined from latest Git tag
- **Dev Versions**: Only `.dev0` format supported (e.g., `0.3.8.dev0`)
- **Tag Format**: Must start with `v` (e.g., `v0.3.7`, `v0.3.8.dev0`)

**Benefits:**
- No manual version bumping in `pyproject.toml`
- Automatic version detection from Git history
- Consistent versioning across all environments
- Support for development versions

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

## Dev Release Automation

Dev releases are **fully automated**:

```bash
# Create automated dev release
./scripts/release.sh --dev

# Preview what would happen
./scripts/release.sh --dev --dry-run
```

**Automation Features:**
- **Auto-Version Increment**: Automatically increments patch version when `.dev0` tag exists
- **Tag Conflict Handling**: Automatically deletes and recreates existing tags
- **VCS Compatibility**: Uses `.dev0` format required by `hatch-vcs`
- **Zero Manual Intervention**: Just run the command!

**Example Version Progression:**
- `v0.3.7` → `v0.3.8.dev0` (first dev release)
- `v0.3.8.dev0` → `v0.3.9.dev0` (next dev release)
- `v0.3.9.dev0` → `v0.3.10.dev0` (and so on...)

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

### Common Issues and Solutions

#### "Working directory is not clean" Error
- **Cause**: Uncommitted changes in working directory
- **Solution**: Commit or stash changes before running release script

#### PyPI Publishing Permission Denied
- **Cause**: Using project-scoped token instead of account-scoped token
- **Solution**: Create "Entire account" scoped token at https://pypi.org/manage/account/token/

### Getting Help

If you encounter issues not listed here:
1. Run with `--dry-run` first to preview changes
2. Check that all prerequisites are met
3. Ensure working directory is clean and up-to-date
4. Verify PyPI token has "Entire account" scope
