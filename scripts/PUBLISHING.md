# Publishing Guide

This guide explains how to publish new versions of scXpand to PyPI.

**scXpand publishes TWO packages:**
- `scxpand` - Standard version (CPU/MPS support)
- `scxpand-cuda` - CUDA version (NVIDIA GPU support)

## Table of Contents

- [Release Script](#release-script)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [What the script does](#what-the-script-does)
  - [PyPI Token Requirements](#pypi-token-requirements)
- [Troubleshooting](#troubleshooting)

## Release Script

### `release.sh` (Dual Package Release Script)

Automates patch, minor, and major releases for both package variants.

### Prerequisites

1. **Environment Setup**:

   **Getting a PyPI Token:**

   a. Go to https://pypi.org/account/register/ and create an account (if you don't have one)

   b. Log in to your PyPI account at https://pypi.org/

   c. Go to https://pypi.org/manage/account/token/

   d. Click "Add API token"

   e. Give it a name (e.g., "scXpand releases")

   f. **CRITICAL**: Set the scope to "Entire account" (NOT project-specific)

      **Why "Entire account" is required:**
      - scXpand publishes TWO packages: `scxpand` and `scxpand-cuda`
      - Project-scoped tokens only work for one package
      - Account-scoped tokens can publish both packages

   g. Click "Add token"

   h. **Copy the token immediately** - you won't be able to see it again!

   **Using the Token:**

   ```bash
   # Option 1: Use token file (recommended)
   # Edit scripts/pypi_token.txt and replace the dummy token with your actual token

   # Option 2: Set environment variable
   export UV_PUBLISH_TOKEN=pypi-your_actual_token_here
   ```


2. **Repository State**:
   - Must be on `main` branch
   - Working directory must be clean (no uncommitted changes)
   - Must be up to date with remote origin/main
   - No unpushed commits

### Usage

```bash
# From the project root directory

# Patch release (default)
./scripts/release.sh

# Minor release
./scripts/release.sh --minor

# Major release
./scripts/release.sh --major

# Dry run (no actual changes)
./scripts/release.sh --dry-run
./scripts/release.sh --minor --dry-run

# Show help
./scripts/release.sh --help
```

### What the script does

1. **Prerequisites Check**:
   - Verifies you're in a git repository
   - Checks that `uv` is installed
   - Confirms `UV_PUBLISH_TOKEN` is set
   - Ensures you're on the `main` branch
   - Verifies working directory is clean
   - Checks if local branch is up to date with remote
   - Ensures no unpushed commits exist

2. **Pre-commit Verification**:
   - Checks if pre-commit hooks are installed
   - Notes that tests will run automatically on push

3. **Version Management**:
   - Bumps version using `uv version --bump [patch|minor|major]`
   - Shows current and new version numbers
   - Supports patch (default), minor, and major version bumps

4. **Package Building**:
   - Cleans build directories
   - Builds standard package (`scxpand`)
   - Builds CUDA package (`scxpand-cuda`) using modified configuration
   - Tests import functionality

5. **Git Operations**:
   - Commits version bump changes
   - Pushes to main branch
   - Creates and pushes version tag

6. **Publishing**:
   - Publishes both packages to PyPI using `uv publish`
   - Includes confirmation prompt before publishing
   - Requires "Entire account" scoped token

7. **Verification**:
   - Tests installation of both packages from PyPI
   - Provides links to PyPI and GitHub release

## PyPI Token Requirements

**CRITICAL**: scXpand publishes TWO packages and requires specific token configuration.

### Token Scope Requirements

- ✅ **"Entire account"** - Required for dual package publishing
- ❌ **Project-scoped** - Will fail when publishing the second package

### Why Account Scope is Required

1. **Two Packages Published**: `scxpand` and `scxpand-cuda`
2. **Project-scoped Limitation**: Only works for one package
3. **403 Forbidden Error**: Occurs when token scope is insufficient

### Getting the Right Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. **Set scope to "Entire account"**
4. Copy the token immediately

## Troubleshooting

**"403 Forbidden" or "project-scoped token is not valid for project: 'scxpand-cuda'"**:
- Your token is project-scoped but scXpand publishes TWO packages
- Go to https://pypi.org/manage/account/token/ and create a new token
- **Set scope to "Entire account"** (not project-specific)
- This allows publishing both `scxpand` and `scxpand-cuda` packages

**"Invalid token" or "Authentication failed"**:
- Check that your token starts with `pypi-`
- Verify the token is complete (no missing characters)
- Make sure you copied the token correctly from PyPI
- Try creating a new token if the old one might be compromised

**"Token file not found"**:
- Make sure `scripts/pypi_token.txt` exists
- Check that the file contains a line starting with `pypi-`
- Verify file permissions allow reading

**"Not on main branch"**:
- Switch to main: `git checkout main`

**"Working directory is not clean"**:
- Commit or stash changes: `git add . && git commit -m "Your changes"`

**"Local branch is not up to date with remote"**:
- Pull latest changes: `git pull origin main`

**"There are unpushed commits"**:
- Push your changes: `git push origin main`

**"Build failed"**:
- Check for syntax errors or missing dependencies
- Run build manually: `uv build`
