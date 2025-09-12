# Publishing Guide

This guide explains how to publish new versions of scXpand to PyPI.

## Table of Contents

- [Release Script](#release-script)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [What the script does](#what-the-script-does)
  - [Safety Features](#safety-features)
  - [Example Output](#example-output)
- [Legacy Scripts](#legacy-scripts)
- [Troubleshooting](#troubleshooting)

## Release Script

### `release.sh` (Unified Release Script)

Automates patch, minor, and major releases following the contributing guide.

### Prerequisites

1. **Environment Setup**:

   **Getting a PyPI Token:**

   a. Go to https://pypi.org/account/register/ and create an account (if you don't have one)

   b. Log in to your PyPI account at https://pypi.org/

   c. Go to https://pypi.org/manage/account/token/

   d. Click "Add API token"

   e. Give it a name (e.g., "scXpand releases")

   f. Set the scope to "Entire account" (for publishing to any project)

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

4. **Build Testing**:
   - Builds the package with `uv build`
   - Tests import functionality

5. **Git Operations**:
   - Commits version bump changes
   - Pushes to main branch
   - Creates and pushes version tag

6. **Publishing**:
   - Publishes to PyPI using `uv publish`
   - Includes confirmation prompt before publishing

7. **Verification**:
   - Tests installation from PyPI
   - Provides links to PyPI and GitHub release


## Troubleshooting

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
