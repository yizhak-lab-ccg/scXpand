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
  - [Changelog Workflow](#changelog-workflow)
  - [PyPI Token Requirements](#pypi-token-requirements)
- [Troubleshooting](#troubleshooting)

## Release Script

### `release.sh` (Dual Package Release Script)

**Fully automated release pipeline** for both package variants with:
- 📝 **Smart changelog management** with automatic [Unreleased] content migration
- 🚀 **Automatic GitHub release creation** with auto-generated release notes
- 📦 **Dual package publishing** (CPU/MPS + CUDA variants)
- 🔄 **Complete automation** from version bump to PyPI publishing

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

   **GitHub CLI Setup (Optional - for GitHub Releases):**

   ```bash
   # Install GitHub CLI
   brew install gh  # macOS
   # or visit https://cli.github.com/ for other platforms

   # Authenticate with GitHub
   gh auth login
   ```

   *Note: GitHub CLI is optional. If not available, the script will skip GitHub release creation.*


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
   - Checks for GitHub CLI availability (optional for GitHub releases)
   - Confirms `UV_PUBLISH_TOKEN` is set
   - Ensures you're on the `main` branch
   - Verifies working directory is clean
   - Checks if local branch is up to date with remote
   - Ensures no unpushed commits exist

2. **Version Management**:
   - Bumps version using `uv version --bump [patch|minor|major]`
   - Shows current and new version numbers
   - Supports patch (default), minor, and major version bumps

3. **📝 Changelog Validation**:
   - **Checks for existing changelog entry** for the new version
   - If no entry exists or it's empty, **creates a template** and exits
   - **Automatically moves content from [Unreleased] section** to the new version
   - **Simple bullet-point format** for easy maintenance
   - You manually edit CHANGELOG.md with your preferred editor
   - **Re-run the script** after filling in the changelog

4. **Package Building**:
   - Cleans build directories
   - Builds standard package (`scxpand`) with CPU/MPS PyTorch support
   - Creates CUDA variant configuration using Python script (`scripts/create_cuda_pyproject.py`)
   - Builds CUDA package (`scxpand-cuda`) with CUDA 12.8 PyTorch support using `[tool.uv.sources]` and `[[tool.uv.index]]` configuration
   - Tests import functionality for both packages

5. **Git Operations**:
   - Commits version bump and changelog changes
   - Pushes to main branch
   - Creates and pushes version tag

6. **🚀 GitHub Release Creation** (if GitHub CLI is available):
   - Automatically creates GitHub release with version tag
   - Generates release notes from changelog content
   - Includes installation instructions for both packages
   - Links to documentation and PyPI packages

7. **Publishing**:
   - Publishes both packages to PyPI using `uv publish`
   - Includes confirmation prompt before publishing
   - Requires "Entire account" scoped token

8. **Verification**:
   - Tests installation of both packages from PyPI
   - Verifies that `scxpand-cuda` installs CUDA-enabled PyTorch (version should show `+cu128`)
   - Provides links to PyPI and GitHub release

### CUDA Package Configuration

The `scxpand-cuda` package uses a sophisticated configuration to ensure CUDA PyTorch installation:

- **CUDA Version**: Uses CUDA (`cu128`) - the latest supported version
- **PyTorch Index**: Configured with `[[tool.uv.index]]` pointing to `https://download.pytorch.org/whl/cu128`
- **Package Sources**: Uses `[tool.uv.sources]` to direct `torch`, `torchvision`, and `torchaudio` to the CUDA index
- **Platform Support**: Works on Windows and Linux (macOS falls back to CPU PyTorch as CUDA is not available)
- **Build Safety**: Uses `explicit = true` flag to prevent the CUDA index from being used for non-PyTorch dependencies

*Changing CUDA version*: regenerate the CUDA configuration with:

```bash
python scripts/create_cuda_pyproject.py -c cu124  # example for CUDA 12.4
```

## Changelog Workflow

### How It Works

1. **First Run**: The script checks if CHANGELOG.md has an entry for the new version
2. **If missing**: Creates a template and exits with instructions
3. **Manual editing**: You edit CHANGELOG.md with your preferred editor
4. **Second run**: Script validates the entry and continues with the release

### Step-by-Step Process

#### 1. Run the Release Script
```bash
./scripts/release.sh --patch
```

#### 2. If No Changelog Entry Exists
The script will:
- Bump the version (e.g., 0.3.4 → 0.3.5)
- Check CHANGELOG.md for a `## [0.3.5]` section
- If not found or empty, create a template and exit with:

```
[ERROR] CHANGELOG.md has been updated with a template for version 0.3.5
Please:
  1. Edit CHANGELOG.md and fill in the release notes
  2. Add meaningful bullet points describing the changes
  3. Replace the placeholder '-' with actual change descriptions
  4. Run the release script again: ./scripts/release.sh --patch
```

#### 3. Edit the Changelog Template

The script automatically moves content from the `[Unreleased]` section to the new version. If there's no unreleased content, it creates a simple template like this:

```markdown
## [0.3.5] - 2025-01-15

```

**Edit with your preferred editor:**
```bash
# Use any editor you prefer
code CHANGELOG.md        # VS Code
vim CHANGELOG.md         # Vim
nano CHANGELOG.md        # Nano
```

**Example of a well-filled changelog:**
```markdown
## [0.3.5] - 2025-01-15

- New hyperparameter optimization for MLP models
- Support for custom loss functions
- Improved memory efficiency in data loading
- Fixed CUDA memory leak in autoencoder training
- Fixed validation metric calculation bug
```

#### 4. Re-run the Release Script
```bash
./scripts/release.sh --patch
```

The script will now:
- Validate that the changelog has content
- Continue with building, publishing, and creating GitHub releases

### Tips for Good Changelog Entries

- **Be specific**: "Fixed CUDA memory leak" vs "Fixed bug"
- **User-focused**: Describe impact on users, not internal changes
- **One feature per line**: Each bullet should be one distinct change
- **Use active voice**: "Added support for..." vs "Support was added for..."
- **Use the [Unreleased] section**: Add changes as you work, they'll be automatically moved during release

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

**"CUDA PyTorch not installing correctly"**:
- Verify the generated CUDA configuration in `temp/pyproject-cuda.toml`
- Check that `[tool.uv.sources]` and `[[tool.uv.index]]` sections are present
- Ensure CUDA index URL points to `https://download.pytorch.org/whl/cu128`
- Test CUDA installation manually: `uv pip install scxpand-cuda==0.1.40` and check `torch.__version__` includes `+cu128`

**"Version detection issues"**:
- The package uses a robust version detection system that tries `scxpand-cuda` first, then `scxpand`, then falls back to `"0.0.0"`
- Both `scxpand` and `scxpand-cuda` packages can be imported as `import scxpand` regardless of which is installed
