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
  - [Interactive Changelog Process](#interactive-changelog-process)
  - [PyPI Token Requirements](#pypi-token-requirements)
- [Troubleshooting](#troubleshooting)

## Release Script

### `release.sh` (Dual Package Release Script)

**Fully automated release pipeline** for both package variants with:
- 📝 **Interactive changelog updates** with organized entry collection
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

3. **📝 Interactive Changelog Update**:
   - **Prompts for changelog entries** organized by category:
     - **Added**: New features
     - **Changed**: Changes in existing functionality
     - **Fixed**: Bug fixes
     - **Removed**: Removed features
   - You can press Enter to skip sections or type entries one per line
   - Type 'done' to finish a section
   - Automatically formats entries with proper markdown
   - Updates CHANGELOG.md with version number and current date
   - Shows preview and waits for confirmation before continuing

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

## Interactive Changelog Process

### When You'll Be Prompted

During the release process, **after version bumping but before building packages**, the script will prompt you to enter changelog entries interactively.

### How It Works

1. **The script will show:**
   ```
   === CHANGELOG ENTRY FOR VERSION 0.3.5 ===

   Please provide changelog entries for this release.
   Press Enter to skip a section, or type entries one per line.
   Type 'done' on a new line when finished with a section.
   ```

2. **You'll be prompted for each category:**

   **Added (new features):**
   ```
   === ADDED (new features) ===
   Enter new features added in this release:
   ```
   - Type each new feature on a separate line
   - Press Enter on an empty line to skip this section
   - Type 'done' to finish this section

   **Changed (changes in existing functionality):**
   ```
   === CHANGED (changes in existing functionality) ===
   Enter changes to existing functionality:
   ```

   **Fixed (bug fixes):**
   ```
   === FIXED (bug fixes) ===
   Enter bug fixes:
   ```

   **Removed (removed features):**
   ```
   === REMOVED (removed features) ===
   Enter removed features:
   ```

3. **Example interaction:**
   ```
   === ADDED (new features) ===
   Enter new features added in this release:
   New hyperparameter optimization for MLP models
   Support for custom loss functions
   done

   === CHANGED (changes in existing functionality) ===
   Enter changes to existing functionality:
   Improved memory efficiency in data loading

   === FIXED (bug fixes) ===
   Enter bug fixes:
   Fixed CUDA memory leak in autoencoder training
   Fixed validation metric calculation bug
   done

   === REMOVED (removed features) ===
   Enter removed features:
   [Press Enter to skip]
   ```

4. **The script will:**
   - Automatically format your entries with markdown bullets
   - Add the current date
   - Update CHANGELOG.md
   - Show you a preview
   - Wait for you to press Enter before continuing

### Tips for Good Changelog Entries

- **Be specific**: "Fixed CUDA memory leak" vs "Fixed bug"
- **User-focused**: Describe impact on users, not internal changes
- **One feature per line**: Each bullet should be one distinct change
- **Use active voice**: "Added support for..." vs "Support was added for..."

### If You Skip All Sections

If you press Enter for all sections without adding entries, the script will add a default entry:
```markdown
### Changed
- Version bump to 0.3.5
```

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
