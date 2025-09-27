#!/bin/bash

# scXpand Release Script
# Publishes both standard (scxpand - CPU/MPS) and CUDA-enabled (scxpand-cuda) versions
# Usage: ./scripts/release.sh [--dry-run] [--patch|--minor|--major]
# The CUDA build defaults to CUDA 12.8 (cu128). Override by passing --cuda-version to create_cuda_pyproject.py inside the script.

# Load PyPI token from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/pypi_token.txt"

# Load CUDA version and torch version from constants.py
CUDA_VERSION=$(python3 -c "import sys; sys.path.append('$SCRIPT_DIR'); from constants import CUDA_VERSION; print(CUDA_VERSION)")
TORCH_VERSION=$(python3 -c "import sys; sys.path.append('$SCRIPT_DIR'); from constants import TORCH_VERSION; print(TORCH_VERSION)")

# Repository configuration
REPO_URL="https://github.com/yizhak-lab-ccg/scXpand"

if [ -f "$TOKEN_FILE" ]; then
    # Extract token from file (line starting with 'pypi-')
    UV_PUBLISH_TOKEN=$(grep "^pypi-" "$TOKEN_FILE" | head -1)
    if [ -n "$UV_PUBLISH_TOKEN" ]; then
        export UV_PUBLISH_TOKEN
    fi
fi

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
DRY_RUN=false
VERSION_TYPE="patch"
DEV_RELEASE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_package() {
    echo -e "${PURPLE}[PACKAGE]${NC} $1"
}


# Check if command exists (consolidated)
require_command() {
    local cmd="$1"
    local install_hint="${2:-}"

    if ! command -v "$cmd" >/dev/null 2>&1; then
        print_error "$cmd is not installed"
        if [ -n "$install_hint" ]; then
            print_status "$install_hint"
        fi
        return 1
    fi
    return 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script automates the complete release process for both standard (scxpand - CPU/MPS)"
    echo "and CUDA-enabled (scxpand-cuda) packages, including changelog updates and GitHub releases."
    echo ""
    echo "Options:"
    echo "  --dry-run          Perform a dry run (no actual changes)"
    echo "  --patch            Bump patch version (default)"
    echo "  --minor            Bump minor version"
    echo "  --major            Bump major version"
    echo "  --dev              Create a dev release (no GitHub announcement)"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Patch release for both packages"
    echo "  $0 --dry-run       # Patch release dry run"
    echo "  $0 --minor         # Minor release for both packages"
    echo "  $0 --major --dry-run # Major release dry run"
    echo "  $0 --dev           # Dev release (no GitHub announcement)"
    echo "  $0 --dev --dry-run # Dev release dry run"
    echo ""
    echo "Environment Setup:"
    echo "  PyPI Token (required for publishing):"
    echo "    Option 1: Edit scripts/pypi_token.txt and replace the dummy token"
    echo "    Option 2: export UV_PUBLISH_TOKEN=your_token_here"
    echo ""
    echo "  GitHub CLI (optional for GitHub releases):"
    echo "    Install: brew install gh (macOS) or visit https://cli.github.com/"
    echo "    Authenticate: gh auth login"
    echo ""
    echo "PyPI Token Requirements:"
    echo "  - Use 'Entire account' scoped token (not project-specific)"
    echo "  - scXpand publishes TWO packages: 'scxpand' and 'scxpand-cuda'"
    echo "  - Project-scoped tokens only work for one package"
    echo "  - Get token at: https://pypi.org/manage/account/token/"
    echo ""
    echo "Automated Features:"
    echo "  - Automatic CHANGELOG.md updates with version and date"
    echo "  - GitHub release creation with auto-generated release notes (skipped for --dev)"
    echo "  - Dual package building and publishing (CPU/MPS + CUDA)"
    echo "  - ReadTheDocs documentation build triggering (skipped for --dev)"
    echo ""
    echo "Dev Release Features:"
    echo "  - Publishes packages to PyPI with dev version suffix (e.g., 0.3.6.dev1)"
    echo "  - Skips GitHub release creation and announcement"
    echo "  - Skips ReadTheDocs documentation build"
    echo "  - Useful for testing releases before official announcement"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --patch)
            VERSION_TYPE="patch"
            shift
            ;;
        --minor)
            VERSION_TYPE="minor"
            shift
            ;;
        --major)
            VERSION_TYPE="major"
            shift
            ;;
        --dev)
            DEV_RELEASE=true
            VERSION_TYPE="patch"  # Dev releases are typically patch-based
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites for dual package release..."

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Check required commands
    require_command "uv" "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/" || exit 1

    # Check if GitHub CLI is installed (optional for GitHub releases)
    if ! require_command "gh" "To enable GitHub releases, install gh: brew install gh (macOS) or visit https://cli.github.com/"; then
        print_warning "GitHub CLI (gh) is not installed. GitHub release creation will be skipped."
        export SKIP_GITHUB_RELEASE=true
    elif ! gh auth status >/dev/null 2>&1; then
        print_warning "GitHub CLI is not authenticated. GitHub release creation will be skipped."
        print_status "To enable GitHub releases, run: gh auth login"
        export SKIP_GITHUB_RELEASE=true
    else
        print_success "GitHub CLI is available and authenticated"
        export SKIP_GITHUB_RELEASE=false
    fi

    # Check if UV_PUBLISH_TOKEN is set (only for non-dry-run)
    if [ "$DRY_RUN" = false ] && [ -z "$UV_PUBLISH_TOKEN" ]; then
        print_error "UV_PUBLISH_TOKEN environment variable is not set"
        print_status "Please set it with: export UV_PUBLISH_TOKEN=your_token_here"
        print_status "Or edit scripts/pypi_token.txt with your token"
        print_warning "IMPORTANT: Use an 'Entire account' scoped token for dual package publishing"
        print_status "scXpand publishes both 'scxpand' and 'scxpand-cuda' packages"
        print_status "Get account-scoped token at: https://pypi.org/manage/account/token/"
        exit 1
    fi

    # Check if we're on main branch (skip for dev releases)
    local current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ] && [ "$DEV_RELEASE" = false ]; then
        print_error "Not on main branch (currently on: $current_branch)"
        print_status "Please checkout main branch first: git checkout main"
        exit 1
    elif [ "$current_branch" != "main" ] && [ "$DEV_RELEASE" = true ]; then
        print_warning "Dev release from branch: $current_branch (not main)"
        print_status "This is allowed for dev releases"
    fi

    # Check if working directory is clean
    if ! git diff-index --quiet HEAD --; then
        print_error "Working directory is not clean. Please commit or stash changes first."
        print_status "Current git status:"
        git status --porcelain
        exit 1
    fi

    # Check if we're up to date with remote (skip for dev releases from non-main branches)
    print_status "Fetching latest changes from remote..."
    if ! git fetch origin; then
        print_error "Failed to fetch from remote origin"
        print_status "Please check your network connection and git credentials"
        exit 1
    fi

    if [ "$DEV_RELEASE" = true ] && [ "$current_branch" != "main" ]; then
        print_warning "Dev release from non-main branch: Skipping remote sync check"
    else
        local local_commit=$(git rev-parse HEAD)
        local remote_commit=$(git rev-parse origin/main)
        if [ "$local_commit" != "$remote_commit" ]; then
            print_error "Local branch is not up to date with remote origin/main"
            print_status "Please pull latest changes: git pull origin main, and push your changes: git push origin main"
            exit 1
        fi
    fi

    # Note: Unpushed commits check is redundant since we already verified
    # local and remote commits are identical above

    # Check if pyproject file exists
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found"
        exit 1
    fi

    print_success "Prerequisites check passed"
}


# Function to check if changelog has entry for version
check_changelog_entry() {
    local version="$1"
    local changelog_file="CHANGELOG.md"

    if [ ! -f "$changelog_file" ]; then
        return 1  # Changelog doesn't exist
    fi

    # Check if there's a section for this version
    if ! grep -q "^## \[$version\]" "$changelog_file"; then
        return 1  # No entry for this version
    fi

    # Extract the content for this version and check if it has meaningful content
    local version_content
    version_content=$(sed -n "/^## \[$version\]/,/^## \[/p" "$changelog_file" | sed '$d' | tail -n +3 | sed '/^$/d')

    # Check if the content has any bullet points (actual entries) and is not just a placeholder dash
    if echo "$version_content" | grep -q "^- " && ! echo "$version_content" | grep -q "^-$"; then
        return 0  # Has content
    else
        return 1  # Empty or no meaningful content
    fi
}

# Function to create changelog entry template
create_changelog_template() {
    local version="$1"
    local changelog_file="CHANGELOG.md"
    local today=$(date +%Y-%m-%d)

    if [ ! -f "$changelog_file" ]; then
        print_error "CHANGELOG.md not found"
        return 1
    fi

    # Create a temporary file for the updated changelog
    local temp_changelog=$(mktemp)

    # Extract content from [Unreleased] section
    local unreleased_content
    unreleased_content=$(awk '/^## \[Unreleased\]/{flag=1; next} /^## \[/{flag=0} flag && /^-/{print}' "$changelog_file")

    # Build the changelog template with unreleased content or placeholder
    local changelog_template
    if [ -n "$unreleased_content" ]; then
        changelog_template="## [$version] - $today

$unreleased_content

"
    else
        changelog_template="## [$version] - $today

-

"
    fi

    # Read the changelog and add the template
    {
        # Copy everything up to the [Unreleased] section header
        sed -n '1,/## \[Unreleased\]/p' "$changelog_file"

        # Add empty [Unreleased] section
        echo ""

        # Add the new version template
        echo "$changelog_template"

        # Copy the rest of the changelog starting from the first version section
        awk '/^## \[[0-9]/{flag=1} flag{print}' "$changelog_file"
    } > "$temp_changelog"

    # Replace the original changelog
    mv "$temp_changelog" "$changelog_file"

    return 0
}

# Function to validate changelog entry
validate_changelog() {
    print_status "Checking CHANGELOG.md for version $NEW_VERSION..."

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would check if CHANGELOG.md has entry for version $NEW_VERSION"
        return
    fi

    local changelog_file="CHANGELOG.md"

    # Check if changelog exists
    if [ ! -f "$changelog_file" ]; then
        print_error "CHANGELOG.md not found"
        return 1
    fi

    # Check if there's a meaningful entry for the new version
    if check_changelog_entry "$NEW_VERSION"; then
        print_success "CHANGELOG.md has entry for version $NEW_VERSION"
        return 0
    else
        # No entry exists or it's empty, create template
        print_warning "No changelog entry found for version $NEW_VERSION"
        print_status "Creating changelog template..."

        if create_changelog_template "$NEW_VERSION"; then
            print_success "Created changelog template for version $NEW_VERSION"
            echo
            print_error "CHANGELOG.md has been updated with a template for version $NEW_VERSION"
            print_status "Please:"
            print_status "  1. Edit CHANGELOG.md and fill in the release notes"
            print_status "  2. Add meaningful bullet points describing the changes"
            print_status "  3. Replace the placeholder '-' with actual change descriptions"
            print_status "  4. Run the release script again: $0 $*"
            echo
            print_status "Template location: $changelog_file"
            print_status "Look for the section: ## [$NEW_VERSION] - $(date +%Y-%m-%d)"
            echo
            exit 1
        else
            print_error "Failed to create changelog template"
            return 1
        fi
    fi
}

# Function to create dev version number
create_dev_version() {
    local base_version="$1"
    local dev_counter=1

    # Check if there are existing dev versions for this base version
    if git tag -l "v${base_version}.dev*" | grep -q "v${base_version}.dev"; then
        # Find the highest dev counter
        local max_dev=$(git tag -l "v${base_version}.dev*" | sed "s/v${base_version}.dev//" | sort -n | tail -1)
        if [ -n "$max_dev" ] && [ "$max_dev" -gt 0 ]; then
            dev_counter=$((max_dev + 1))
        fi
    fi

    echo "${base_version}.dev${dev_counter}"
}

# Function to preview version bump and validate changelog
preview_and_validate() {
    if [ "$DEV_RELEASE" = true ]; then
        print_status "Previewing dev release..."
    else
        print_status "Previewing $VERSION_TYPE version bump..."
    fi

    local current_version=$(uv version | cut -d' ' -f2)
    print_status "Current version: $current_version"

    # Validate version format (should be x.y.z or x.y.z.devN)
    if ! echo "$current_version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(\.[a-zA-Z0-9]+)*$'; then
        print_error "Invalid version format: $current_version (expected x.y.z or x.y.z.devN)"
        exit 1
    fi

    # Calculate what the new version would be (without modifying files)
    local new_version
    if [ "$DEV_RELEASE" = true ]; then
        # For dev releases, use the current version as base and add .devN
        local base_version=$(echo "$current_version" | sed 's/\.dev[0-9]*$//')
        new_version=$(create_dev_version "$base_version")
        print_status "Dev release: $current_version → $new_version"
    else
        case "$VERSION_TYPE" in
            "major")
                new_version=$(echo "$current_version" | awk -F. '{print ($1+1)".0.0"}')
                ;;
            "minor")
                new_version=$(echo "$current_version" | awk -F. '{print $1"."($2+1)".0"}')
                ;;
            "patch")
                new_version=$(echo "$current_version" | awk -F. '{print $1"."$2"."($3+1)}')
                ;;
            *)
                print_error "Invalid version type: $VERSION_TYPE"
                exit 1
                ;;
        esac
        print_status "Would bump to version: $new_version"
    fi

    export NEW_VERSION="$new_version"

    # For dev releases, skip changelog validation
    if [ "$DEV_RELEASE" = false ]; then
        # Validate changelog entry BEFORE actually bumping version
        validate_changelog
    else
        print_status "Dev release: Skipping changelog validation"
    fi
}

# Function to bump version (only called after validation passes)
bump_version() {
    if [ "$DEV_RELEASE" = true ]; then
        print_status "Setting dev version..."
    else
        print_status "Bumping $VERSION_TYPE version..."
    fi

    # Get current version from main pyproject.toml
    current_version=$(uv version | cut -d' ' -f2)
    print_status "Current version: $current_version"

    # Backup original pyproject.toml before version bump (for restoration on cancellation)
    if [ "$DRY_RUN" = false ]; then
        backup_file "pyproject.toml" "original"
    fi

    # Handle dev releases differently
    if [ "$DEV_RELEASE" = true ]; then
        if [ "$DRY_RUN" = false ]; then
            # For dev releases, manually set the version in pyproject.toml
            # Extract the base version (remove any existing .dev suffix)
            local base_version=$(echo "$current_version" | sed 's/\.dev[0-9]*$//')
            local dev_version=$(create_dev_version "$base_version")

            # Update version in pyproject.toml using sed
            sed -i.bak "s/^version = \".*\"/version = \"$dev_version\"/" pyproject.toml
            rm -f pyproject.toml.bak

            print_success "Dev version set to: $dev_version"
            export NEW_VERSION="$dev_version"
        else
            print_status "DRY RUN: Would set dev version to: $NEW_VERSION"
        fi
    else
        # Regular version bump
        if [ "$DRY_RUN" = false ]; then
            uv version --bump "$VERSION_TYPE"
        fi

        # Get new version
        if [ "$DRY_RUN" = false ]; then
            new_version=$(uv version | cut -d' ' -f2)
            print_success "Version bumped to: $new_version"
            export NEW_VERSION="$new_version"
        else
            print_status "DRY RUN: Would bump to version: $NEW_VERSION"
        fi
    fi

    # Clean up any backup files created during version bump
    cleanup_backups
}

# Function to clean build directories
clean_build_dirs() {
    print_status "Cleaning build directories..."

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would clean build, dist, and *.egg-info directories"
        return
    fi

    # Remove build directories
    rm -rf build/ dist/ *.egg-info/
    print_success "Build directories cleaned"
}

# Consolidated backup/restore functionality for pyproject.toml
backup_file() {
    local source="$1"
    local backup_suffix="$2"
    local backup_file="${source}.${backup_suffix}"

    if [ -f "$source" ]; then
        cp "$source" "$backup_file"
        return 0
    else
        print_error "Source file $source not found for backup"
        return 1
    fi
}

restore_file() {
    local source="$1"
    local backup_suffix="$2"
    local backup_file="${source}.${backup_suffix}"
    local show_messages="${3:-true}"

    if [ -f "$backup_file" ]; then
        if [ "$show_messages" = "true" ]; then
            print_status "Restoring $source from backup..."
        fi
        mv "$backup_file" "$source"
        if [ "$show_messages" = "true" ]; then
            print_success "File restored from backup"
        fi
        return 0
    else
        if [ "$show_messages" = "true" ]; then
            print_warning "No backup found: $backup_file"
        fi
        return 1
    fi
}

cleanup_backups() {
    # Remove all backup files created during the release process
    rm -f pyproject.toml.original pyproject.toml.backup pyproject.toml.temp
    rm -f pyproject.toml.bak
    rm -f temp/pyproject-cuda*.toml
    rm -f pyproject-cuda-temp*.toml

    # Clean up any temporary build artifacts
    rm -rf temp/test_cuda_env 2>/dev/null || true

    print_status "Cleaned up backup and temporary files"
}



# Function to create CUDA variant of pyproject.toml using Python script
create_cuda_pyproject() {
    print_status "Creating CUDA variant configuration using Python script..."

    # Safeguard: Check if original pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        print_error "Original pyproject.toml not found"
        return 1
    fi

    # Safeguard: Check if Python script exists
    if [ ! -f "scripts/create_cuda_pyproject.py" ]; then
        print_error "Python script scripts/create_cuda_pyproject.py not found"
        return 1
    fi

    # Safeguard: Clean up any existing temp files
    rm -f temp/pyproject-cuda*.toml pyproject-cuda-temp*.toml

    # Run Python script to create CUDA variant
    if ! python3 scripts/create_cuda_pyproject.py \
        --input pyproject.toml \
        --output temp/pyproject-cuda.toml \
        --verbose; then
        print_error "Failed to create CUDA variant using Python script"
        return 1
    fi

    # Safeguard: Verify the file was created
    if [ ! -f "temp/pyproject-cuda.toml" ] || [ ! -s "temp/pyproject-cuda.toml" ]; then
        print_error "CUDA variant file was not created or is empty"
        return 1
    fi

    # Verify CUDA configuration
    verify_cuda_pyproject "temp/pyproject-cuda.toml"

    print_success "CUDA variant pyproject.toml created successfully using Python script"
}

# Function to verify CUDA pyproject.toml configuration
verify_cuda_pyproject() {
    local cuda_file="$1"
    print_status "Verifying CUDA pyproject.toml configuration..."

    # Check package name
    if ! grep -q 'name = "scxpand-cuda"' "$cuda_file"; then
        print_error "CUDA variant missing correct package name"
        return 1
    fi

    # Check CUDA torch dependency
    if ! grep -q "torch==${TORCH_VERSION}+${CUDA_VERSION}" "$cuda_file"; then
        print_error "CUDA variant missing torch==${TORCH_VERSION}+${CUDA_VERSION} dependency"
        print_status "Found torch dependencies:"
        grep "torch" "$cuda_file" || echo "  (none found)"
        return 1
    fi

    # Check [tool.uv.sources] section
    if ! grep -q "\[tool.uv.sources\]" "$cuda_file"; then
        print_error "CUDA variant missing [tool.uv.sources] section"
        return 1
    fi

    # Check PyTorch index configuration
    if ! grep -q "pytorch-${CUDA_VERSION}" "$cuda_file"; then
        print_error "CUDA variant missing pytorch-${CUDA_VERSION} index configuration"
        return 1
    fi

    # Check [[tool.uv.index]] section
    if ! grep -q "\[\[tool.uv.index\]\]" "$cuda_file"; then
        print_error "CUDA variant missing [[tool.uv.index]] section"
        return 1
    fi

    # Check explicit = true
    if ! grep -q "explicit = true" "$cuda_file"; then
        print_error "CUDA variant missing 'explicit = true' in index configuration"
        return 1
    fi

    print_success "CUDA pyproject.toml verification passed"
}

# Consolidated package building with error handling
build_package() {
    local package_type="$1"
    local description="$2"
    local dry_run_message="$3"

    print_package "Building $description..."

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: $dry_run_message"
        return 0
    fi

    # Handle CUDA package special case
    if [ "$package_type" = "cuda" ]; then
        backup_file "pyproject.toml" "backup"
        create_cuda_pyproject || {
            restore_file "pyproject.toml" "backup" "false"
        return 1
        }
        mv temp/pyproject-cuda.toml pyproject.toml
    fi

    # Build package
    if ! uv build; then
        print_error "Failed to build $description"
        print_status "Check for syntax errors in pyproject.toml or missing dependencies"
        print_status "Run 'uv build' manually to see detailed error messages"
        [ "$package_type" = "cuda" ] && restore_file "pyproject.toml" "backup" "false"
        return 1
    fi

    # Restore original pyproject.toml for CUDA package
    [ "$package_type" = "cuda" ] && restore_file "pyproject.toml" "backup" "false"

    print_success "$description built successfully"
    return 0
}

# Function to build standard package
build_standard_package() {
    build_package "standard" "standard package (scxpand - CPU/MPS support)" "Would build standard package with CPU/MPS support"
}

# Function to build CUDA package
build_cuda_package() {
    build_package "cuda" "CUDA package (scxpand-cuda - CUDA support)" "Would build CUDA package with CUDA support"
}

# Function to test package imports and CUDA configuration
test_package_imports() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would test package imports and CUDA configuration"
        return
    fi

    print_status "Testing package imports and CUDA configuration..."

    # Test standard package import
    if uv run --with scxpand --no-project -- python -c "import scxpand; print('Standard package import successful')" >/dev/null 2>&1; then
        print_success "Standard package import test passed"
    else
        print_warning "Standard package import test failed (package may not be available yet)"
    fi

    # Test CUDA package configuration before publishing
    test_cuda_wheel_locally
}

# Function to test CUDA wheel locally before publishing
test_cuda_wheel_locally() {
    print_status "Testing CUDA wheel configuration locally..."

    # Find the CUDA wheel file
    local cuda_wheel=$(find dist/ -name "*scxpand_cuda*.whl" | head -1)
    if [ -z "$cuda_wheel" ]; then
        print_error "CUDA wheel not found in dist/"
        return 1
    fi

    print_status "Testing CUDA wheel: $cuda_wheel"

    # Check if we're on macOS ARM64 (where CUDA PyTorch isn't available)
    local platform=$(uname -s)
    local arch=$(uname -m)
    if [[ "$platform" == "Darwin" && "$arch" == "arm64" ]]; then
        print_warning "Skipping CUDA wheel installation test on macOS ARM64 (CUDA not available)"
        print_status "Verifying CUDA wheel structure and metadata instead..."

        # Verify wheel contents using Python
        if ! python3 -c "
import zipfile
import sys

wheel_path = '$cuda_wheel'
print(f'Checking wheel: {wheel_path}')

with zipfile.ZipFile(wheel_path, 'r') as wheel:
    files = wheel.namelist()

    # Check for required metadata files
    metadata_files = [f for f in files if f.endswith('METADATA')]
    if not metadata_files:
        print('ERROR: No METADATA file found in wheel')
        sys.exit(1)

    # Read and check METADATA
    metadata = wheel.read(metadata_files[0]).decode('utf-8')
    if 'Name: scxpand-cuda' not in metadata:
        print('ERROR: Package name not scxpand-cuda in metadata')
        sys.exit(1)

    if 'torch==${TORCH_VERSION}+${CUDA_VERSION}' not in metadata:
        print('ERROR: CUDA torch dependency not found in metadata')
        sys.exit(1)

    print('✓ CUDA wheel structure and metadata verified')
"; then
            print_error "CUDA wheel metadata verification failed"
            return 1
        fi

        print_success "CUDA wheel metadata verification passed"
        rm -rf "temp/test_cuda_env" 2>/dev/null || true
        return 0
    fi

    # For non-macOS ARM64 platforms, try full installation test
    print_status "Platform supports CUDA, testing full installation..."

    # Create a temporary test environment
    local test_env_dir="temp/test_cuda_env"
    rm -rf "$test_env_dir"
    mkdir -p "$test_env_dir"

    # Install uv if not available
    if ! command -v uv >/dev/null 2>&1; then
        print_status "Installing uv for CUDA wheel testing..."
        if ! pip install uv; then
            print_error "Failed to install uv"
            rm -rf "$test_env_dir"
            return 1
        fi
    fi

    # Create test environment with uv and install CUDA wheel
    print_status "Creating test environment and installing CUDA wheel with uv..."
    if ! (cd "$test_env_dir" && uv venv test_venv && source test_venv/bin/activate && uv pip install "../../$cuda_wheel" --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"); then
        print_error "Failed to install CUDA wheel in test environment with uv"
        rm -rf "$test_env_dir"
        return 1
    fi

    # Test that the wheel has correct torch dependency
    print_status "Verifying torch dependency in CUDA wheel..."
    if ! (cd "$test_env_dir" && source test_venv/bin/activate && python -c "
import pkg_resources
import sys

# Get scxpand-cuda package info
try:
    dist = pkg_resources.get_distribution('scxpand-cuda')
    print(f'Package: {dist.project_name} {dist.version}')

    # Check if torch dependency includes CUDA suffix
    torch_req = None
    for req in dist.requires():
        if req.project_name.lower() == 'torch':
            torch_req = str(req)
            break

    if torch_req:
        print(f'Torch requirement: {torch_req}')
        if '+${CUDA_VERSION}' in torch_req:
            print('✓ CUDA torch dependency found')
            sys.exit(0)
        else:
            print('❌ CUDA torch dependency missing')
            sys.exit(1)
    else:
        print('❌ No torch dependency found')
        sys.exit(1)

except Exception as e:
    print(f'❌ Error checking dependencies: {e}')
    sys.exit(1)
"); then
        print_error "CUDA wheel does not have correct torch dependency"
        rm -rf "$test_env_dir"
        return 1
    fi

    print_success "CUDA wheel configuration verified"
    rm -rf "$test_env_dir"
    return 0
}

# Function to build and test both packages
build_and_test_packages() {
    print_status "Building and testing both packages..."

    # Clean build directories
    clean_build_dirs

    # Build both packages
    build_standard_package
    if [ $? -ne 0 ]; then
        print_error "Standard package build failed"
        exit 1
    fi

    build_cuda_package
    if [ $? -ne 0 ]; then
        print_error "CUDA package build failed"
        exit 1
    fi

    # Test imports
    test_package_imports

    print_success "Both packages built and tested successfully"
}

# Function to show what would be committed (for dry run)
show_changes() {
    if [ "$DRY_RUN" = true ]; then
        print_status "Changes that would be committed:"
        git diff --stat
        echo
        print_status "Files that would be added/modified:"
        git status --porcelain
    fi
}

# Consolidated git operations with error handling
execute_git_command() {
    local command="$1"
    local description="$2"
    local dry_run_message="$3"

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: $dry_run_message"
        return 0
    fi

    print_status "$description..."
    if eval "$command"; then
        print_success "$description completed"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Function to commit and push changes
commit_and_push() {
    local commit_message
    local current_branch=$(git branch --show-current)

    if [ "$DEV_RELEASE" = true ]; then
        commit_message="Dev release version $NEW_VERSION (dual package release)"
    else
        commit_message="Bump version to $NEW_VERSION and update CHANGELOG.md (dual package release)"
    fi

    if [ "$DEV_RELEASE" = true ] && [ "$current_branch" != "main" ]; then
        # For dev releases from non-main branches, just push to current branch
        execute_git_command \
            "git add -A && git commit -m '$commit_message' && git push origin $current_branch" \
            "Committing and pushing dev release changes to $current_branch" \
            "Would commit and push dev release changes to $current_branch"
    else
        # Regular release or dev release from main branch
        execute_git_command \
            "git add -A && git commit -m '$commit_message' && git push origin main" \
            "Committing and pushing changes" \
            "Would commit and push changes"
    fi
}

# Function to create and push tag
create_and_push_tag() {
    execute_git_command \
        "git tag 'v$NEW_VERSION' && git push origin 'v$NEW_VERSION'" \
        "Creating and pushing tag v$NEW_VERSION" \
        "Would create and push tag v$NEW_VERSION"
}

# Function to publish both packages to PyPI
publish_to_pypi() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would publish both packages version $NEW_VERSION to PyPI..."
        return
    fi

    print_status "Publishing both packages to PyPI..."

    # Confirm before publishing
    echo
    print_warning "About to publish BOTH packages (scxpand and scxpand-cuda) version $NEW_VERSION to PyPI"
    echo
    echo
    local user_response
    read -p "Are you sure you want to continue? (y/N): " -r user_response
    echo

    if [[ ! $user_response =~ ^[Yy]$ ]]; then
        print_status "Publishing cancelled"
        exit 0
    fi

    # Publish all packages from dist directory
    print_package "Publishing both packages to PyPI..."
    if ! uv publish dist/*; then
        print_error "Publishing packages to PyPI failed"
        print_status "If you get '403 Forbidden' error, check that your PyPI token has 'Entire account' scope"
        exit 1
    fi
    print_success "Both packages successfully published to PyPI"
}

# Function to generate release notes from changelog
generate_release_notes() {
    local changelog_file="CHANGELOG.md"
    local release_notes_file=$(mktemp)

    # Extract release notes for the current version from changelog
    if [ -f "$changelog_file" ]; then
        # Find the section for the current version and extract it
        sed -n "/## \[$NEW_VERSION\]/,/## \[/p" "$changelog_file" | sed '$d' | tail -n +2 > "$release_notes_file"

        # If no specific notes found, create generic ones
        if [ ! -s "$release_notes_file" ]; then
            cat > "$release_notes_file" << EOF
## What's Changed

This release includes improvements to the scXpand package for single-cell RNA-seq analysis.

### Packages Released
- **scxpand** (CPU/MPS support): Standard version for CPU and Apple Silicon
- **scxpand-cuda** (CUDA support): GPU-accelerated version with CUDA ${CUDA_VERSION} support

### Installation
\`\`\`bash
# Standard version (CPU/MPS)
pip install scxpand==$NEW_VERSION

# CUDA version (GPU)
pip install scxpand-cuda==$NEW_VERSION --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
\`\`\`

For more details, see the [documentation](https://scxpand.readthedocs.io/en/latest/).
EOF
        fi
    else
        # Fallback if no changelog
        cat > "$release_notes_file" << EOF
## scXpand v$NEW_VERSION

This release includes improvements to the scXpand package for single-cell RNA-seq analysis.

### Packages Released
- **scxpand** (CPU/MPS support): Standard version for CPU and Apple Silicon
- **scxpand-cuda** (CUDA support): GPU-accelerated version with CUDA ${CUDA_VERSION} support

### Installation
\`\`\`bash
# Standard version (CPU/MPS)
pip install scxpand==$NEW_VERSION

# CUDA version (GPU)
pip install scxpand-cuda==$NEW_VERSION --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
\`\`\`

For more details, see the [documentation](https://scxpand.readthedocs.io/en/latest/).
EOF
    fi

    echo "$release_notes_file"
}

# Function to create GitHub release
create_github_release() {
    if [ "$DEV_RELEASE" = true ]; then
        print_status "Dev release: Skipping GitHub release creation"
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would create GitHub release v$NEW_VERSION..."
        return
    fi

    if [ "$SKIP_GITHUB_RELEASE" = true ]; then
        print_warning "Skipping GitHub release creation (GitHub CLI not available or not authenticated)"
        return
    fi

    print_status "Creating GitHub release v$NEW_VERSION..."

    # Generate release notes
    local release_notes_file=$(generate_release_notes)

    # Create the GitHub release
    if gh release create "v$NEW_VERSION" \
        --title "scXpand v$NEW_VERSION" \
        --notes-file "$release_notes_file" \
        --latest; then
        print_success "GitHub release v$NEW_VERSION created successfully"
        print_status "Release URL: $REPO_URL/releases/tag/v$NEW_VERSION"
    else
        print_error "Failed to create GitHub release"
        print_status "You can create it manually at: $REPO_URL/releases/new"
    fi

    # Clean up temporary file
    rm -f "$release_notes_file"
}

# Function to trigger ReadTheDocs build
trigger_readthedocs_build() {
    if [ "$DEV_RELEASE" = true ]; then
        print_status "Dev release: Skipping ReadTheDocs build"
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would trigger ReadTheDocs build..."
        return
    fi

    print_status "Triggering ReadTheDocs documentation build..."

    # ReadTheDocs will automatically build when we push the tag
    # But we can also trigger it manually via webhook if configured
    print_status "ReadTheDocs will automatically build documentation from the new tag v$NEW_VERSION"
    print_status "Documentation will be available at: https://scxpand.readthedocs.io/en/latest/"

    # Wait a moment for the webhook to trigger
    sleep 5

    print_success "ReadTheDocs build triggered"
}

# Function to verify both releases
verify_releases() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would verify both releases..."
        return
    fi

    print_status "Verifying both package releases..."

    # Wait a moment for PyPI to update
    sleep 10

    # Test standard package installation
    print_package "Testing standard package installation..."
    if ! pip install "scxpand==$NEW_VERSION" --no-cache-dir; then
        print_warning "Could not install standard package from PyPI immediately (this is normal)"
        print_status "Please check manually: pip install scxpand==$NEW_VERSION"
    else
        print_success "Standard package successfully installed from PyPI"
        pip uninstall scxpand -y
    fi

    # Test CUDA package installation
    print_package "Testing CUDA package installation..."
    if ! pip install "scxpand-cuda==$NEW_VERSION" --no-cache-dir --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}; then
        print_warning "Could not install CUDA package from PyPI immediately (this is normal)"
        print_status "Please check manually: pip install scxpand-cuda==$NEW_VERSION"
    else
        print_success "CUDA package successfully installed from PyPI"
        pip uninstall scxpand-cuda -y
    fi

    print_status "Release verification complete"
    print_status "Check your packages at:"
    print_status "  Standard (CPU/MPS): https://pypi.org/project/scxpand/"
    print_status "  CUDA:               https://pypi.org/project/scxpand-cuda/"
}

# Function to show summary
show_summary() {
    echo
    if [ "$DRY_RUN" = true ]; then
        print_success "DRY RUN completed successfully!"
        echo
        if [ "$DEV_RELEASE" = true ]; then
            print_status "What would happen in a real dev release:"
            echo "  - Preview dev version: $NEW_VERSION"
            echo "  - Skip CHANGELOG.md validation"
            echo "  - Set dev version in pyproject.toml"
            echo "  - Packages: scxpand (CPU/MPS) and scxpand-cuda (CUDA)"
            echo "  - Commit message: 'Dev release version $NEW_VERSION (dual package release)'"
            echo "  - Tag: v$NEW_VERSION"
            echo "  - Push to main branch"
            echo "  - Skip GitHub release creation"
            echo "  - Publish both packages to PyPI"
            echo "  - Skip ReadTheDocs documentation build"
        else
            print_status "What would happen in a real dual release:"
            echo "  - Preview version bump: $NEW_VERSION ($VERSION_TYPE)"
            echo "  - Validate CHANGELOG.md entry (create template if missing)"
            echo "  - Bump version in pyproject.toml"
            echo "  - Packages: scxpand (CPU/MPS) and scxpand-cuda (CUDA)"
            echo "  - Commit message: 'Bump version to $NEW_VERSION and update CHANGELOG.md (dual package release)'"
            echo "  - Tag: v$NEW_VERSION"
            echo "  - Push to main branch"
            echo "  - Create GitHub release with auto-generated notes"
            echo "  - Publish both packages to PyPI"
            echo "  - Trigger ReadTheDocs documentation build"
        fi
        echo
        print_warning "This was a DRY RUN - no actual changes were made to git or PyPI"
        if [ "$DEV_RELEASE" = true ]; then
            print_status "To perform the actual dev release, run: ./scripts/release.sh --dev"
        else
            print_status "To perform the actual release, run: ./scripts/release.sh --$VERSION_TYPE"
        fi
    else
        if [ "$DEV_RELEASE" = true ]; then
            print_success "Dev release completed successfully!"
            echo
            print_status "Summary:"
            echo "  - Version: $NEW_VERSION (dev)"
            echo "  - Tag: v$NEW_VERSION"
            echo "  - Standard Package (CPU/MPS): https://pypi.org/project/scxpand/"
            echo "  - CUDA Package:               https://pypi.org/project/scxpand-cuda/"
            echo "  - GitHub release: SKIPPED (dev release)"
            echo "  - Documentation build: SKIPPED (dev release)"
            echo
            print_success "Dev release published to PyPI without GitHub announcement"
        else
            print_success "Dual package $VERSION_TYPE release completed successfully!"
            echo
            print_status "Summary:"
            echo "  - Version: $NEW_VERSION ($VERSION_TYPE)"
            echo "  - Tag: v$NEW_VERSION"
            echo "  - Standard Package (CPU/MPS): https://pypi.org/project/scxpand/"
            echo "  - CUDA Package:               https://pypi.org/project/scxpand-cuda/"
            echo "  - Documentation:              https://scxpand.readthedocs.io/en/latest/"
            echo "  - GitHub: $REPO_URL/releases/tag/v$NEW_VERSION"
            echo "  - Changelog: $REPO_URL/blob/main/CHANGELOG.md"
            echo
            print_success "CHANGELOG.md entry validated for version $NEW_VERSION"
            if [ "$SKIP_GITHUB_RELEASE" != true ]; then
                print_success "GitHub release automatically created at: $REPO_URL/releases/tag/v$NEW_VERSION"
            fi
        fi
    fi
    echo
}

# Main execution
main() {
    echo "=========================================="
    if [ "$DEV_RELEASE" = true ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  scXpand Dev Release Script - DRY RUN"
        else
            echo "  scXpand Dev Release Script"
        fi
    else
        if [ "$DRY_RUN" = true ]; then
            echo "  scXpand Release Script - DRY RUN"
        else
            echo "  scXpand Release Script"
        fi
    fi
    echo "=========================================="
    echo
    if [ "$DEV_RELEASE" = true ]; then
        print_status "Release type: dev (no GitHub announcement)"
    else
        print_status "Release type: $VERSION_TYPE"
    fi
    print_status "Packages: scxpand (CPU/MPS) + scxpand-cuda (CUDA)"
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
    fi
    echo

    # Run all steps
    check_prerequisites
    preview_and_validate
    bump_version
    build_and_test_packages
    show_changes
    commit_and_push
    create_and_push_tag
    create_github_release
    publish_to_pypi
    trigger_readthedocs_build
    verify_releases
    show_summary

    # Clean up backup files after successful completion
    cleanup_backups
}

# Consolidated cleanup on exit
cleanup_on_exit() {
    # Try to restore files first (in case of script interruption)
    restore_file "pyproject.toml" "backup" "false" 2>/dev/null || true
    restore_file "pyproject.toml" "original" "false" 2>/dev/null || true

    # Clean up all backup and temporary files
    cleanup_backups
    rm -f temp/pyproject-cuda*.toml pyproject-cuda-temp*.toml 2>/dev/null || true
}

# Trap to ensure cleanup on exit
trap cleanup_on_exit EXIT

# Run main function
main "$@"
