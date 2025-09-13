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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script publishes both standard (scxpand - CPU/MPS) and CUDA-enabled (scxpand-cuda) packages."
    echo ""
    echo "Options:"
    echo "  --dry-run          Perform a dry run (no actual changes)"
    echo "  --patch            Bump patch version (default)"
    echo "  --minor            Bump minor version"
    echo "  --major            Bump major version"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Patch release for both packages"
    echo "  $0 --dry-run       # Patch release dry run"
    echo "  $0 --minor         # Minor release for both packages"
    echo "  $0 --major --dry-run # Major release dry run"
    echo ""
    echo "Environment Setup:"
    echo "  Option 1: Edit scripts/pypi_token.txt and replace the dummy token"
    echo "  Option 2: export UV_PUBLISH_TOKEN=your_token_here"
    echo ""
    echo "PyPI Token Requirements:"
    echo "  - Use 'Entire account' scoped token (not project-specific)"
    echo "  - scXpand publishes TWO packages: 'scxpand' and 'scxpand-cuda'"
    echo "  - Project-scoped tokens only work for one package"
    echo "  - Get token at: https://pypi.org/manage/account/token/"
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites for dual package release..."

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Check if uv is installed
    if ! command_exists uv; then
        print_error "uv is not installed. Please install uv first."
        exit 1
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

    # Check if we're on main branch
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "main" ]; then
        print_error "Not on main branch (currently on: $current_branch)"
        print_status "Please checkout main branch first: git checkout main"
        exit 1
    fi

    # Check if working directory is clean
    if ! git diff-index --quiet HEAD --; then
        print_error "Working directory is not clean. Please commit or stash changes first."
        print_status "Current git status:"
        git status --porcelain
        exit 1
    fi

    # Check if we're up to date with remote
    git fetch origin
    local_commit=$(git rev-parse HEAD)
    remote_commit=$(git rev-parse origin/main)
    if [ "$local_commit" != "$remote_commit" ]; then
        print_error "Local branch is not up to date with remote origin/main"
        print_status "Please pull latest changes: git pull origin main"
        exit 1
    fi

    # Check if there are any unpushed commits
    if ! git diff --quiet origin/main..HEAD; then
        print_error "There are unpushed commits"
        print_status "Please push your changes first: git push origin main"
        exit 1
    fi

    # Check if pyproject file exists
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to bump version
bump_version() {
    print_status "Bumping $VERSION_TYPE version..."

    # Get current version from main pyproject.toml ()
    current_version=$(uv version | cut -d' ' -f2)
    print_status "Current version: $current_version"

    # Bump version in main pyproject.toml
    if [ "$DRY_RUN" = false ]; then
        uv version --bump "$VERSION_TYPE"
    fi

    # Get new version
    new_version=$(uv version | cut -d' ' -f2)
    print_success "Version bumped to: $new_version"

    # Export for later use
    export NEW_VERSION="$new_version"
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

# Function to backup and restore pyproject.toml
backup_pyproject() {
    cp pyproject.toml pyproject.toml.backup
}

restore_pyproject() {
    if [ -f pyproject.toml.backup ]; then
        mv pyproject.toml.backup pyproject.toml
    fi
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

# Function to build standard package
build_standard_package() {
    print_package "Building standard package (scxpand - CPU/MPS support)..."

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would build standard package with CPU/MPS support"
        return
    fi

    # Build using original pyproject.toml
    if ! uv build; then
        print_error "Failed to build standard package"
        return 1
    fi

    print_success "Standard package built successfully"
}

# Function to build CUDA package
build_cuda_package() {
    print_package "Building CUDA package (scxpand-cuda - CUDA support)..."

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would build CUDA package with CUDA support"
        return
    fi

    # Backup original pyproject.toml
    backup_pyproject

    # Create CUDA variant configuration
    create_cuda_pyproject

    # Replace the original with CUDA version temporarily
    mv temp/pyproject-cuda.toml pyproject.toml

    # Build using CUDA configuration
    if ! uv build; then
        print_error "Failed to build CUDA package"
        restore_pyproject
        return 1
    fi

    # Restore original pyproject.toml
    restore_pyproject

    print_success "CUDA package built successfully"
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

# Function to commit and push changes
commit_and_push() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would commit and push changes..."
        return
    fi

    print_status "Committing and pushing changes..."

    # Add all changes
    git add -A

    # Commit with version bump message
    git commit -m "Bump version to $NEW_VERSION (dual package release)"

    # Push to main
    git push origin main

    print_success "Changes committed and pushed to main"
}

# Function to create and push tag
create_and_push_tag() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would create and push tag v$NEW_VERSION..."
        return
    fi

    print_status "Creating and pushing tag..."

    # Create tag
    git tag "v$NEW_VERSION"

    # Push tag
    git push origin "v$NEW_VERSION"

    print_success "Tag v$NEW_VERSION created and pushed"
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
    read -p "Are you sure you want to continue? (y/N): " -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
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

# Function to trigger ReadTheDocs build
trigger_readthedocs_build() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would trigger ReadTheDocs build..."
        return
    fi

    print_status "Triggering ReadTheDocs documentation build..."

    # ReadTheDocs will automatically build when we push the tag
    # But we can also trigger it manually via webhook if configured
    print_status "ReadTheDocs will automatically build documentation from the new tag v$NEW_VERSION"
    print_status "Documentation will be available at: https://scxpand.readthedocs.io/"

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
        print_status "What would happen in a real dual release:"
        echo "  - Version: $NEW_VERSION ($VERSION_TYPE)"
        echo "  - Packages: scxpand (CPU/MPS) and scxpand-cuda (CUDA)"
        echo "  - Commit message: 'Bump version to $NEW_VERSION (dual package release)'"
        echo "  - Tag: v$NEW_VERSION"
        echo "  - Push to main branch"
        echo "  - Publish both packages to PyPI"
        echo "  - Trigger ReadTheDocs documentation build"
        echo
        print_warning "This was a DRY RUN - no actual changes were made to git or PyPI"
        print_status "To perform the actual release, run: ./scripts/release.sh --$VERSION_TYPE"
    else
        print_success "Dual package $VERSION_TYPE release completed successfully!"
        echo
        print_status "Summary:"
        echo "  - Version: $NEW_VERSION ($VERSION_TYPE)"
        echo "  - Tag: v$NEW_VERSION"
        echo "  - Standard Package (CPU/MPS): https://pypi.org/project/scxpand/"
        echo "  - CUDA Package:               https://pypi.org/project/scxpand-cuda/"
        echo "  - Documentation:              https://scxpand.readthedocs.io/"
        echo "  - GitHub: https://github.com/yizhak-lab-ccg/scXpand/releases/tag/v$NEW_VERSION"
    fi
    echo
}

# Main execution
main() {
    echo "=========================================="
    if [ "$DRY_RUN" = true ]; then
        echo "  scXpand Release Script - DRY RUN"
    else
        echo "  scXpand Release Script"
    fi
    echo "=========================================="
    echo
    print_status "Release type: $VERSION_TYPE"
    print_status "Packages: scxpand (CPU/MPS) + scxpand-cuda (CUDA)"
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
    fi
    echo

    # Run all steps
    check_prerequisites
    bump_version
    build_and_test_packages
    show_changes
    commit_and_push
    create_and_push_tag
    publish_to_pypi
    trigger_readthedocs_build
    verify_releases
    show_summary
}

# Trap to ensure cleanup on exit
trap 'restore_pyproject; rm -f temp/pyproject-cuda*.toml pyproject-cuda-temp*.toml' EXIT

# Run main function
main "$@"
