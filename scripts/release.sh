#!/bin/bash

# scXpand Release Script
# Unified script for patch, minor, and major releases with optional dry run
# Usage: ./scripts/release.sh [--dry-run] [--patch|--minor|--major]

# Load PyPI token from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/pypi_token.txt"

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run          Perform a dry run (no actual changes)"
    echo "  --patch            Bump patch version (default)"
    echo "  --minor            Bump minor version"
    echo "  --major            Bump major version"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Patch release (normal)"
    echo "  $0 --dry-run       # Patch release dry run"
    echo "  $0 --minor         # Minor release"
    echo "  $0 --major --dry-run # Major release dry run"
    echo ""
    echo "Environment Setup:"
    echo "  Option 1: Edit scripts/pypi_token.txt and replace the dummy token"
    echo "  Option 2: export UV_PUBLISH_TOKEN=your_token_here"
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
    print_status "Checking prerequisites..."

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

    print_success "Prerequisites check passed"
}

# Function to verify pre-commit hooks status
verify_precommit_status() {
    print_status "Verifying pre-commit hooks status..."

    # Check if pre-commit is installed
    if ! command_exists pre-commit; then
        print_warning "pre-commit is not installed. Tests will run via pre-commit hooks on push."
    else
        print_status "Pre-commit hooks will run tests automatically on push."
    fi

    print_success "Pre-commit verification complete"
}

# Function to bump version
bump_version() {
    print_status "Bumping $VERSION_TYPE version..."

    # Get current version
    current_version=$(uv version | cut -d' ' -f2)
    print_status "Current version: $current_version"

    # Bump version
    uv version --bump "$VERSION_TYPE"

    # Get new version
    new_version=$(uv version | cut -d' ' -f2)
    print_success "Version bumped to: $new_version"

    # Export for later use
    export NEW_VERSION="$new_version"
}

# Function to test the build
test_build() {
    print_status "Testing the build..."

    # Build the package
    if ! uv build; then
        print_error "Build failed"
        exit 1
    fi

    # Test installation
    if ! uv run --with scxpand --no-project -- python -c "import scxpand; print('Import successful')"; then
        print_error "Import test failed"
        exit 1
    fi

    print_success "Build and import test passed"
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
    git commit -m "Bump version to $NEW_VERSION"

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

# Function to publish to PyPI
publish_to_pypi() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would publish version $NEW_VERSION to PyPI..."
        return
    fi

    print_status "Publishing to PyPI..."

    # Confirm before publishing
    echo
    print_warning "About to publish version $NEW_VERSION to PyPI"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Publishing cancelled"
        exit 0
    fi

    # Publish to PyPI
    if ! uv publish; then
        print_error "Publishing to PyPI failed"
        exit 1
    fi

    print_success "Successfully published to PyPI"
}

# Function to verify release
verify_release() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would verify release..."
        return
    fi

    print_status "Verifying release..."

    # Wait a moment for PyPI to update
    sleep 5

    # Test installation from PyPI
    if ! pip install "scxpand==$NEW_VERSION" --no-cache-dir; then
        print_warning "Could not install from PyPI immediately (this is normal)"
        print_status "Please check manually: pip install scxpand==$NEW_VERSION"
    else
        print_success "Successfully installed from PyPI"
        # Uninstall to clean up
        pip uninstall scxpand -y
    fi

    print_status "Release verification complete"
    print_status "Check your package at: https://pypi.org/project/scxpand/"
}

# Function to show summary
show_summary() {
    echo
    if [ "$DRY_RUN" = true ]; then
        print_success "DRY RUN completed successfully!"
        echo
        print_status "What would happen in a real release:"
        echo "  - Version: $NEW_VERSION ($VERSION_TYPE)"
        echo "  - Commit message: 'Bump version to $NEW_VERSION'"
        echo "  - Tag: v$NEW_VERSION"
        echo "  - Push to main branch"
        echo "  - Publish to PyPI"
        echo
        print_warning "This was a DRY RUN - no actual changes were made to git or PyPI"
        print_status "To perform the actual release, run: ./scripts/release.sh --$VERSION_TYPE"
    else
        print_success "$VERSION_TYPE release completed successfully!"
        echo
        print_status "Summary:"
        echo "  - Version: $NEW_VERSION ($VERSION_TYPE)"
        echo "  - Tag: v$NEW_VERSION"
        echo "  - PyPI: https://pypi.org/project/scxpand/"
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
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
    fi
    echo

    # Run all steps
    check_prerequisites
    verify_precommit_status
    bump_version
    test_build
    show_changes
    commit_and_push
    create_and_push_tag
    publish_to_pypi
    verify_release
    show_summary
}

# Run main function
main "$@"
