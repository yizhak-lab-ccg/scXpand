#!/bin/bash

# Build script for creating both standard and CUDA variants of scXpand packages
# Usage: ./scripts/build_packages.sh [--dry-run] [--clean]

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
CLEAN=false

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
    echo "This script builds both standard (scxpand - CPU/MPS) and CUDA-enabled (scxpand-cuda) packages."
    echo ""
    echo "Options:"
    echo "  --dry-run          Show what would be built without actually building"
    echo "  --clean            Clean build directories before building"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Build both packages"
    echo "  $0 --dry-run       # Show what would be built"
    echo "  $0 --clean         # Clean and build both packages"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --clean)
            CLEAN=true
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

# Function to create CUDA variant of pyproject.toml
create_cuda_pyproject() {
    print_status "Creating CUDA variant configuration..."

    # Create CUDA version by modifying the original
    sed \
        -e 's/name = "scxpand"/name = "scxpand-cuda"/' \
        -e 's/Pan-cancer detection of T-cell clonal expansion from single-cell RNA sequencing/Pan-cancer detection of T-cell clonal expansion from single-cell RNA sequencing (CUDA-enabled)/' \
        -e 's/"single-cell", "RNA-seq", "T-cell", "clonal-expansion", "machine-learning", "bioinformatics"/"single-cell", "RNA-seq", "T-cell", "clonal-expansion", "machine-learning", "bioinformatics", "cuda", "gpu"/' \
        -e 's/# PyTorch (CPU\/MPS backend - users can install CUDA variant separately)/# PyTorch with CUDA support/' \
        pyproject.toml > pyproject-cuda-temp.toml
}

# Function to build standard package
build_standard_package() {
    print_package "Building standard package (scxpand - CPU/MPS support)..."

    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would build standard package with CPU/MPS support"
        return
    fi

    # Build using original pyproject.toml (no --config-file needed for default)
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
    mv pyproject-cuda-temp.toml pyproject.toml

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

# Function to test package imports
test_package_imports() {
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would test package imports"
        return
    fi

    print_status "Testing package imports..."

    # Test standard package import
    if uv run --with scxpand --no-project -- python -c "import scxpand; print('Standard package import successful')" >/dev/null 2>&1; then
        print_success "Standard package import test passed"
    else
        print_warning "Standard package import test failed (package may not be available yet)"
    fi
}

# Function to show build summary
show_summary() {
    echo
    if [ "$DRY_RUN" = true ]; then
        print_success "DRY RUN completed successfully!"
        echo
        print_status "What would be built:"
        echo "  - scxpand (standard package with CPU/MPS support)"
        echo "  - scxpand-cuda (CUDA-enabled package)"
        echo
        print_warning "This was a DRY RUN - no packages were actually built"
        print_status "To build packages, run: ./scripts/build_packages.sh"
    else
        print_success "Both package variants built successfully!"
        echo
        print_status "Built packages:"
        echo "  - scxpand (CPU/MPS support)"
        echo "  - scxpand-cuda (CUDA support)"
        echo
        print_status "Package files created in dist/ directory:"
        if [ -d "dist" ]; then
            ls -la dist/ | grep -E "\.(whl|tar\.gz)$" || echo "  No package files found"
        fi
        echo
        print_status "Next steps:"
        echo "  1. Test packages: pip install dist/scxpand-*.whl"
        echo "  2. Publish: ./scripts/release_dual.sh"
    fi
}

# Main execution
main() {
    echo "=" * 60
    if [ "$DRY_RUN" = true ]; then
        echo "  scXpand Package Builder - DRY RUN"
    else
        echo "  scXpand Package Builder"
    fi
    echo "=" * 60
    echo

    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the project root."
        exit 1
    fi

    # Clean build directories if requested
    if [ "$CLEAN" = true ]; then
        clean_build_dirs
    fi

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

    # Show summary
    show_summary
}

# Trap to ensure cleanup on exit
trap 'restore_pyproject; rm -f pyproject-cuda-temp.toml' EXIT

# Run main function
main "$@"
