#!/bin/bash

# PyPI Cleanup Wrapper Script
# Provides convenient shortcuts for common cleanup operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/cleanup_pypi.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
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

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  list                   List all versions without deleting anything"
    echo "  clean                  Clean old versions (dry run by default)"
    echo "  clean-for-real         Actually delete old versions (use with caution!)"
    echo "  keep-3                 Keep only the 3 most recent versions (dry run)"
    echo "  keep-5                 Keep only the 5 most recent versions (dry run)"
    echo "  help                   Show this help message"
    echo ""
    echo "Options (passed to Python script):"
    echo "  --keep-latest N        Keep N most recent versions"
    echo "  --keep-pattern PATTERN Keep versions matching pattern"
    echo "  --package-name NAME    Package name (default: scxpand)"
    echo ""
    echo "Examples:"
    echo "  $0 list                           # List all versions"
    echo "  $0 clean --keep-latest 5          # Dry run keeping 5 versions"
    echo "  $0 clean-for-real --keep-latest 2 # Actually delete, keep 2"
    echo "  $0 keep-3                         # Quick dry run keeping 3"
    echo ""
    echo "Environment Setup:"
    echo "  export PYPI_TOKEN=your_token_here"
    echo "  OR edit scripts/pypi_token.txt"
}

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if requests is available
if ! python3 -c "import requests" 2>/dev/null; then
    print_error "requests library is required. Install with:"
    echo "  pip install requests"
    exit 1
fi

# Parse command
COMMAND="${1:-help}"
shift

case "$COMMAND" in
    list)
        print_info "Listing all package versions..."
        python3 "$PYTHON_SCRIPT" --dry-run --keep-latest 0 "$@"
        ;;
    clean)
        print_info "Cleaning old versions (dry run)..."
        python3 "$PYTHON_SCRIPT" --dry-run "$@"
        ;;
    clean-for-real)
        print_warning "This will actually delete versions from PyPI!"
        print_warning "Make sure you have backups and are certain about this action."
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Proceeding with actual deletion..."
            python3 "$PYTHON_SCRIPT" --no-dry-run "$@"
        else
            print_info "Cancelled."
        fi
        ;;
    keep-3)
        print_info "Dry run: keeping 3 most recent versions..."
        python3 "$PYTHON_SCRIPT" --dry-run --keep-latest 3 "$@"
        ;;
    keep-5)
        print_info "Dry run: keeping 5 most recent versions..."
        python3 "$PYTHON_SCRIPT" --dry-run --keep-latest 5 "$@"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
