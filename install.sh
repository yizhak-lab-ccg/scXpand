#!/bin/bash
set -e

REQUIRED_UV_VERSION="0.8.14"

# Read Python version from .python-version file
if [ -f ".python-version" ]; then
    PYTHON_VERSION=$(cat .python-version | tr -d '\n' | tr -d ' ')
    echo "Using Python version from .python-version: $PYTHON_VERSION"
else
    PYTHON_VERSION="3.13"
    echo "No .python-version file found, using default: $PYTHON_VERSION"
fi

function version_ge() {
  # returns 0 if $1 >= $2
  [ "$1" = "$2" ] && return 0
  local IFS=.
  local i ver1=($1) ver2=($2)
  # fill empty fields in ver1 with zeros
  for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
    ver1[i]=0
  done
  for ((i=0; i<${#ver1[@]}; i++)); do
    if [[ -z ${ver2[i]} ]]; then
      # fill empty fields in ver2 with zeros
      ver2[i]=0
    fi
    if ((10#${ver1[i]} > 10#${ver2[i]})); then
      return 0
    fi
    if ((10#${ver1[i]} < 10#${ver2[i]})); then
      return 1
    fi
  done
  return 0
}

echo "Setting up scXpand development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv version $REQUIRED_UV_VERSION or newer by following the official installation guide:"
    echo "https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    echo "Quick install options:"
    echo "  Standalone installer: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  PyPI:                pipx install uv"
    echo "  Homebrew:            brew install uv"
    exit 1
fi

UV_VERSION=$(uv --version | awk '{print $2}')
if ! version_ge "$UV_VERSION" "$REQUIRED_UV_VERSION"; then
    echo "Error: uv version $REQUIRED_UV_VERSION or newer is required. Found version: $UV_VERSION"
    echo "Please update uv by following the official installation guide:"
    echo "https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    echo "Quick update options:"
    echo "  Standalone installer: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  PyPI:                pipx install uv --force"
    echo "  Homebrew:            brew upgrade uv"
    exit 1
fi

echo "Found uv: $UV_VERSION"

# Handle existing .venv directory
if [ -d ".venv" ]; then
    echo "Warning: Existing .venv directory found."
    read -p "Do you want to remove it and create a fresh environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing .venv directory..."
        rm -rf .venv
    else
        echo "Keeping existing .venv directory. Note: This may cause issues if dependencies have changed."
        echo "If you encounter problems, please remove .venv manually and re-run this script."
    fi
fi

# Install Python using uv
echo "Installing Python $PYTHON_VERSION..."
uv python install $PYTHON_VERSION

# Update the lock file to ensure it's current
echo "Updating lock file..."
uv lock

# Create virtual environment and install dependencies with PyTorch backend detection
echo "Creating virtual environment and installing dependencies with optimal PyTorch backend..."
if [ -d ".venv" ]; then
    echo "Using existing virtual environment..."
    uv sync --extra dev --extra docs
else
    echo "Creating new virtual environment..."
    uv sync --extra dev --extra docs
fi

# Configure PyTorch backend based on system capabilities
echo "Configuring PyTorch backend..."
uv run python install_torch.py

# Register Jupyter kernel
echo "Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name=scxpand_env --display-name="scXpand (.venv)" --env PYTHONPATH "$(pwd)"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

echo "Installation complete! Activate the environment with: source .venv/bin/activate"
