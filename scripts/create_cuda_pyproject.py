#!/usr/bin/env python3
"""Create CUDA variant of pyproject.toml for scXpand dual package release.

This script safely modifies the original pyproject.toml to create a CUDA-enabled variant
that forces PyTorch installation from the pytorch-cu128 index.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

# Add the project root to the path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.constants import CUDA_VERSION as DEFAULT_CUDA_VERSION
from scripts.constants import TORCH_VERSION


def load_toml_lines(file_path: Path) -> list[str]:
    """Load TOML file as lines for processing."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def save_toml_lines(lines: list[str], file_path: Path) -> None:
    """Save lines to TOML file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"ERROR: Failed to save {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def validate_toml(file_path: Path) -> bool:
    """Validate that a file is valid TOML."""
    try:
        with open(file_path, "rb") as f:
            tomllib.load(f)
        return True
    except Exception:
        return False


def create_cuda_variant(input_path: Path, output_path: Path, cuda_version: str) -> bool:
    """Create CUDA variant of pyproject.toml.

    Args:
        input_path: Path to original pyproject.toml
        output_path: Path to save CUDA variant
        cuda_version: CUDA version to target (e.g., cu128, cu124)

    Returns:
        True if successful, False otherwise
    """
    # Create CUDA configuration based on version
    pytorch_cuda_index_name = f"pytorch-{cuda_version}"
    pytorch_cuda_index_url = f"https://download.pytorch.org/whl/{cuda_version}"
    print(f"Loading original pyproject.toml from {input_path}")

    # Load original TOML lines
    lines = load_toml_lines(input_path)

    # Validate original TOML
    if not validate_toml(input_path):
        print("ERROR: Original pyproject.toml is not valid TOML")
        return False

    print("Modifying package metadata for CUDA variant...")

    # Create modified lines
    modified_lines = []
    sources_section_found = False
    sources_section_start = -1
    sources_section_end = -1
    in_dependencies = False
    injected_cuda_deps = False

    for i, line in enumerate(lines):
        # Change package name
        if re.match(r'^\s*name\s*=\s*"scxpand"', line):
            original_line = line.strip()
            modified_line = re.sub(
                r'name\s*=\s*"scxpand"', 'name = "scxpand-cuda"', line
            )
            modified_lines.append(modified_line)
            print(
                f"  ✓ Changed package name: {original_line} → {modified_line.strip()}"
            )
            continue

        # Update description
        if re.match(r"^\s*description\s*=", line) and "CUDA-enabled" not in line:
            original_line = line.strip()
            modified_line = re.sub(
                r'(description\s*=\s*"[^"]*)"', r'\1 (CUDA-enabled)"', line
            )
            modified_lines.append(modified_line)
            print("  ✓ Updated description: Added '(CUDA-enabled)' suffix")
            continue

        # Add CUDA keywords
        if re.match(r"^\s*keywords\s*=", line):
            original_line = line.strip()
            # Add cuda and gpu keywords if not present
            if "cuda" not in line.lower():
                modified_line = re.sub(
                    r'(keywords\s*=\s*\[[^\]]*)"([^"]*)"(\s*\])',
                    r'\1"\2", "cuda", "gpu"\3',
                    line,
                )
                modified_lines.append(modified_line)
                print("  ✓ Added CUDA keywords: cuda, gpu")
                continue

        # Remove PyTorch comment lines
        if "PyTorch" in line and ("CPU/MPS" in line or "will use" in line):
            print("  ✓ Removed PyTorch comment line")
            continue

        # Track [tool.uv.sources] section - replace with CUDA variant
        if re.match(r"^\s*\[tool\.uv\.sources\]", line):
            sources_section_found = True
            sources_section_start = i
            print(
                f"  ✓ Found [tool.uv.sources] section at line {i + 1} - will replace with CUDA config"
            )

            # Find end of this section
            for j in range(i + 1, len(lines)):
                if re.match(r"^\s*\[", lines[j]) and not lines[j].strip().startswith(
                    "#"
                ):
                    sources_section_end = j
                    break
            else:
                sources_section_end = len(lines)

            # Replace with CUDA-specific configuration
            modified_lines.append("[tool.uv.sources]\n")
            modified_lines.append(
                f'torch = {{ index = "{pytorch_cuda_index_name}" }}\n'
            )
            modified_lines.append("\n")
            modified_lines.append("[[tool.uv.index]]\n")
            modified_lines.append(f'name = "{pytorch_cuda_index_name}"\n')
            modified_lines.append(f'url = "{pytorch_cuda_index_url}"\n')
            modified_lines.append("explicit = true\n")
            print("  ✓ Added PyTorch CUDA index configuration")

            # Skip the original sources content
            continue

        # Detect dependencies list and replace torch dependency
        if re.match(r"^\s*dependencies\s*=\s*\[", line):
            in_dependencies = True
            modified_lines.append(line)
            continue

        if in_dependencies:
            # end of dependencies list
            if "]" in line:
                # inject CUDA-aware dependencies once
                if not injected_cuda_deps:
                    modified_lines.extend(
                        [
                            f'    "torch>={TORCH_VERSION}",\n',
                        ]
                    )
                    injected_cuda_deps = True
                modified_lines.append(line)
                in_dependencies = False
                continue
            # skip original torch lines and replace with CUDA version
            if re.search(r'"torch', line):
                continue

        # Skip lines that are part of the original sources section
        if sources_section_found and sources_section_start < i < sources_section_end:
            continue

        # Keep all other lines unchanged
        modified_lines.append(line)

    # Add PyTorch CUDA configuration for both uv and pip compatibility
    print("  ✓ Adding PyTorch CUDA configuration for pip compatibility")
    modified_lines.append("\n")

    # Add uv sources configuration (for uv users)
    modified_lines.append("[tool.uv.sources]\n")
    modified_lines.append(f'torch = {{ index = "{pytorch_cuda_index_name}" }}\n')
    modified_lines.append("\n")

    # Add uv index configuration
    modified_lines.append("[[tool.uv.index]]\n")
    modified_lines.append(f'name = "{pytorch_cuda_index_name}"\n')
    modified_lines.append(f'url = "{pytorch_cuda_index_url}"\n')
    modified_lines.append("explicit = true\n")

    # Add installation instructions and metadata
    modified_lines.append("\n")
    modified_lines.append("# CUDA Package Installation Instructions:\n")
    modified_lines.append(
        "# For pip users: pip install scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128\n"
    )
    modified_lines.append(
        "# For uv users: uv pip install scxpand-cuda --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match\n"
    )
    modified_lines.append(
        "# This package requires PyTorch with CUDA support for optimal performance.\n"
    )

    # Save CUDA variant
    print(f"Saving CUDA variant to {output_path}")
    save_toml_lines(modified_lines, output_path)

    # Verify the saved file
    print("Verifying CUDA variant...")
    if not validate_toml(output_path):
        print("ERROR: Generated CUDA variant is not valid TOML")
        return False

    # Check package name
    with open(output_path, encoding="utf-8") as f:
        content = f.read()
        if 'name = "scxpand-cuda"' not in content:
            print("ERROR: Package name verification failed")
            return False

        # Check PyTorch sources (new format with platform markers)
        if f'index = "{pytorch_cuda_index_name}"' not in content:
            print("ERROR: PyTorch sources verification failed")
            return False

    print("  ✓ All verifications passed")
    print("✅ CUDA variant created successfully!")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create CUDA variant of pyproject.toml for scXpand dual package release"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("pyproject.toml"),
        help="Input pyproject.toml file (default: pyproject.toml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("temp/pyproject-cuda.toml"),
        help="Output CUDA variant file (default: temp/pyproject-cuda.toml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--cuda-version",
        "-c",
        default=DEFAULT_CUDA_VERSION,
        help="CUDA version to target, e.g., cu128, cu124, cu121",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"ERROR: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create CUDA variant
    success = create_cuda_variant(args.input, args.output, args.cuda_version)

    if success:
        print(f"\n🎉 Success! CUDA variant saved to: {args.output}")
        if args.verbose:
            print(f"\nFile size: {args.output.stat().st_size} bytes")
            print(f"Lines: {len(args.output.read_text().splitlines())}")
    else:
        print("\n❌ Failed to create CUDA variant", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
