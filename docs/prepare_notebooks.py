#!/usr/bin/env python3
"""
Prepare notebooks for documentation build.
This script ensures notebooks are available in docs/_notebooks/ for both local and ReadTheDocs builds.
"""

import shutil
import sys

from pathlib import Path


def prepare_notebooks():
    """Copy notebooks from project root to docs/_notebooks/ if needed."""

    # Get paths
    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent
    source_notebooks_dir = project_root / "notebooks"
    target_notebooks_dir = docs_dir / "_notebooks"

    print("📓 Preparing notebooks...")
    print(f"   Source: {source_notebooks_dir}")
    print(f"   Target: {target_notebooks_dir}")

    # Create target directory if it doesn't exist
    target_notebooks_dir.mkdir(exist_ok=True)

    # Check if source notebooks directory exists
    if not source_notebooks_dir.exists():
        print(f"   ⚠️  Warning: Source notebooks directory not found at {source_notebooks_dir}")
        print("   Notebooks may already be in target directory or missing entirely.")
        return

    # Copy notebooks
    notebook_count = 0
    for notebook in source_notebooks_dir.glob("*.ipynb"):
        target_file = target_notebooks_dir / notebook.name

        # Only copy if source is newer or target doesn't exist
        if not target_file.exists() or notebook.stat().st_mtime > target_file.stat().st_mtime:
            shutil.copy2(notebook, target_file)
            print(f"   Copied {notebook.name}")
            notebook_count += 1
        else:
            print(f"   Skipped {notebook.name} (up to date)")

    if notebook_count > 0:
        print(f"   ✅ Updated {notebook_count} notebook(s)")
    else:
        print("   ✅ All notebooks are up to date")


if __name__ == "__main__":
    try:
        prepare_notebooks()
    except Exception as e:
        print(f"❌ Error preparing notebooks: {e}")
        sys.exit(1)
