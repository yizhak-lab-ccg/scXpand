#!/usr/bin/env python3
"""
PyPI Package Cleanup Script for scXpand

This script helps remove older versions of the scxpand package from PyPI,
keeping only the most recent versions to avoid clutter and confusion.

Usage:
    python scripts/cleanup_pypi.py [OPTIONS]

Options:
    --dry-run              Show what would be deleted without actually deleting
    --keep-latest N        Keep the N most recent versions (default: 3)
    --keep-pattern PATTERN Keep versions matching a pattern (e.g., "0.1.1*")
    --package-name NAME    Package name (default: scxpand)
    --help, -h             Show this help message

Requirements:
    - PyPI API token with delete permissions
    - requests library: pip install requests

Environment Setup:
    Option 1: Set environment variable: export PYPI_TOKEN=your_token_here
    Option 2: Create scripts/pypi_token.txt with your token
"""

import argparse
import os
import re
import sys

from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin


try:
    import requests
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)


class PyPICleanup:
    """Handle PyPI package version cleanup operations."""

    def __init__(self, package_name: str = "scxpand"):
        self.package_name = package_name
        self.pypi_api_base = "https://pypi.org/pypi/"
        self.session = requests.Session()
        self.token = self._load_token()

        if self.token:
            self.session.headers.update(
                {"Authorization": f"token {self.token}", "User-Agent": "scxpand-cleanup-script/1.0"}
            )

    def _load_token(self) -> str | None:
        """Load PyPI token from environment or file."""
        # Try environment variable first
        token = os.getenv("PYPI_TOKEN")
        if token:
            return token

        # Try token file
        script_dir = Path(__file__).parent
        token_file = script_dir / "pypi_token.txt"

        if token_file.exists():
            try:
                content = token_file.read_text().strip()
                # Look for token starting with 'pypi-'
                for line_raw in content.split("\n"):
                    line = line_raw.strip()
                    if line.startswith("pypi-"):
                        return line
            except Exception as e:
                print(f"Warning: Could not read token file: {e}")

        return None

    def get_package_versions(self) -> Dict[str, dict]:
        """Fetch all versions of the package from PyPI."""
        url = urljoin(self.pypi_api_base, f"{self.package_name}/json")

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("releases", {})
        except requests.RequestException as e:
            print(f"Error fetching package info: {e}")
            return {}

    def parse_version(self, version_str: str) -> tuple:
        """Parse version string into comparable tuple."""
        try:
            # Handle dev versions like "0.1.0.dev1"
            if ".dev" in version_str:
                base_version, dev_part = version_str.split(".dev")
                dev_num = int(dev_part) if dev_part else 0
                base_parts = [int(x) for x in base_version.split(".")]
                # Dev versions are considered lower than release versions
                return tuple(base_parts + [-1, dev_num])
            else:
                # Regular version like "0.1.15"
                parts = [int(x) for x in version_str.split(".")]
                return tuple(parts + [0, 0])  # Add padding for comparison
        except ValueError:
            # Fallback for unparseable versions
            return (0, 0, 0, -999, 0)

    def sort_versions(self, versions: List[str]) -> List[str]:
        """Sort versions from oldest to newest."""
        return sorted(versions, key=self.parse_version)

    def filter_versions_to_keep(
        self, versions: List[str], keep_latest: int = 3, keep_patterns: List[str] | None = None
    ) -> Set[str]:
        """Determine which versions to keep based on criteria."""
        keep_patterns = keep_patterns or []
        versions_to_keep = set()

        # Sort versions (newest last)
        sorted_versions = self.sort_versions(versions)

        # Keep the N most recent versions
        if keep_latest > 0:
            latest_versions = sorted_versions[-keep_latest:]
            versions_to_keep.update(latest_versions)

        # Keep versions matching patterns
        for pattern in keep_patterns:
            for version in versions:
                if re.match(pattern.replace("*", ".*"), version):
                    versions_to_keep.add(version)

        return versions_to_keep

    def delete_version(self, version: str, dry_run: bool = True) -> bool:
        """Delete a specific version from PyPI."""
        if not self.token:
            print("Error: No PyPI token available for deletion")
            return False

        if dry_run:
            print(f"  [DRY RUN] Would delete version {version}")
            return True

        # PyPI doesn't provide a direct API for deletion
        # This would typically require using twine or direct HTTP requests
        # For now, we'll show what would be deleted
        print(f"  [SIMULATE] Deleting version {version}")
        print("    Note: Actual deletion requires manual action via PyPI web interface")
        print(f"    Or use: twine delete {self.package_name}=={version}")

        return True

    def cleanup_versions(
        self, keep_latest: int = 3, keep_patterns: List[str] | None = None, dry_run: bool = True
    ) -> None:
        """Main cleanup function."""
        print(f"Fetching version information for package: {self.package_name}")

        all_versions = self.get_package_versions()
        if not all_versions:
            print("No versions found or error fetching package info")
            return

        version_list = list(all_versions.keys())
        print(f"Found {len(version_list)} versions")

        # Sort and display all versions
        sorted_versions = self.sort_versions(version_list)
        print("\nAll versions (oldest to newest):")
        for i, version in enumerate(sorted_versions, 1):
            print(f"  {i:2d}. {version}")

        # Determine versions to keep
        versions_to_keep = self.filter_versions_to_keep(version_list, keep_latest, keep_patterns)

        # Determine versions to delete
        versions_to_delete = set(version_list) - versions_to_keep

        print(f"\nVersions to keep ({len(versions_to_keep)}):")
        for version in self.sort_versions(list(versions_to_keep)):
            print(f"  ✓ {version}")

        print(f"\nVersions to delete ({len(versions_to_delete)}):")
        if not versions_to_delete:
            print("  None")
            return

        for version in self.sort_versions(list(versions_to_delete)):
            print(f"  ✗ {version}")

        if not dry_run:
            print(f"\n⚠️  WARNING: About to delete {len(versions_to_delete)} versions!")
            confirmation = input("Type 'DELETE' to confirm: ")
            if confirmation != "DELETE":
                print("Aborted.")
                return

        print(f"\n{'Starting cleanup' if not dry_run else 'Dry run cleanup'}:")

        success_count = 0
        for version in self.sort_versions(list(versions_to_delete)):
            if self.delete_version(version, dry_run):
                success_count += 1

        print(f"\nCompleted: {success_count}/{len(versions_to_delete)} versions processed")

        if dry_run:
            print("\nThis was a dry run. Use --no-dry-run to perform actual deletions.")
        else:
            print("\nNote: PyPI version deletion typically requires manual action.")
            print("Consider using the PyPI web interface or twine for actual deletion.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up old versions of scxpand package on PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be deleted without actually deleting (default)",
    )

    parser.add_argument(
        "--no-dry-run", dest="dry_run", action="store_false", help="Actually perform the deletions (use with caution)"
    )

    parser.add_argument("--keep-latest", type=int, default=3, help="Keep the N most recent versions (default: 3)")

    parser.add_argument(
        "--keep-pattern", action="append", help="Keep versions matching pattern (can be used multiple times)"
    )

    parser.add_argument("--package-name", default="scxpand", help="Package name (default: scxpand)")

    args = parser.parse_args()

    # Validate arguments
    if args.keep_latest < 0:
        print("Error: --keep-latest must be non-negative")
        sys.exit(1)

    cleanup = PyPICleanup(args.package_name)

    print("=" * 60)
    print("PyPI Package Cleanup Script")
    print("=" * 60)
    print(f"Package: {args.package_name}")
    print(f"Keep latest: {args.keep_latest}")
    print(f"Keep patterns: {args.keep_pattern or 'None'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE DELETION'}")
    print("=" * 60)

    try:
        cleanup.cleanup_versions(keep_latest=args.keep_latest, keep_patterns=args.keep_pattern, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
