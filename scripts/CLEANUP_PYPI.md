# PyPI Package Cleanup Guide

This guide explains how to use the PyPI cleanup scripts to remove older versions of the scxpand package from PyPI.

## Overview

The cleanup scripts help manage package versions on PyPI by:
- Listing all published versions
- Identifying which versions to keep vs. delete based on configurable rules
- Providing safe dry-run mode to preview changes
- Supporting pattern-based version retention

## Files

- `cleanup_pypi.py` - Main Python script with full functionality
- `cleanup_pypi.sh` - Bash wrapper with convenient shortcuts
- `CLEANUP_PYPI.md` - This documentation

## Prerequisites

1. **Python Requirements**:
   ```bash
   pip install requests
   ```

2. **PyPI Token** (for actual deletions):
   - Go to https://pypi.org/manage/account/token/
   - Create a token with "Delete packages" permission
   - Set up authentication (see below)

## Authentication Setup

### Option 1: Environment Variable
```bash
export PYPI_TOKEN=pypi-your-token-here
```

### Option 2: Token File
Edit `scripts/pypi_token.txt` and add your token:
```
pypi-your-token-here
```

## Usage

### Quick Commands (Shell Wrapper)

```bash
# List all versions without making changes
./scripts/cleanup_pypi.sh list

# Dry run: see what would be deleted (keeps 3 most recent)
./scripts/cleanup_pypi.sh clean

# Dry run: keep 5 most recent versions
./scripts/cleanup_pypi.sh clean --keep-latest 5

# Actually delete old versions (use with caution!)
./scripts/cleanup_pypi.sh clean-for-real --keep-latest 3
```

### Python Script Direct Usage

```bash
# Basic dry run (default: keep 3 most recent)
python scripts/cleanup_pypi.py

# Keep 5 most recent versions
python scripts/cleanup_pypi.py --keep-latest 5

# Keep versions matching patterns
python scripts/cleanup_pypi.py --keep-pattern "0.1.15*" --keep-pattern "0.1.14*"

# Actually perform deletions (not just dry run)
python scripts/cleanup_pypi.py --no-dry-run --keep-latest 3

# Different package name
python scripts/cleanup_pypi.py --package-name mypackage
```

## Safety Features

1. **Dry Run by Default**: All operations are dry runs unless explicitly disabled
2. **Confirmation Prompts**: Real deletions require typing "DELETE" to confirm
3. **Version Sorting**: Intelligent version parsing handles dev versions correctly
4. **Pattern Matching**: Flexible rules for keeping specific versions

## Example Output

```
PyPI Package Cleanup Script
============================================================
Package: scxpand
Keep latest: 3
Keep patterns: None
Mode: DRY RUN
============================================================
Fetching version information for package: scxpand

Found 15 versions

All versions (oldest to newest):
   1. 0.1.0.dev1
   2. 0.1.0
   3. 0.1.2
   4. 0.1.6
   5. 0.1.7
   6. 0.1.9
   7. 0.1.10
   8. 0.1.11
   9. 0.1.12
  10. 0.1.13
  11. 0.1.14
  12. 0.1.15

Versions to keep (3):
  ✓ 0.1.13
  ✓ 0.1.14
  ✓ 0.1.15

Versions to delete (12):
  ✗ 0.1.0.dev1
  ✗ 0.1.0
  ✗ 0.1.2
  ✗ 0.1.6
  ✗ 0.1.7
  ✗ 0.1.9
  ✗ 0.1.10
  ✗ 0.1.11
  ✗ 0.1.12

Dry run cleanup:
  [DRY RUN] Would delete version 0.1.0.dev1
  [DRY RUN] Would delete version 0.1.0
  ...

Completed: 12/12 versions processed

This was a dry run. Use --no-dry-run to perform actual deletions.
```

## Version Retention Strategies

### Keep Latest N Versions
```bash
# Keep only the 2 most recent versions
./scripts/cleanup_pypi.sh clean --keep-latest 2
```

### Pattern-Based Retention
```bash
# Keep all 0.1.15.x versions and 3 most recent overall
python scripts/cleanup_pypi.py --keep-pattern "0.1.15*" --keep-latest 3
```

### Mixed Strategies
```bash
# Keep 5 latest + specific important versions
python scripts/cleanup_pypi.py \
    --keep-latest 5 \
    --keep-pattern "0.1.0" \
    --keep-pattern "1.0.*"
```

## Important Notes

1. **PyPI Limitations**: PyPI doesn't provide a direct API for package deletion. The script shows what would be deleted, but actual deletion may require:
   - Using the PyPI web interface
   - Using `twine` command-line tool
   - Manual API calls

2. **Irreversible Action**: Once versions are deleted from PyPI, they cannot be re-uploaded with the same version number.

3. **Testing**: Always run with `--dry-run` first to verify the intended changes.

4. **Permissions**: Your PyPI token must have "Delete packages" permission for the target package.

## Troubleshooting

### Common Issues

1. **"No PyPI token available"**
   - Set up authentication as described above
   - Verify token has correct permissions

2. **"requests library is required"**
   ```bash
   pip install requests
   ```

3. **"No versions found"**
   - Check package name spelling
   - Verify package exists on PyPI
   - Check network connectivity

4. **Permission errors**
   - Ensure token has "Delete packages" permission
   - Verify you're the package owner/maintainer

### Getting Help

Run with `--help` for detailed usage information:
```bash
python scripts/cleanup_pypi.py --help
./scripts/cleanup_pypi.sh help
```

## Integration with Release Process

Consider integrating cleanup into your release workflow:

```bash
# After successful release, clean up old versions
./scripts/release.sh --minor
./scripts/cleanup_pypi.sh clean-for-real --keep-latest 5
```

This ensures your PyPI page stays clean and users aren't confused by too many old versions.
