# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.1.0.html).

## [Unreleased]

### Changed

- `save_path` parameter in inference function - now `None` means no saving results to file
- Less redundant log messages
- Code formatting improvements

## [0.3.5] - 2025-09-16

### Added

- test release

### Added

- Interactive changelog validation with template creation
- Pre-release changelog checking to ensure proper documentation

### Changed

- Release script now validates changelog entries before running tests and builds
- Improved user experience by separating changelog editing from release process
- Enhanced PUBLISHING.md documentation with step-by-step workflow

### Fixed

- Changelog workflow now allows manual editing with preferred editors

## [0.3.4] - 2025-09-16

### Changed

- Removed `device` parameter from `run_inference()` function - device is now automatically detected for optimal performance
- Reorganize the documentation

## [0.3.2] - 2025-09-15

### Changed

- Renamed "main.predict" to "main.inference"

## [0.3.0] - 2025-09-14

### Added

- Initial release of scXpand
