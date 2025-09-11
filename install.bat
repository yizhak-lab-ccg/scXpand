@echo off
setlocal EnableDelayedExpansion

set REQUIRED_UV_VERSION=0.8.14

:: Read Python version from .python-version file
if exist .python-version (
    for /f "delims=" %%i in (.python-version) do set PYTHON_VERSION=%%i
    echo Using Python version from .python-version: !PYTHON_VERSION!
) else (
    set PYTHON_VERSION=3.13
    echo No .python-version file found, using default: !PYTHON_VERSION!
)

:: Function to compare versions
set COMPARE_RESULT=0
for /f "tokens=1,2,3 delims=." %%a in ("%REQUIRED_UV_VERSION%") do (
    set REQ_MAJOR=%%a
    set REQ_MINOR=%%b
    set REQ_PATCH=%%c
)

echo Setting up scXpand development environment...

:: Check if uv is installed
where uv >nul 2>&1
if errorlevel 1 (
    echo Error: uv is not installed.
    echo Please install uv !REQUIRED_UV_VERSION! or newer by following the official installation guide:
    echo https://docs.astral.sh/uv/getting-started/installation/
    echo.
    echo Quick install options:
    echo   Standalone installer: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo   Scoop:               scoop install main/uv
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('uv --version') do set UV_VERSION=%%i
for /f "tokens=1,2,3 delims=." %%a in ("!UV_VERSION!") do (
    set UV_MAJOR=%%a
    set UV_MINOR=%%b
    set UV_PATCH=%%c
)

:: Compare versions (major, minor, patch)
if !UV_MAJOR! LSS !REQ_MAJOR! goto :update_uv
if !UV_MAJOR! GTR !REQ_MAJOR! goto :uv_ok
if !UV_MINOR! LSS !REQ_MINOR! goto :update_uv
if !UV_MINOR! GTR !REQ_MINOR! goto :uv_ok
if !UV_PATCH! LSS !REQ_PATCH! goto :update_uv

goto :uv_ok

:update_uv
    echo Error: uv version !REQUIRED_UV_VERSION! or newer is required. Found version: !UV_VERSION!
    echo Please update uv by following the official installation guide:
    echo https://docs.astral.sh/uv/getting-started/installation/
    echo.
    echo Quick update options:
    echo   Standalone installer: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo   WinGet:              winget install --id=astral-sh.uv -e
    echo   Scoop:               scoop update uv
    pause
    exit /b 1

:uv_ok

echo Found uv: !UV_VERSION!

:: error if existing .venv directory is found
if exist .venv (
    echo Error: Existing .venv directory found. Please remove it before running this script.
    echo To remove it, run: rmdir .venv /s /q
    pause
    exit /b 1
)

:: Install Python using uv
echo Installing Python !PYTHON_VERSION!...
uv python install !PYTHON_VERSION!

:: Update the lock file to ensure it's current
echo Updating lock file...
uv lock

:: Create virtual environment and install dependencies from lock file
echo Creating virtual environment and installing dependencies from lock file...
uv sync --extra dev --extra docs

:: Check if virtual environment was created successfully
if not exist .venv\Scripts\activate.bat (
    echo Error: Virtual environment was not created properly.
    echo Trying to create it manually...
    uv venv .venv
    if not exist .venv\Scripts\activate.bat (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Re-running sync to install dependencies...
    uv sync --extra dev --extra docs
)

:: Install torch with auto backend selection and update lock file
uv run python install_torch.py

:: Register Jupyter kernel for this environment
echo Registering Jupyter kernel...
uv run python -m ipykernel install --user --name=scxpand_env --display-name="scXpand (.venv)" --env PYTHONPATH "%CD%"

:: Setup pre-commit hooks
echo Setting up pre-commit hooks...
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

echo Installation complete!
echo To activate the environment later, run in Powershell: .\.venv\Scripts\activate
pause
