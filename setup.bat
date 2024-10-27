@echo off
setlocal enabledelayedexpansion

echo Checking for existing virtual environment...
if exist "venv\" (
    echo Found existing virtual environment. Deleting...
    rmdir /s /q "venv"
    if errorlevel 1 (
        echo Failed to delete existing virtual environment.
        exit /b 1
    )
    echo Existing virtual environment deleted.
)

echo Searching for Python installations...

set "PYTHON310_PATH="

for /f "delims=" %%I in ('where python') do (
    set "PYTHON_CMD=%%I"
    for /f "tokens=2 delims= " %%A in ('"!PYTHON_CMD!" -V 2^>^&1') do (
        set "PYTHON_VERSION=%%A"
        echo Found Python !PYTHON_VERSION! at: !PYTHON_CMD!
        if "!PYTHON_VERSION:~0,4!"=="3.10" (
            set "PYTHON310_PATH=!PYTHON_CMD!"
            goto :FOUND_PYTHON
        )
    )
)

:FOUND_PYTHON
if "!PYTHON310_PATH!"=="" (
    echo Python 3.10.x not found in the PATH.
    echo Please ensure Python 3.10.x is installed and added to your PATH.
    echo.
    echo Available Python versions:
    where python
    exit /b 1
)

echo.
echo Python 3.10.x found at: !PYTHON310_PATH!
echo Setup will proceed with Python 3.10.x.

echo Creating virtual environment with Python 3.10.x...
"!PYTHON310_PATH!" -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

if exist requirements.txt (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements.
        exit /b 1
    )
)

echo Setup complete. Virtual environment is active.
echo Type 'deactivate' to exit the virtual environment.
cmd /
