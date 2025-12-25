@echo off
REM Interactive CLI Menu - Windows Launcher
REM This script launches the interactive Node.js menu system

echo.
echo ========================================
echo  Interactive CLI Menu
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM Check if npm dependencies are installed
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Start the CLI menu
echo Starting CLI menu...
echo.
node cli-menu.js

REM Exit code propagation
if errorlevel 1 (
    echo.
    echo CLI menu exited with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

exit /b 0
