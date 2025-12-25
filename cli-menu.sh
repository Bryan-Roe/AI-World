#!/bin/bash

# Interactive CLI Menu - Unix/Linux/macOS Launcher
# This script launches the interactive Node.js menu system

echo ""
echo "========================================"
echo "  Interactive CLI Menu"
echo "========================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "Error: Node.js v18+ required (found v${NODE_VERSION})"
    exit 1
fi

# Check if npm dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Start the CLI menu
echo "Starting CLI menu..."
echo ""

node cli-menu.js

# Exit code propagation
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "CLI menu exited with error code $EXIT_CODE"
    exit $EXIT_CODE
fi

exit 0
