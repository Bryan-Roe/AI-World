@echo off
REM Windows startup script for Web LLM Chat
REM Checks for Ollama, installs deps, and starts the server

echo.
echo ========================================
echo   Web LLM Chat - Windows Launcher
echo ========================================
echo.

REM Check for Node.js
echo [1/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install from https://nodejs.org
    pause
    exit /b 1
)
echo OK - Node.js installed

REM Check for Ollama (optional but recommended)
echo.
echo [2/4] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama not found. Install from https://ollama.com for local models
    echo You can still use cloud models with OPENAI_API_KEY
    echo.
    pause
) else (
    echo OK - Ollama found
    echo.
    echo [!] Run 'ollama serve' in a separate terminal to start Ollama
    echo [!] Then pull a model: ollama pull gpt-oss-20
    echo.
    pause
)

REM Install dependencies
echo.
echo [3/4] Installing npm dependencies...
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed
    pause
    exit /b 1
)
echo OK - Dependencies installed

REM Start server
echo.
echo [4/4] Starting server on http://localhost:3000
echo.
call npm run dev
pause
