@echo off
REM setup-validator.bat - Verify environment and dependencies (Windows)

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo   Web LLM Chat - Setup Validator
echo ==========================================
echo.

set CHECKS_PASSED=0
set CHECKS_FAILED=0
set WARNINGS=0

echo 1. Checking System Requirements...
echo.

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo X Node.js not found. Install from https://nodejs.org ^(18+^)
    set /a CHECKS_FAILED+=1
) else (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo + Node.js installed: !NODE_VERSION!
    set /a CHECKS_PASSED+=1
)

REM Check npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo X npm not found. Install Node.js ^(includes npm^)
    set /a CHECKS_FAILED+=1
) else (
    for /f "tokens=*" %%i in ('npm --version') do set NPM_VERSION=%%i
    echo + npm installed: !NPM_VERSION!
    set /a CHECKS_PASSED+=1
)

echo.
echo 2. Checking Project Files...
echo.

if exist "server.js" (
    echo + server.js found
    set /a CHECKS_PASSED+=1
) else (
    echo X server.js not found
    set /a CHECKS_FAILED+=1
)

if exist "package.json" (
    echo + package.json found
    set /a CHECKS_PASSED+=1
) else (
    echo X package.json not found
    set /a CHECKS_FAILED+=1
)

if exist "public\" (
    echo + public/ directory found
    set /a CHECKS_PASSED+=1
) else (
    echo X public/ directory not found
    set /a CHECKS_FAILED+=1
)

if exist ".env" (
    echo + .env file exists
    set /a CHECKS_PASSED+=1
) else (
    if exist ".env.example" (
        echo W .env not found - copy from .env.example and configure
        set /a WARNINGS+=1
    ) else (
        echo W .env.example not found
        set /a WARNINGS+=1
    )
)

echo.
echo 3. Checking Dependencies...
echo.

if exist "node_modules\" (
    echo + node_modules/ exists ^(dependencies installed^)
    set /a CHECKS_PASSED+=1
) else (
    echo W node_modules/ not found - run: npm install
    set /a WARNINGS+=1
)

echo.
echo 4. Checking Ollama ^(Local Models^)...
echo.

where ollama >nul 2>&1
if errorlevel 1 (
    echo W Ollama not installed - for local models, install from https://ollama.com
    set /a WARNINGS+=1
) else (
    echo + Ollama installed
    set /a CHECKS_PASSED+=1
    
    REM Check if running
    powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    if errorlevel 1 (
        echo W Ollama is not running - start with: ollama serve
        set /a WARNINGS+=1
    ) else (
        echo + Ollama is running on http://localhost:11434
        set /a CHECKS_PASSED+=1
    )
)

echo.
echo 5. Checking OpenAI ^(Cloud Models^)...
echo.

if exist ".env" (
    findstr /R "OPENAI_API_KEY=sk-" .env >nul 2>&1
    if errorlevel 1 (
        echo W OPENAI_API_KEY not set - for cloud models, set in .env
        set /a WARNINGS+=1
    ) else (
        echo + OPENAI_API_KEY is set in .env
        set /a CHECKS_PASSED+=1
    )
) else (
    echo W .env file not found - create from .env.example
    set /a WARNINGS+=1
)

echo.
echo 6. Checking Port Availability...
echo.

netstat -ano | findstr ":3000 " >nul 2>&1
if errorlevel 1 (
    echo + Port 3000 is available
    set /a CHECKS_PASSED+=1
) else (
    echo X Port 3000 is already in use
    set /a CHECKS_FAILED+=1
)

echo.
echo ==========================================
echo   Summary
echo ==========================================
echo.
echo Passed:   !CHECKS_PASSED!
if !WARNINGS! gtr 0 (
    echo Warnings: !WARNINGS!
)
if !CHECKS_FAILED! gtr 0 (
    echo Failed:   !CHECKS_FAILED!
)
echo.

if !CHECKS_FAILED! equ 0 (
    echo + Setup validation passed!
    echo.
    echo You can now start the server:
    echo   npm run dev
    echo.
    echo Then open http://localhost:3000 in your browser
    pause
    exit /b 0
) else (
    echo X Setup validation failed. Fix errors above and try again.
    pause
    exit /b 1
)
