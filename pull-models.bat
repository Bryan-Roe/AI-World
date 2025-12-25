# @echo off
# Helper batch script to pull Ollama models (Windows)

@echo off
setlocal enabledelayedexpansion

echo Ollama Model Puller
echo ===================
echo.

REM Check if Ollama is running
echo Checking if Ollama is running...
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo ERROR: Ollama is not running on http://localhost:11434
    echo.
    echo Start Ollama first, then run this script
    pause
    exit /b 1
)

echo OK - Ollama is running
echo.

REM Pull models
setlocal enabledelayedexpansion
set models=gpt-oss-20 llama3.2 qwen2.5

for %%m in (%models%) do (
    echo Pulling %%m...
    call ollama pull %%m
    if errorlevel 1 (
        echo X Failed to pull %%m
    ) else (
        echo + %%m pulled successfully
    )
    echo.
)

echo Done! Available models:
ollama list
pause
