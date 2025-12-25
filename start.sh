#!/bin/bash
# Unix/Linux/macOS startup script for Web LLM Chat

echo ""
echo "========================================"
echo "  Web LLM Chat - Unix Launcher"
echo "========================================"
echo ""

# Check for Node.js
echo "[1/4] Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Please install from https://nodejs.org"
    exit 1
fi
echo "OK - Node.js $(node --version) installed"

# Check for Ollama
echo ""
echo "[2/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama not found. Install from https://ollama.com for local models"
    echo "You can still use cloud models with OPENAI_API_KEY"
else
    echo "OK - Ollama found"
    echo ""
    echo "[!] Run 'ollama serve' in a separate terminal to start Ollama"
    echo "[!] Then pull a model: ollama pull gpt-oss-20"
fi

# Install dependencies
echo ""
echo "[3/4] Installing npm dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: npm install failed"
    exit 1
fi
echo "OK - Dependencies installed"

# Start server
echo ""
echo "[4/4] Starting server on http://localhost:3000"
echo ""
npm run dev
