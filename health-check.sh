#!/bin/bash
# health-check.sh - Monitor server and Ollama health

echo "Health Check"
echo "============"
echo ""

# Check server
echo "[1/2] Checking web server..."
if curl -s http://localhost:3000/health > /dev/null 2>&1; then
    response=$(curl -s http://localhost:3000/health)
    echo "✓ Server OK: $response"
else
    echo "✗ Server FAILED (http://localhost:3000/health)"
fi

echo ""

# Check Ollama
echo "[2/2] Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    models=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    if [ -z "$models" ]; then
        echo "⚠ Ollama OK but no models installed"
        echo "  Run: ollama pull gpt-oss-20"
    else
        echo "✓ Ollama OK. Models:"
        echo "$models" | sed 's/^/  - /'
    fi
else
    echo "✗ Ollama FAILED (http://localhost:11434)"
fi

echo ""
echo "Done."
