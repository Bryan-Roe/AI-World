#!/bin/bash
# Helper script to pull Ollama models

MODELS=("gpt-oss-20" "llama3.2" "qwen2.5")

echo "Ollama Model Puller"
echo "=================="
echo ""

# Check if Ollama is running
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "ERROR: Ollama is not running on http://localhost:11434"
    echo ""
    echo "Start Ollama with: ollama serve"
    exit 1
fi

echo "OK - Ollama is running"
echo ""

# Pull models
for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    ollama pull "$model"
    if [ $? -eq 0 ]; then
        echo "✓ $model pulled successfully"
    else
        echo "✗ Failed to pull $model"
    fi
    echo ""
done

echo "Done! Available models:"
ollama list
