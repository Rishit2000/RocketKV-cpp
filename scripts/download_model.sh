#!/usr/bin/env bash
# ==============================================================================
# Helper script to download Llama-3.2-3B-Instruct GGUF weights from Hugging Face
# ==============================================================================

set -e

MODEL_NAME="Llama-3.2-3B-Instruct-Q8_0.gguf"
REPO_ID="bartowski/Llama-3.2-3B-Instruct-GGUF"

echo "Checking for existing model weights: $MODEL_NAME..."
if [ -f "$MODEL_NAME" ]; then
    echo "Model '$MODEL_NAME' already exists in the current directory."
    exit 0
fi

if command -v huggingface-cli &> /dev/null; then
    echo "Downloading $MODEL_NAME using huggingface-cli..."
    huggingface-cli download "$REPO_ID" "$MODEL_NAME" --local-dir . --local-dir-use-symlinks False
elif command -v curl &> /dev/null; then
    echo "Downloading $MODEL_NAME using curl..."
    curl -L -o "$MODEL_NAME" "https://huggingface.co/$REPO_ID/resolve/main/$MODEL_NAME"
else
    echo "Error: Neither 'huggingface-cli' nor 'curl' found. Please install one of them."
    exit 1
fi

echo "Model weights successfully downloaded to $MODEL_NAME"

