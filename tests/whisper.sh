#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_TRANSCRIBE_MODEL="openai/whisper-small"
TRANSCRIBE_MODEL_NAME="$DEFAULT_TRANSCRIBE_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            TRANSCRIBE_MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>  Whisper model (default: $DEFAULT_TRANSCRIBE_MODEL)"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Using model: $TRANSCRIBE_MODEL_NAME"

echo "Downloading model..."
"$PROJECT_ROOT/cli/cactus" download "$TRANSCRIBE_MODEL_NAME" || exit 1

echo "Building library..."
cd "$PROJECT_ROOT" && cactus/build.sh || exit 1

echo "Building test..."
cd "$PROJECT_ROOT/tests"
mkdir -p build && cd build
cmake .. > /dev/null 2>&1 || exit 1
make test_streaming_mic -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) || exit 1

TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
export CACTUS_TEST_TRANSCRIBE_MODEL="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"

echo ""
echo "Press Enter to start recording (Ctrl+C to stop)..."
read

./test_streaming_mic
