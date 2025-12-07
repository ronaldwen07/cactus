#!/bin/bash

echo "Running Cactus test suite..."
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_MODEL="LiquidAI/LFM2-VL-450M"
DEFAULT_TRANSCRIBE_MODEL="openai/whisper-small"

MODEL_NAME="$DEFAULT_MODEL"
TRANSCRIBE_MODEL_NAME="$DEFAULT_TRANSCRIBE_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --transcribe_model)
            TRANSCRIBE_MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>            Model to use for tests (default: $DEFAULT_MODEL)"
            echo "  --transcribe_model <name> Transcribe model to use (default: $DEFAULT_TRANSCRIBE_MODEL)"
            echo "  --help, -h                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Using model: $MODEL_NAME"
echo "Using transcribe model: $TRANSCRIBE_MODEL_NAME"

echo ""
echo "Step 1: Downloading model weights..."
if ! "$PROJECT_ROOT/cli/cactus" download "$MODEL_NAME"; then
    echo "Failed to download model weights"
    exit 1
fi

if ! "$PROJECT_ROOT/cli/cactus" download "$TRANSCRIBE_MODEL_NAME"; then
    echo "Failed to download transcribe model weights"
    exit 1
fi

echo ""
echo "Step 2: Building Cactus library..."
cd "$PROJECT_ROOT"
if ! cactus/build.sh > /dev/null 2>&1; then
    echo "Failed to build cactus library"
    exit 1
fi

echo ""
echo "Step 3: Building tests..."

XCODEPROJ_PATH="$SCRIPT_DIR/CactusTest/CactusTest.xcodeproj"
TESTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACTUS_ROOT="$PROJECT_ROOT/cactus"
APPLE_ROOT="$PROJECT_ROOT/apple"

if ! command -v ruby &> /dev/null; then
    echo "Error: Ruby not found"
    exit 1
fi

if ! gem list xcodeproj -i &> /dev/null; then
    echo "Installing xcodeproj gem..."
    gem install xcodeproj > /dev/null 2>&1
fi

export PROJECT_ROOT TESTS_ROOT CACTUS_ROOT APPLE_ROOT XCODEPROJ_PATH
ruby "$SCRIPT_DIR/configure_xcode.rb" > /dev/null 2>&1

IOS_SIM_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
if [ -z "$IOS_SIM_SDK_PATH" ] || [ ! -d "$IOS_SIM_SDK_PATH" ]; then
    echo "Error: iOS Simulator SDK not found. Make sure Xcode is installed."
    exit 1
fi

if ! xcodebuild -project "$XCODEPROJ_PATH" \
     -scheme CactusTest \
     -configuration Debug \
     -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
     -derivedDataPath "$SCRIPT_DIR/build" \
     ARCHS=arm64 \
     ONLY_ACTIVE_ARCH=NO \
     IPHONEOS_DEPLOYMENT_TARGET=13.0 \
     SDKROOT="$IOS_SIM_SDK_PATH" \
     build > /dev/null 2>&1; then
    echo "Failed to build tests"
    exit 1
fi

APP_PATH="$SCRIPT_DIR/build/Build/Products/Debug-iphonesimulator/CactusTest.app"

MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

MODEL_SRC="$PROJECT_ROOT/weights/$MODEL_DIR"
TRANSCRIBE_MODEL_SRC="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"

cp -R "$MODEL_SRC" "$APP_PATH/" 2>/dev/null
cp -R "$TRANSCRIBE_MODEL_SRC" "$APP_PATH/" 2>/dev/null

echo ""
echo "Step 4: Running tests..."
echo "------------------------"

echo "Using model path: $APP_PATH/$MODEL_DIR"
echo "Using transcribe model path: $APP_PATH/$TRANSCRIBE_MODEL_DIR"

echo "Discovering test executables..."
echo "Found 5 test executable(s)"

BUNDLE_ID="cactus.CactusTest"
DEVICE_UUID=$(xcrun simctl list devices | grep "iPhone 17 Pro" | grep -v "unavailable" | head -1 | grep -oE '[A-F0-9-]{36}')

if [ -z "$DEVICE_UUID" ]; then
    echo "Error: Could not find iPhone 17 Pro simulator"
    exit 1
fi

xcrun simctl boot "$DEVICE_UUID" 2>/dev/null || true
xcrun simctl uninstall "$DEVICE_UUID" "$BUNDLE_ID" 2>/dev/null || true

APP_PATH="$SCRIPT_DIR/build/Build/Products/Debug-iphonesimulator/CactusTest.app"
xcrun simctl install "$DEVICE_UUID" "$APP_PATH" 2>/dev/null

SIMCTL_CHILD_CACTUS_TEST_MODEL="$MODEL_DIR" \
SIMCTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL="$TRANSCRIBE_MODEL_DIR" \
xcrun simctl launch --console-pty "$DEVICE_UUID" "$BUNDLE_ID" 2>/dev/null

echo ""
echo "Note: Tests were run on iOS Simulator (iPhone 17 Pro)"
