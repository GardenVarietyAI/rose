#!/bin/bash
# Start API server with Rust inference server

set -e

echo "Starting ROSE Server with Rust inference..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Please install uv first."
    exit 1
fi

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "Cargo not found. Please install Rust first."
    exit 1
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Installing Python dependencies..."
    uv sync --no-dev
fi

# Kill any existing processes on our ports
echo "Checking for existing processes..."
lsof -ti:8004 | xargs kill -9 2>/dev/null || true
lsof -ti:8005 | xargs kill -9 2>/dev/null || true
pkill -f rose-server 2>/dev/null || true
pkill -f rose-inference 2>/dev/null || true
pkill -f rose-trainer 2>/dev/null || true
sleep 1

# Build Rust inference server
echo "Building Rust inference server..."
cd src/rose_inference_rs
cargo build --release
cd ../..

# Function to cleanup on exit
cleanup() {
    echo -e "\nShutting down..."

    # Send SIGTERM to processes
    if kill -0 "$SERVICE_PID" 2>/dev/null; then
        echo "   Stopping API service..."
        kill -TERM "$SERVICE_PID"
    fi

    if kill -0 "$INFERENCE_PID" 2>/dev/null; then
        echo "   Stopping Rust inference service..."
        kill -TERM "$INFERENCE_PID"
    fi

    if kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "   Stopping trainer..."
        kill -TERM "$TRAINER_PID"
    fi

    # Wait a bit for graceful shutdown
    sleep 2

    # Force kill if still running
    if kill -0 "$SERVICE_PID" 2>/dev/null; then
        kill -9 "$SERVICE_PID" 2>/dev/null || true
    fi

    if kill -0 "$INFERENCE_PID" 2>/dev/null; then
        kill -9 "$INFERENCE_PID" 2>/dev/null || true
    fi

    if kill -0 "$TRAINER_PID" 2>/dev/null; then
        kill -9 "$TRAINER_PID" 2>/dev/null || true
    fi

    echo "Shutdown complete"
    exit 0
}

trap cleanup EXIT SIGINT SIGTERM

echo ""
# Generate auth token if not set
if [ -z "$ROSE_API_KEY" ]; then
    export ROSE_API_KEY=$(uv run rose auth generate-token)
    echo "Generated API key: $ROSE_API_KEY"
    echo "To use this key in the future, export ROSE_API_KEY=$ROSE_API_KEY"
else
    echo "Using existing API key"
fi
echo ""

# Start the service
echo "Starting API service on http://localhost:8004..."
uv run rose-server &
SERVICE_PID=$!

# Wait a bit for service to start
sleep 2

# Start the Rust inference service
echo "Starting Rust inference service on ws://localhost:8005..."
RUST_LOG=info cargo run --release --manifest-path src/rose_inference_rs/Cargo.toml &
INFERENCE_PID=$!

# Wait a bit for inference to start
sleep 2

# Start the trainer
echo "Starting trainer..."
uv run rose-trainer &
TRAINER_PID=$!

echo "ROSE Server with Rust inference is running!"
echo "   API: http://localhost:8004"
echo "   Docs: http://localhost:8004/docs"
echo "   Inference: ws://localhost:8005 (Rust)"
echo "   Trainer: PID $TRAINER_PID"
echo ""
echo "Press Ctrl+C to stop"

# Wait for all processes
wait "$SERVICE_PID" "$INFERENCE_PID" "$TRAINER_PID"
