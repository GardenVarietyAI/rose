#!/bin/bash
# Start API server and worker

set -e

echo "Starting ROSE Server..."

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first."
    exit 1
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Installing dependencies..."
    poetry install --without dev --without test
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\nShutting down..."

    # Send SIGTERM to processes
    if kill -0 $SERVICE_PID 2>/dev/null; then
        echo "   Stopping API service..."
        kill -TERM $SERVICE_PID
    fi

    if kill -0 $INFERENCE_PID 2>/dev/null; then
        echo "   Stopping inference service..."
        kill -TERM $INFERENCE_PID
    fi

    if kill -0 $TRAINER_PID 2>/dev/null; then
        echo "   Stopping trainer..."
        kill -TERM $TRAINER_PID
    fi

    # Wait a bit for graceful shutdown
    sleep 2

    # Force kill if still running
    if kill -0 $SERVICE_PID 2>/dev/null; then
        kill -9 $SERVICE_PID 2>/dev/null || true
    fi

    if kill -0 $INFERENCE_PID 2>/dev/null; then
        kill -9 $INFERENCE_PID 2>/dev/null || true
    fi

    if kill -0 $TRAINER_PID 2>/dev/null; then
        kill -9 $TRAINER_PID 2>/dev/null || true
    fi

    echo "Shutdown complete"
    exit 0
}

trap cleanup EXIT SIGINT SIGTERM

# Start the service
echo "Starting API service on http://localhost:8004..."
poetry run rose-server &
SERVICE_PID=$!

# Wait a bit for service to start
sleep 2

# Start the inference service
echo "Starting inference service on ws://localhost:8005..."
poetry run rose-inference &
INFERENCE_PID=$!

# Wait a bit for inference to start
sleep 2

# Start the trainer
echo "Starting trainer..."
poetry run rose-trainer &
TRAINER_PID=$!

echo "ROSE Server is running!"
echo "   API: http://localhost:8004"
echo "   Docs: http://localhost:8004/docs"
echo "   Inference: ws://localhost:8005"
echo "   Trainer: PID $TRAINER_PID"
echo ""
echo "Press Ctrl+C to stop"

# Wait for all processes
wait $SERVICE_PID $INFERENCE_PID $TRAINER_PID
