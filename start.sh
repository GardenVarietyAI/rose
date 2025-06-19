#!/bin/bash
# Start API server, training worker, and eval worker

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

    if kill -0 $TRAINING_WORKER_PID 2>/dev/null; then
        echo "   Stopping training worker..."
        kill -TERM $TRAINING_WORKER_PID
    fi

    if kill -0 $EVAL_WORKER_PID 2>/dev/null; then
        echo "   Stopping eval worker..."
        kill -TERM $EVAL_WORKER_PID
    fi

    # Wait a bit for graceful shutdown
    sleep 2

    # Force kill if still running
    if kill -0 $SERVICE_PID 2>/dev/null; then
        kill -9 $SERVICE_PID 2>/dev/null || true
    fi

    if kill -0 $TRAINING_WORKER_PID 2>/dev/null; then
        kill -9 $TRAINING_WORKER_PID 2>/dev/null || true
    fi

    if kill -0 $EVAL_WORKER_PID 2>/dev/null; then
        kill -9 $EVAL_WORKER_PID 2>/dev/null || true
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

# Start the training worker
echo "Starting training worker..."
poetry run rose-worker &
TRAINING_WORKER_PID=$!

# Start the eval worker
echo "Starting eval worker..."
poetry run rose-eval-worker &
EVAL_WORKER_PID=$!

echo "ROSE Server is running!"
echo "   API: http://localhost:8004"
echo "   Docs: http://localhost:8004/docs"
echo "   Training Worker: PID $TRAINING_WORKER_PID"
echo "   Eval Worker: PID $EVAL_WORKER_PID"
echo ""
echo "Press Ctrl+C to stop"

# Wait for all processes
wait $SERVICE_PID $TRAINING_WORKER_PID $EVAL_WORKER_PID
