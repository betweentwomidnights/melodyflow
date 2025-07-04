#!/bin/bash
set -e

# Function to handle cleanup
cleanup() {
    echo "Received shutdown signal - cleaning up..."
    if [ -n "$flask_pid" ]; then
        echo "Sending SIGTERM to Flask process ($flask_pid)"
        kill -TERM "$flask_pid" 2>/dev/null || true
    fi
    if [ -n "$worker_pid" ]; then
        echo "Sending SIGTERM to RQ worker ($worker_pid)"
        kill -TERM "$worker_pid" 2>/dev/null || true
    fi
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT SIGQUIT

echo "Starting MelodyFlow RQ worker..."
python3 melodyflow_worker.py &
worker_pid=$!

echo "Starting MelodyFlow service..."
python3 spaces_melodyflow.py &
flask_pid=$!

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?