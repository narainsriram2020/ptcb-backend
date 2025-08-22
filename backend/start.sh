#!/bin/bash
# Start script for Render deployment

echo "Starting PlanetTerp Chatbot Backend..."
echo "Port: $PORT"
echo "Environment: $NODE_ENV"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port $PORT 