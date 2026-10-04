#!/bin/bash
# MOVE-OD Web Application - Complete Startup Guide

echo "=========================================="
echo "MOVE-OD Web Application Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    echo "❌ Error: Please run this script from the move_od root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install backend dependencies if needed
echo "📦 Checking backend dependencies..."
pip install -q fastapi uvicorn python-multipart pydantic 2>/dev/null

echo ""
echo "=========================================="
echo "Starting MOVE-OD Web Application"
echo "=========================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    deactivate 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend from the correct directory
echo "📡 Starting Backend (FastAPI) on http://localhost:8000..."
cd /home/rishav/move_od
python backend/app.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "❌ Backend failed to start. Check backend.log for errors"
    cat backend.log
    exit 1
fi

echo "✅ Backend started successfully (PID: $BACKEND_PID)"

# Start frontend
echo "🌐 Starting Frontend on http://localhost:8080..."
cd frontend
python3 -m http.server 8080 > /dev/null 2>&1 &
FRONTEND_PID=$!

sleep 2

echo "✅ Frontend started successfully (PID: $FRONTEND_PID)"
echo ""
echo "=========================================="
echo "✅ MOVE-OD Web Application is Running!"
echo "=========================================="
echo ""
echo "📍 Frontend:     http://localhost:8080"
echo "📍 Backend API:  http://localhost:8000"
echo "📍 API Docs:     http://localhost:8000/docs"
echo "📍 Test Page:    http://localhost:8080/test.html"
echo ""
echo "📋 Logs:"
echo "   Backend:  tail -f ../backend.log"
echo "   Frontend: Check terminal output"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for processes
wait $BACKEND_PID
wait $FRONTEND_PID
