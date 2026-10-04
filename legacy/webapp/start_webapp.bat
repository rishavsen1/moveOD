@echo off
REM MOVE-OD Web Application Startup Script for Windows

echo.
echo Starting MOVE-OD Web Application...
echo.

REM Check if required directories exist
if not exist "backend" (
    echo Error: backend directory not found
    echo Please run this script from the move_od root directory
    pause
    exit /b 1
)

if not exist "frontend" (
    echo Error: frontend directory not found
    echo Please run this script from the move_od root directory
    pause
    exit /b 1
)

REM Start backend
echo Starting Backend (FastAPI) on http://localhost:8000...
cd backend
start "MOVE-OD Backend" python app.py
cd ..

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting Frontend on http://localhost:8080...
cd frontend
start "MOVE-OD Frontend" python -m http.server 8080
cd ..

REM Wait for frontend to start
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo MOVE-OD Web Application is running!
echo ========================================
echo.
echo Frontend: http://localhost:8080
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Close this window or press Ctrl+C to stop all servers
echo.

pause
