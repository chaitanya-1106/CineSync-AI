@echo off
title CineGenre AI Launcher
echo =========================================
echo       Starting CineGenre AI Services
echo =========================================
echo.

echo Starting Backend Server and Frontend (FastAPI)...
start "CineGenre API" cmd /k "cd backend && python app.py"

echo.
echo All services launched! 
echo Open your browser to http://localhost:8000
echo.
echo Note: To stop the application, just close the command prompt window that popped up.
pause
