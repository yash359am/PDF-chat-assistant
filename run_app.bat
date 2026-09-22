@echo off
title PDF Chat Assistant Launcher
echo ===================================================
echo           Starting PDF Chat Assistant UI
echo ===================================================
echo.
echo Starting server on http://localhost:8501 ...
echo (Keep this window open while using the app. Press Ctrl+C to stop)
echo.
timeout /t 2 /nobreak >nul
start http://localhost:8501
streamlit run app.py --server.port 8501 --server.headless true
pause
