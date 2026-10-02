@echo off
echo ==============================================
echo ☀ Starting Shamsna PV Forecast Services
echo ==============================================

echo [1/2] Starting FastAPI Backend on port 8000...
start cmd /k "python scripts/07_api_server.py"

echo Waiting 3 seconds for API to initialize...
timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit Dashboard...
start cmd /k "streamlit run dashboard/app.py"

echo Done! The dashboard should open in your browser shortly.
