@echo off
echo ============================================
echo  Shamsna - STEG PV Grid Ops Dashboard
echo ============================================
echo.
echo Stopping any previous instances...
taskkill /F /FI "WINDOWTITLE eq streamlit*" /T >nul 2>&1

echo Starting Streamlit Dashboard (standalone)...
echo The browser will open automatically at http://localhost:8501
echo.
echo Press CTRL+C to stop.
echo.

streamlit run dashboard\app.py --server.port 8501 --server.headless false
