@echo off
echo.
echo ================================================
echo   CadenceWorks Analytics Engine
echo ================================================
echo.

echo Checking dependencies...
pip install -r requirements.txt -q

echo.
echo Starting app - opening in your browser...
echo.

streamlit run app.py
pause
