@echo off
setlocal
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" "%~dp0process_tif.py" %*
echo.
pause
