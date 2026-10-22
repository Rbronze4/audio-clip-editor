@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "from core import find_ffmpeg; import subprocess; subprocess.run([find_ffmpeg(), '-version'], check=True, stdout=subprocess.DEVNULL)"
if errorlevel 1 goto failed
echo Setup complete. Double-click run.bat to start.
pause
exit /b 0
:failed
echo Setup failed. Check Python 3.12 and your internet connection, then retry.
pause
exit /b 1
