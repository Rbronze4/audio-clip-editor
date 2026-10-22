@echo off
cd /d "%~dp0"
if not exist .venv-bgm\Scripts\python.exe (
  py -3.12 -m venv .venv-bgm
  if errorlevel 1 goto failed
)
set "BGM_TORCH_INDEX=https://download.pytorch.org/whl/cu121"
if /I "%~1"=="cpu" set "BGM_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
.venv-bgm\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url %BGM_TORCH_INDEX%
if errorlevel 1 goto failed
.venv-bgm\Scripts\python.exe -m pip install -r requirements-bgm.txt
if errorlevel 1 goto failed
.venv-bgm\Scripts\python.exe demucs_runner.py --prepare --models models --status models-setup.json
if errorlevel 1 goto failed
echo BGM removal is ready. Restart the app.
pause
exit /b 0
:failed
echo Setup failed. Check the error above and your internet connection.
pause
exit /b 1
