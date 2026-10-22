@echo off
cd /d "%~dp0"
if not exist .venv-bgm\Scripts\python.exe (
  py -3.12 -m venv .venv-bgm
  if errorlevel 1 goto failed
)
set "SPEECH_TORCH_INDEX=https://download.pytorch.org/whl/cu121"
if /I "%~1"=="cpu" set "SPEECH_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
.venv-bgm\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url %SPEECH_TORCH_INDEX%
if errorlevel 1 goto failed
.venv-bgm\Scripts\python.exe -m pip install -r requirements-speech.txt
if errorlevel 1 goto failed
.venv-bgm\Scripts\python.exe speech_runner.py --prepare --models models --status speech-setup.json
if errorlevel 1 goto failed
echo Speech enhancement is ready. Restart the app and choose MossFormer2.
pause
exit /b 0
:failed
echo Setup failed. Check the error above and your internet connection.
pause
exit /b 1
