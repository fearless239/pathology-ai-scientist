@echo off
setlocal
title Path Scientist
start "" cmd /c "timeout /t 7 /nobreak >nul & start http://127.0.0.1:8501"
wsl.exe --cd "%~dp0" bash ./scripts/pathmnist.sh web --server.headless true
if errorlevel 1 (
  echo.
  echo Path Scientist failed to start. Keep this window open and check the message above.
  pause
)
