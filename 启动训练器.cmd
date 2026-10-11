@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
start "" "%~dp0WushuBridge-Trainer.exe"
