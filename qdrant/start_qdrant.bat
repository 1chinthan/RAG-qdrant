@echo off
REM Starts the local Qdrant vector database. Keep this window open while you use it.
REM Dashboard: http://localhost:6333/dashboard
cd /d "%~dp0"
qdrant.exe --config-path config.yaml
