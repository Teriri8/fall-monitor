@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "PY_EXE="

for %%P in (
    "%~dp0.venv\Scripts\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) do (
    if not defined PY_EXE (
        if exist %%~P (
            %%~P -c "import cv2, ultralytics, fastapi" >nul 2>&1
            if not errorlevel 1 set "PY_EXE=%%~P"
        )
    )
)

if not defined PY_EXE (
    for /f "delims=" %%I in ('py -3 -c "import sys, cv2, ultralytics, fastapi; print(sys.executable)" 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%I"
    )
)

if not defined PY_EXE (
    python -c "import cv2, ultralytics, fastapi" >nul 2>&1
    if not errorlevel 1 set "PY_EXE=python"
)

if not defined PY_EXE (
    echo [ERROR] No Python environment with opencv-python, ultralytics, and fastapi found.
    echo         Please install dependencies: pip install -r requirements.txt
    pause
    exit /b 1
)

set "AUTO_OPEN=1"
set "ARGS=%*"
if defined ARGS (
    echo %ARGS% | findstr /i /c:"--list-cameras" /c:"--help" /c:"-h" /c:"--port" >nul
    if not errorlevel 1 set "AUTO_OPEN=0"
)

if "%AUTO_OPEN%"=="1" (
    start "" /b cmd /c "timeout /t 4 /nobreak >nul && start "" http://127.0.0.1:8000"
)

"%PY_EXE%" -m pc.run_all %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Process exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
