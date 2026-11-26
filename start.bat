@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not exist "%PY_EXE%" (
    echo [ERROR] Python 3.11 not found at: %PY_EXE%
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
