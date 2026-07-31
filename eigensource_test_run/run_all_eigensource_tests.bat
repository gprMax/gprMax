@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "RUNNER=%SCRIPT_DIR%run_all_eigensource_tests.py"

if defined PYTHON (
    "%PYTHON%" "%RUNNER%" %*
) else (
    python "%RUNNER%" %*
)

exit /b %ERRORLEVEL%

