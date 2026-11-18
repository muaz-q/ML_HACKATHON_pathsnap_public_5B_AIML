@echo off
REM ============================================================
REM  PathSnap UI  (Laptop 2 - Meha)
REM  Double-click. Opens the UI at http://localhost:5500
REM ============================================================
cd /d "%~dp0ui"

set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY ( where py >nul 2>&1 && set "PY=py" )

if not defined PY (
  echo  Python not found - opening the page directly from the file instead.
  start "" "%~dp0ui\index.html"
  pause
  exit /b
)

echo.
echo  PathSnap UI is running at  http://localhost:5500
echo  In the page: enter  http://^<Muaz's IP^>:8000  and the API key, then press Connect.
echo  Keep this window open. Close it to stop the UI.
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5500/"
%PY% -m http.server 5500 --bind 127.0.0.1
pause
