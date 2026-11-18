@echo off
REM ============================================================
REM  PathSnap BACKEND  (Laptop 1 - Muaz)
REM  Double-click to start. Optional argument: trip folder.
REM    start_backend.bat                       -> data\evaluator_test
REM    start_backend.bat data\student_test
REM ============================================================
cd /d "%~dp0"

set "PATHSNAP_EVAL_DIR=%~dp0data\evaluator_test"
if not "%~1"=="" set "PATHSNAP_EVAL_DIR=%~f1"

echo.
echo  PathSnap backend
echo  ----------------
echo  Trip folder: %PATHSNAP_EVAL_DIR%
echo.
set "PATHSNAP_API_KEY="
set /p "PATHSNAP_API_KEY=Type an API key for this session (or just press Enter for none): "
echo.

REM Allow other laptops to reach port 8000 (needs admin once; harmless if already added)
netsh advfirewall firewall show rule name="PathSnap API 8000" >nul 2>&1
if errorlevel 1 (
  netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000 profile=any >nul 2>&1
  if errorlevel 1 (
    echo  [!] Could not open port 8000 in Windows Firewall.
    echo      Close this window, right-click start_backend.bat, choose "Run as administrator" ONCE.
    echo.
  ) else (
    echo  Firewall: port 8000 opened for other laptops.
  )
) else (
  echo  Firewall: port 8000 already open.
)

echo.
echo  Tell the UI laptop to enter ONE of these addresses (the Wi-Fi one):
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.InterfaceAlias -notmatch 'VMware|VirtualBox|vEthernet|Bluetooth|Loopback|WSL' } | ForEach-Object { '     http://' + $_.IPAddress + ':8000      (' + $_.InterfaceAlias + ')' }"
if defined PATHSNAP_API_KEY (echo  API key: %PATHSNAP_API_KEY%) else (echo  API key: none)
echo.
echo  Keep this window open. Every request from the UI appears below as REQUEST / RESULT lines.
echo  Press Ctrl+C to stop.
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
