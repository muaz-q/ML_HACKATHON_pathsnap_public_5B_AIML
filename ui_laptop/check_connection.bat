@echo off
REM Checks whether this laptop can reach the PathSnap backend on Muaz's laptop.
set "IP="
set /p "IP=Muaz's laptop IP, for example 172.20.10.11 : "
if "%IP%"=="" goto noip
echo.
echo  Checking http://%IP%:8000/health ...
echo.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 6 -Uri 'http://%IP%:8000/health'; Write-Host $r.Content; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 goto fail

echo.
echo  [OK] Backend reachable. Now run start_ui.bat and enter  http://%IP%:8000
goto end

:fail
echo.
echo  [FAIL] Cannot reach the backend. Check, in this order:
echo    1. Is start_backend.bat running on Muaz's laptop, window still open?
echo    2. Are BOTH laptops on the SAME Wi-Fi or the same phone hotspot?
echo    3. Did Muaz run start_backend.bat with "Run as administrator" once? That opens the firewall.
echo    4. Is the IP exactly what start_backend.bat printed? It can change after reconnecting.
goto end

:noip
echo  No IP typed.

:end
echo.
pause
