@echo off
REM Checks whether this laptop can reach the PathSnap backend on Muaz's laptop.
set /p "IP=Muaz's laptop IP (e.g. 192.168.1.23): "
echo.
echo  Pinging http://%IP%:8000/health ...
curl -s -m 5 http://%IP%:8000/health
if errorlevel 1 (
  echo.
  echo  [FAIL] Cannot reach the backend. Check, in this order:
  echo    1. Is start_backend.bat running on Muaz's laptop?
  echo    2. Are BOTH laptops on the SAME Wi-Fi / hotspot?
  echo    3. Did Muaz run start_backend.bat as administrator once (firewall)?
  echo    4. College Wi-Fi often blocks laptop-to-laptop traffic: use a phone hotspot for both.
) else (
  echo.
  echo.
  echo  [OK] Backend reachable. Now run start_ui.bat.
)
echo.
pause
