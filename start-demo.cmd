@echo off
setlocal
cd /d "%~dp0"

set "DEMO_URL=http://localhost:5173"
set "HEALTH_URL=http://127.0.0.1:4173/api/health"
set "WINDOW_TITLE=Shadow Runner Lab"

call :probe_demo
if not errorlevel 1 (
    echo [Shadow Runner Lab] Demo is already running.
    start "" "%DEMO_URL%" >nul 2>&1
    exit /b 0
)

if not exist "%~dp0init.ps1" (
    echo [Shadow Runner Lab] Missing init.ps1 in %~dp0
    pause
    exit /b 1
)

where node.exe >nul 2>&1
if errorlevel 1 (
    echo [Shadow Runner Lab] Node.js was not found. Install Node.js 24 or newer.
    pause
    exit /b 1
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [Shadow Runner Lab] npm was not found. Check the Node.js installation and PATH.
    pause
    exit /b 1
)

echo [Shadow Runner Lab] Starting the demo. The first run may install dependencies.
start "%WINDOW_TITLE%" powershell.exe -NoLogo -NoExit -NoProfile -ExecutionPolicy Bypass -File "%~dp0init.ps1" -Mode dev

echo [Shadow Runner Lab] Waiting for the web and API services...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'SilentlyContinue'; $deadline = (Get-Date).AddSeconds(180); while ((Get-Date) -lt $deadline) { try { $web = Invoke-WebRequest -UseBasicParsing -Uri '%DEMO_URL%' -TimeoutSec 2; $health = Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($web.StatusCode -eq 200 -and $web.Content -match '<title>Shadow Runner Lab' -and $health.success -eq $true -and $health.data.status -eq 'ok' -and $health.data.mode -eq 'simulation' -and $health.data.compute -eq 'cpu-only') { exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"

if errorlevel 1 (
    echo [Shadow Runner Lab] Startup timed out after 180 seconds.
    echo [Shadow Runner Lab] Check the "%WINDOW_TITLE%" window for the error details.
    pause
    exit /b 1
)

echo [Shadow Runner Lab] Ready: %DEMO_URL%
start "" "%DEMO_URL%" >nul 2>&1
exit /b 0

:probe_demo
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'SilentlyContinue'; try { $web = Invoke-WebRequest -UseBasicParsing -Uri '%DEMO_URL%' -TimeoutSec 2; $health = Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($web.StatusCode -eq 200 -and $web.Content -match '<title>Shadow Runner Lab' -and $health.success -eq $true -and $health.data.status -eq 'ok' -and $health.data.mode -eq 'simulation' -and $health.data.compute -eq 'cpu-only') { exit 0 } } catch {}; exit 1"
exit /b %errorlevel%
