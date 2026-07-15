@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "GAME_CONFIG=configs\game-route.json"
set "PROFILE=profiles\route-01\templates.json"

if not exist "%GAME_CONFIG%" (
    echo [Delta Vision] 找不到 %GAME_CONFIG%。请先复制并填写 game-route.example.json。
    pause
    exit /b 2
)

if not exist "%PROFILE%" (
    echo [Delta Vision] 找不到 %PROFILE%。请先完成 calibration 和模板生成。
    pause
    exit /b 2
)

echo [Delta Vision] Preflight 会依次执行：配置校验、受控 SendInput E2E、游戏窗口 60 秒截图基准。
echo [Delta Vision] 请先打开三角洲行动并保持固定分辨率；受控窗口结束后有 5 秒切回游戏。
echo [Delta Vision] 受控 E2E 会发送 WASD 到独立测试窗口，F12 为急停，不会读取游戏进程。
choice /C YN /N /M "确认开始 Windows Preflight？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode Preflight -Config "%GAME_CONFIG%" -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] Preflight 未通过，退出码: %EXIT_CODE%。请检查 artifacts\runs 下的证据和 preflight-report.json。
) else (
    echo [Delta Vision] Preflight 已通过，完整证据保存在 artifacts\runs。
)
pause
exit /b %EXIT_CODE%
