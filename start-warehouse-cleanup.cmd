@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "GAME_CONFIG=configs\game-route.json"
set "EXAMPLE_CONFIG=configs\game-route.example.json"

if not exist "%GAME_CONFIG%" (
    echo [Delta Vision] 尚未找到 %GAME_CONFIG%。
    echo [Delta Vision] 先复制 %EXAMPLE_CONFIG%，再填写仓库 Profile 路径和精确窗口标题。
    echo copy "%EXAMPLE_CONFIG%" "%GAME_CONFIG%"
    pause
    exit /b 2
)

echo [Delta Vision] D = 仓库 dry-run（识别主页、空保险箱和返回动作，不发送输入）
echo [Delta Vision] A = 仓库 armed（只允许进入仓库和返回主页两次受保护点击）
echo [Delta Vision] Q = 退出
choice /C DAQ /N /M "请选择 D/A/Q: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto armed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode WarehouseDryRun -Config "%GAME_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:armed
echo [Delta Vision] 只允许对已经验证为空的 0/2 保险箱执行本流程。
echo [Delta Vision] warehouse_cleanup.armed_ready 必须为 true；F12 为急停键。
choice /C YN /N /M "确认发送两次仓库点击？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0
echo [Delta Vision] 启动后请在 5 秒内切回三角洲行动窗口。
timeout /T 5 /NOBREAK >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode WarehouseArmed -Config "%GAME_CONFIG%" -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"

:finish
echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] 仓库流程停止或启动失败，退出码: %EXIT_CODE%
) else (
    echo [Delta Vision] 仓库流程完成；证据已保存到 artifacts\runs。
)
pause
exit /b %EXIT_CODE%
