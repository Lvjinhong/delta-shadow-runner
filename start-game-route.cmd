@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "GAME_CONFIG=configs\game-route.json"
set "EXAMPLE_CONFIG=configs\game-route.example.json"

if not exist "%GAME_CONFIG%" (
    echo [Delta Vision] 尚未找到 %GAME_CONFIG%。
    echo [Delta Vision] 先复制 %EXAMPLE_CONFIG%，再按 README 填写路线节点和键鼠参数。
    echo copy "%EXAMPLE_CONFIG%" "%GAME_CONFIG%"
    pause
    exit /b 2
)

echo [Delta Vision] D = 局内 dry-run（从已进图画面开始，只记录不发送输入）
echo [Delta Vision] L = 完整外循环 dry-run（大厅、路线、回厅、仓库只识别记录）
echo [Delta Vision] A = 完整外循环 armed（大厅进图、跑图、回厅、仓库清理）
echo [Delta Vision] Q = 退出
choice /C DLAQ /N /M "请选择 D/L/A/Q: "
if errorlevel 4 exit /b 0
if errorlevel 3 goto armed
if errorlevel 2 goto loop_dry

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode DryRun -Config "%GAME_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:loop_dry
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode LoopDryRun -Config "%GAME_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:armed
echo [Delta Vision] 完整外循环 armed 前必须完成菜单、路线、仓库 blind 评估。
echo [Delta Vision] 根 armed_ready 和 warehouse_cleanup.armed_ready 必须都为 true。
echo [Delta Vision] F12 为急停；切走目标窗口也会停止并释放按键。
choice /C YN /N /M "确认进入 armed 模式？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0
echo [Delta Vision] 启动后请在 5 秒内切回三角洲行动窗口。
timeout /T 5 /NOBREAK >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode LoopArmed -Config "%GAME_CONFIG%" -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"

:finish
echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] Worker 未到达目标或启动失败，退出码: %EXIT_CODE%
) else (
    echo [Delta Vision] 会话运行完成；证据已保存到 artifacts\runs。
)
pause
exit /b %EXIT_CODE%
