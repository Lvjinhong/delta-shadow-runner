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
echo [Delta Vision] A = 全会话 armed（从大厅识别零号大坝，确认进图后再跑路线）
echo [Delta Vision] Q = 退出
choice /C DAQ /N /M "请选择 D/A/Q: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto armed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode DryRun -Config "%GAME_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:armed
echo [Delta Vision] 全会话 armed 前必须完成菜单和路线 blind 评估，并把配置 armed_ready 改为 true。
echo [Delta Vision] F12 为急停；切走目标窗口也会停止并释放按键。
choice /C YN /N /M "确认进入 armed 模式？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0
echo [Delta Vision] 启动后请在 5 秒内切回三角洲行动窗口。
timeout /T 5 /NOBREAK >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode SessionArmed -Config "%GAME_CONFIG%" -ConfirmArmed
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
