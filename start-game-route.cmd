@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "GAME_CONFIG=configs\game-route.json"
set "EXAMPLE_CONFIG=configs\game-route.example.json"
set "PROFILE=profiles\route-01\templates.json"

if not exist "%GAME_CONFIG%" (
    echo [Delta Vision] 尚未找到 %GAME_CONFIG%。
    echo [Delta Vision] 先复制 %EXAMPLE_CONFIG%，再按 README 填写路线节点和键鼠参数。
    echo copy "%EXAMPLE_CONFIG%" "%GAME_CONFIG%"
    pause
    exit /b 2
)

if not exist "%PROFILE%" (
    echo [Delta Vision] 尚未找到 %PROFILE%。
    echo [Delta Vision] 先按 README 完成 calibration 采样、标签和模板标定。
    pause
    exit /b 2
)

echo [Delta Vision] D = dry-run（只截图和记录，不发送输入）
echo [Delta Vision] A = armed（向前台三角洲行动窗口发送标准键鼠输入）
echo [Delta Vision] Q = 退出
choice /C DAQ /N /M "请选择 D/A/Q: "
if errorlevel 3 exit /b 0
if errorlevel 2 goto armed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode DryRun -Config "%GAME_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:armed
echo [Delta Vision] armed 前必须完成独立 blind 评估，并把配置 armed_ready 改为 true。
echo [Delta Vision] F12 为急停；切走目标窗口也会停止并释放按键。
choice /C YN /N /M "确认进入 armed 模式？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0
echo [Delta Vision] 启动后请在 5 秒内切回三角洲行动窗口。
timeout /T 5 /NOBREAK >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode Armed -Config "%GAME_CONFIG%" -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"

:finish
echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] Worker 未到达目标或启动失败，退出码: %EXIT_CODE%
) else (
    echo [Delta Vision] 路线运行完成；证据已保存到 artifacts\runs。
)
pause
exit /b %EXIT_CODE%
