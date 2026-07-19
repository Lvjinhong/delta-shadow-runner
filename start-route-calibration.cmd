@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "GAME_CONFIG=configs\game-route.json"
set "EXAMPLE_CONFIG=configs\game-route.example.json"
set "MENU_PROFILE=%~1"
if not defined MENU_PROFILE set "MENU_PROFILE=profiles\menu-zero-cost\menu.json"

echo [Delta Vision] D = 路线采集 dry-run（确认 1920x1080 和局内 HUD，不发送输入）
echo [Delta Vision] A = 路线采集 armed（执行 route_capture 中的受保护短脉冲）
echo [Delta Vision] M = 旧版人工采样（自动进图后只读采样 120 秒）
echo [Delta Vision] Q = 退出
choice /C DAMQ /N /M "请选择 D/A/M/Q: "
if errorlevel 4 exit /b 0
if errorlevel 3 goto manual
if errorlevel 2 goto armed

call :require_game_config
if errorlevel 1 exit /b 2
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0vision.ps1" ^
    -Mode RouteCaptureDryRun ^
    -Config "%GAME_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:armed
call :require_game_config
if errorlevel 1 exit /b 2
echo [Delta Vision] 请先停在已确认的零号大坝局内 HUD；F12 为急停键。
echo [Delta Vision] route_capture.armed_ready 必须为 true；首次只保留一个最短 w 脉冲。
choice /C YN /N /M "确认发送 route_capture 中的键鼠动作？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0
echo [Delta Vision] 启动后请在 5 秒内切回三角洲行动窗口。
timeout /T 5 /NOBREAK >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0vision.ps1" ^
    -Mode RouteCaptureArmed ^
    -Config "%GAME_CONFIG%" ^
    -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:manual
if not exist "%MENU_PROFILE%" (
    echo [Delta Vision] 尚未找到 %MENU_PROFILE%。
    echo [Delta Vision] 请先部署与你当前 UI 匹配的菜单 Profile bundle。
    pause
    exit /b 2
)
echo [Delta Vision] 人工模式会通过菜单自动进图，确认 IN_MATCH 后只读采样 120 秒。
echo [Delta Vision] 看到 HUD 后请人工走固定短路线；F12 为急停。
choice /C YN /N /M "确认开始人工 calibration？按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0
echo [Delta Vision] 请在 5 秒内切回三角洲行动窗口。
timeout /T 5 /NOBREAK >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0vision.ps1" ^
    -Mode SessionSample ^
    -ProfilePath "%MENU_PROFILE%" ^
    -WindowTitle "三角洲行动  " ^
    -Backend dxcam ^
    -Split calibration ^
    -Duration 120 ^
    -SampleFps 5 ^
    -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"

:finish
echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] 路线采集未完成，退出码: %EXIT_CODE%。
) else (
    echo [Delta Vision] 路线采集完成；数据集和审计摘要已保存到 artifacts\runs。
)
pause
exit /b %EXIT_CODE%

:require_game_config
if exist "%GAME_CONFIG%" exit /b 0
echo [Delta Vision] 尚未找到 %GAME_CONFIG%。
echo [Delta Vision] 先复制 %EXAMPLE_CONFIG%，再填写精确窗口标题和 route_capture 计划。
echo copy "%EXAMPLE_CONFIG%" "%GAME_CONFIG%"
pause
exit /b 2
