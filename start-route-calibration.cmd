@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "MENU_PROFILE=%~1"
if not defined MENU_PROFILE set "MENU_PROFILE=profiles\menu-zero-cost\menu.json"

if not exist "%MENU_PROFILE%" (
    echo [Delta Vision] 尚未找到 %MENU_PROFILE%。
    echo [Delta Vision] 请先按 README 生成并部署与你当前 UI 匹配的菜单 Profile bundle。
    pause
    exit /b 2
)

echo [Delta Vision] 本流程会通过截图确认页面并发送菜单键鼠输入。
echo [Delta Vision] 确认 IN_MATCH 后会立即停止自动输入，只读采样 120 秒。
echo [Delta Vision] 看到 HUD 后请立刻人工走固定短路线；F12 为急停。
choice /C YN /N /M "确认开始进图并采集 calibration？按 Y 继续，按 N 取消: "
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

echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] 进图采样未完成，退出码: %EXIT_CODE%。
) else (
    echo [Delta Vision] calibration 数据集和审计摘要已保存到 artifacts\runs。
)
pause
exit /b %EXIT_CODE%
