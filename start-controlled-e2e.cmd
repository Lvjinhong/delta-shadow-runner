@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo [Delta Vision] 这会启动独立测试窗口，并通过截图识别和标准 SendInput 控制 WASD。
echo [Delta Vision] 不会读取进程内存，也不会连接或操作游戏。
echo [Delta Vision] F12 为急停键。是否继续？
choice /C YN /N /M "按 Y 继续，按 N 取消: "
if errorlevel 2 exit /b 0

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision.ps1" -Mode ControlledE2E -ConfirmArmed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo [Delta Vision] 运行失败，退出码: %EXIT_CODE%
) else (
    echo [Delta Vision] 受控 E2E 已完成，证据保存在 artifacts\runs。
)
pause
exit /b %EXIT_CODE%
