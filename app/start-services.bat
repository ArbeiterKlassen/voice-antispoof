@echo off
chcp 65001 >nul
REM 启动两个模型服务。合成与辨伪共用同一个 conda 环境 gptsovits。
REM 若该环境不存在，说明合成部署包还没解压或环境未创建，按包内 README_DEPLOY.md 先建环境。

set WORKSPACE=%~dp0..
set ANTI=%WORKSPACE%\antispoof_deploy_20261016
set TTS=%WORKSPACE%\gptsovits_deploy_20261016

echo [1/2] 启动辨伪服务（AASIST，端口 9890）
if exist "%ANTI%\start_detect_api.bat" (
    start "antispoof-api" cmd /c ""%ANTI%\start_detect_api.bat""
) else (
    echo 未找到 %ANTI%\start_detect_api.bat，请确认辨伪部署包已解压
)

echo [2/2] 启动合成服务（GPT-SoVITS，端口 9880）
if exist "%TTS%\start_api.bat" (
    start "gptsovits-api" cmd /c ""%TTS%\start_api.bat""
) else if exist "%TTS%\start_tts_api.bat" (
    start "gptsovits-api" cmd /c ""%TTS%\start_tts_api.bat""
) else (
    echo 未找到合成服务的启动脚本，请先解压 gptsovits_deploy_*.tar 并阅读其 README_DEPLOY.md
)

echo.
echo 两个服务就绪后，运行 start-voice-arena.bat 打开页面。
pause
