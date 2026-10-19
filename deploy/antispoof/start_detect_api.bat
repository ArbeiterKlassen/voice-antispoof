@echo off
REM AASIST 语音辨伪服务启动（与合成服务配套；同一 conda 环境 gptsovits）
setlocal
cd /d "%~dp0"
start "antispoof-api" cmd /c "conda run -n gptsovits python -u detect_api.py --port 9890 > detect_api.log 2>&1"
echo 辨伪服务启动中（模型仅 1.7MB，约 10-20 秒就绪）...
echo 地址 http://127.0.0.1:9890 ；自检：python test_detect.py
pause
