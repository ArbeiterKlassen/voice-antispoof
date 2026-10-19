@echo off
REM GPT-SoVITS v4 API 启动（含自动预热）
REM 说明：预热请求是必须的——首跑含 CUDA 图捕获，慢且输出不稳（见 README_DEPLOY.md 已知坑 1/6）。
setlocal
cd /d "%~dp0GPT-SoVITS"

echo [1/2] 启动 API 服务（日志 api_v2.log，模型加载约 30-60 秒）...
start "gptsovits-api" cmd /c "conda run -n gptsovits python -u api_v2.py > api_v2.log 2>&1"

echo [2/2] 等待服务就绪...
:waitloop
timeout /t 5 /nobreak >nul
findstr /C:"Uvicorn running" api_v2.log >nul 2>&1
if errorlevel 1 goto waitloop
echo     服务已就绪，发送预热请求...
curl -s -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" -d "{\"text\":\"预热。\",\"text_lang\":\"zh\",\"ref_audio_path\":\"%~dp0voices\\ref_user1.wav\",\"prompt_text\":\"一大早沁凉的风就暗暗催送时间拨弄湖水的声音。\",\"prompt_lang\":\"zh\",\"streaming_mode\":false,\"media_type\":\"wav\"}" -o "%~dp0out\warmup.wav"

echo.
echo 启动完成：http://127.0.0.1:9880（预热输出 out\warmup.wav）
echo 自检：python test_acceptance.py
pause
