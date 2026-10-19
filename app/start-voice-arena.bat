@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 语音攻防演示平台启动中，页面地址 http://127.0.0.1:8790
node server\index.mjs
pause
