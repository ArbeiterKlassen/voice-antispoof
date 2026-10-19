@echo off
rem ---------------------------------------------------------------
rem  Start the Cloudflare Tunnel for voice-arena.
rem  Public URL: https://voice.soyorin.work  ->  http://127.0.0.1:8790
rem  Press Ctrl+C in this window to stop the tunnel.
rem
rem  NOTE: this .bat stays ASCII-only on purpose -- cmd.exe parses
rem  .bat with the OEM code page, so Chinese text here would break it.
rem  All Chinese output comes from scripts\start-tunnel.mjs .
rem ---------------------------------------------------------------
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title voice-arena Tunnel  -  Ctrl+C to stop

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found in PATH.
  echo.
  pause
  exit /b 1
)

node "scripts\start-tunnel.mjs" %*
set EXITCODE=%errorlevel%

if not "%EXITCODE%"=="0" (
  echo.
  echo [voice-arena Tunnel] exited with code %EXITCODE%.
  pause
)
endlocal
exit /b %EXITCODE%
