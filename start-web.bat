@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   VTuber Subtitle Web 一键启动
echo ============================================
echo.

rem 可选：模型缓存放到 D 盘（本机 C 盘空间较小）
if exist "D:\hf-cache" set "HF_HOME=D:\hf-cache"

rem 可选：CUDA 运行库（GPU 识别用）
set "CUDA_BIN=D:\vtuber-cuda\nvidia\cudnn\bin;D:\vtuber-cuda\nvidia\cublas\bin;D:\vtuber-cuda\nvidia\cuda_nvrtc\bin"
if exist "D:\vtuber-cuda" set "PATH=%CUDA_BIN%;%PATH%"

rem 可选：FFmpeg
set "FFMPEG_BIN=C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin"
if exist "%FFMPEG_BIN%" set "PATH=%FFMPEG_BIN%;%PATH%"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先运行:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
)

echo 正在启动 Web 服务 (http://127.0.0.1:8000) ...
echo 关闭本窗口即停止服务。

start "VTuber-Subtitle-Server" cmd /k ".venv\Scripts\python.exe -m vtuber_subtitle.web"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000"
exit /b 0
