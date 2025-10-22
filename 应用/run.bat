@echo off
setlocal

REM === 1) 激活 conda 环境 ===
REM 如果这行报错，请把路径改成你的 Miniconda 安装路径
call "E:\Programs\miniconda3\Scripts\activate.bat" qwen_pulse

REM === 2) 进入项目目录（改成你的）===
cd /e "E:\T4b3_Works\AnacondaWorks\final\应用"

REM === 3) 启动服务（若你已在 app_chat.py 里写死了路径和端口，直接运行）===
REM 想在后台窗口运行：/B；想看到日志，删掉 /B
start "" /B python app_chat.py

REM === 4) 简单等待几秒（可调）===
timeout /t 10 >nul

REM === 5) 打开默认浏览器 ===
REM start "" "http://127.0.0.1:7860"

REM === 4) 检测接口是否可用（直到返回200才继续）
echo 正在等待接口启动...
:WAIT_LOOP
for /f "tokens=2 delims=:" %%A in ('curl -s -o nul -w "HTTP:%%{http_code}" http://127.0.0.1:7860') do set code=%%A
if "%code%"=="200" (
    echo 接口已就绪，打开浏览器...
    start "" "http://127.0.0.1:7860"
    goto :END
) else (
    timeout /t 2 >nul
    goto :WAIT_LOOP
)

endlocal
