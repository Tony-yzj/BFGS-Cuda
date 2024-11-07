@echo off
chcp 65001 > nul

REM 检查是否有传入参数
if "%~1"=="" (
    echo No input parameters detected. Compiling with CMake...

    REM 检查是否存在 build 文件夹，如果不存在则创建
    if not exist build (
        mkdir build
        echo Created build directory.
    )

    REM 使用 CMake 生成构建系统
    cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1
    REM cmake . -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON -B build

    REM 使用 CMake 编译项目
    cmake --build build --parallel 16
    
) 