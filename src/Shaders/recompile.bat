@echo off
setlocal

set GLSLC=%VULKAN_SDK%/Bin/glslc.exe

echo Compiling Vulkan GLSL Shaders

REM --- RTX pipeline shaders ---
call :compile_shader rgen RTX/RayGeneration.glsl RTX/RayGeneration.spv
call :compile_shader rchit RTX/ClosestHit.glsl RTX/ClosestHit.spv
call :compile_shader rmiss RTX/Miss.glsl RTX/Miss.spv

REM --- Volume tracing shaders ---
call :compile_shader rint RTX/IntersectionVolume.glsl RTX/IntersectionVolume.spv
call :compile_shader rchit RTX/ClosestHitVolume.glsl RTX/ClosestHitVolume.spv

REM --- Compute pipeline shaders ---
call :compile_shader comp Compute/PathTracer.comp Compute/PathTracer.spv "-DUSE_COMPUTE=1"
call :compile_shader comp Tonemapping/Tonemapper.comp Tonemapping/Tonemapper.spv

echo.
echo Shader compilation complete.
exit /b 0


REM Function to compile a single shader
:compile_shader
set STAGE=%~1
set SRC=%~2
set OUT=%~3
set EXTRA=%~4

echo.
echo [INFO] Compiling %SRC% ...
%GLSLC% -fshader-stage=%STAGE% --target-env=vulkan1.3 %EXTRA% %SRC% -o %OUT%
if errorlevel 1 (
    echo [ERROR] Failed to compile %SRC%
    exit /b 1
)
exit /b 0
