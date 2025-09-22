@echo off
setlocal

set GLSLC=%VULKAN_SDK%/Bin/glslc.exe

echo Compiling shaders...

REM Helper macro to compile shaders and report errors
call :compile_shader "RTX/RayGeneration.glsl" "RTX/RayGeneration.spv" "--target-env=vulkan1.3"
call :compile_shader "RTX/ClosestHit.glsl" "RTX/ClosestHit.spv" "--target-env=vulkan1.3"
call :compile_shader "RTX/Miss.glsl" "RTX/Miss.spv" "--target-env=vulkan1.3"
call :compile_shader "Compute/PathTracer.comp" "Compute/PathTracer.spv" "-DUSE_COMPUTE=1"
call :compile_shader "Tonemapping/Tonemapper.comp" "Tonemapping/Tonemapper.spv" ""

REM Rml Shaders
call :compile_shader "Rml/RmlFrag.frag" "Rml/RmlFrag.spv" ""
call :compile_shader "Rml/RmlVert.vert" "Rml/RmlVert.spv" ""
call :compile_shader "Rml/RmlGradientFrag.frag" "Rml/RmlGradientFrag.spv" ""
call :compile_shader "Rml/RmlFilterFrag.frag" "Rml/RmlFilterFrag.spv" ""

call :compile_shader "Rml/RmlPassthroughFrag.frag" "Rml/RmlPassthroughFrag.spv" ""
call :compile_shader "Rml/RmlPassthroughVert.vert" "Rml/RmlPassthroughVert.spv" ""

echo Shader compilation complete.
exit /b

:compile_shader
set SHADER_SRC=%~1
set SHADER_OUT=%~2
set SHADER_FLAGS=%~3

echo Compiling %SHADER_SRC%...
%GLSLC% %SHADER_SRC% -o %SHADER_OUT% %SHADER_FLAGS%
if errorlevel 1 (
    echo ERROR: Failed to compile %SHADER_SRC%
    exit /b 1
)
exit /b 0
