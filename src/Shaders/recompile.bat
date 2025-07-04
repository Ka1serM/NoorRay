@echo off
setlocal

set GLSLC=%VULKAN_SDK%/Bin/glslc.exe

echo Compiling shaders...

%GLSLC% PathTracing/RayGeneration.glsl -o PathTracing/RayGeneration.spv --target-env=vulkan1.3
%GLSLC% PathTracing/ClosestHit.glsl -o PathTracing/ClosestHit.spv --target-env=vulkan1.3
%GLSLC% PathTracing/Miss.glsl -o PathTracing/Miss.spv --target-env=vulkan1.3

%GLSLC% Tonemapper.glsl -o Tonemapper.spv --target-env=vulkan1.3

echo Shader compilation complete.