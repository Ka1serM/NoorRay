@echo off
setlocal

set GLSLC=%VULKAN_SDK%/Bin/glslc.exe

echo Compiling shaders...

%GLSLC% PathTracing/RTX/RayGeneration.glsl -o PathTracing/RayGeneration.spv --target-env=vulkan1.3
%GLSLC% PathTracing/RTX/ClosestHit.glsl -o PathTracing/ClosestHit.spv --target-env=vulkan1.3
%GLSLC% PathTracing/RTX/Miss.glsl -o PathTracing/Miss.spv --target-env=vulkan1.3

%GLSLC% PathTracing/PathTracer.comp -o PathTracing/PathTracer.spv --target-env=vulkan1.3 -DUSE_COMPUTE=1

%GLSLC% Tonemapper.comp -o Tonemapper.spv --target-env=vulkan1.3

echo Shader compilation complete.