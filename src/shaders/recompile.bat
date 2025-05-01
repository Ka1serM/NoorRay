@echo off
setlocal

set GLSLC=%VULKAN_SDK%/Bin/glslc.exe

echo Compiling shaders...

%GLSLC% RayGeneration.glsl -o RayGeneration.spv --target-env=vulkan1.3
%GLSLC% RayTracing/ClosestHit.glsl -o RayTracing/ClosestHit.spv --target-env=vulkan1.3
%GLSLC% PathTracing/ClosestHit.glsl -o PathTracing/ClosestHit.spv --target-env=vulkan1.3
%GLSLC% ShadowRay/ClosestHit.glsl -o ShadowRay/ClosestHit.spv --target-env=vulkan1.3
%GLSLC% RayTracing/Miss.glsl -o RayTracing/Miss.spv --target-env=vulkan1.3
%GLSLC% PathTracing/Miss.glsl -o PathTracing/Miss.spv --target-env=vulkan1.3
%GLSLC% ShadowRay/Miss.glsl -o ShadowRay/Miss.spv --target-env=vulkan1.3

echo Shader compilation complete.