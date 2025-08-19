#!/bin/bash

GLSLC="$VULKAN_SDK/Bin/glslc"

echo "Compiling shaders..."

# RTX shaders
$GLSLC RTX/RayGeneration.glsl -o RTX/RayGeneration.spv --target-env=vulkan1.3
$GLSLC RTX/ClosestHit.glsl -o RTX/ClosestHit.spv --target-env=vulkan1.3
$GLSLC RTX/Miss.glsl -o RTX/Miss.spv --target-env=vulkan1.3

# Compute shader
$GLSLC Compute/PathTracer.comp -o Compute/PathTracer.spv -DUSE_COMPUTE=1

# Tonemapper shader
$GLSLC Tonemapping/Tonemapper.comp -o Tonemapping/Tonemapper.spv

echo "Shader compilation complete."
