%VULKAN_SDK%/Bin/glslc.exe raygen.glsl -o raygen.spv --target-env=vulkan1.3
%VULKAN_SDK%/Bin/glslc.exe closesthit.glsl -o closesthit.spv --target-env=vulkan1.3
%VULKAN_SDK%/Bin/glslc.exe miss.glsl -o miss.spv --target-env=vulkan1.3
