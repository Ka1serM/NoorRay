#include "GPU/CudaDevice.h"

#include <cstring>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "GPU/Checks.h"

int selectCudaDeviceForVulkan(const vk::PhysicalDevice physicalDevice)
{
    vk::PhysicalDeviceIDProperties idProperties{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &idProperties;
    physicalDevice.getProperties2(&properties);

    int deviceCount = 0;
    NR_GPU_CHECK(cudaGetDeviceCount(&deviceCount));
    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
    {
        cudaDeviceProp cudaProperties{};
        NR_GPU_CHECK(cudaGetDeviceProperties(&cudaProperties, deviceIndex));
        if (std::memcmp(idProperties.deviceUUID, cudaProperties.uuid.bytes, VK_UUID_SIZE) == 0)
        {
            NR_GPU_CHECK(cudaSetDevice(deviceIndex));
            return deviceIndex;
        }
    }

    throw std::runtime_error("No CUDA device matches the selected Vulkan physical device UUID");
}

std::string cudaDeviceName(const int deviceIndex)
{
    cudaDeviceProp properties{};
    NR_GPU_CHECK(cudaGetDeviceProperties(&properties, deviceIndex));
    return properties.name;
}
