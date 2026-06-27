#pragma once

#include <string>

#include <vulkan/vulkan.hpp>

int selectCudaDeviceForVulkan(vk::PhysicalDevice physicalDevice);
std::string cudaDeviceName(int deviceIndex);
