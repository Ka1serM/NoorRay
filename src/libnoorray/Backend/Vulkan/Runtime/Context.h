#pragma once

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_beta.h>
#include <cstdint>
#include <vector>
#include <functional>
#include <mutex>

#include "vk_mem_alloc.h"

#include <cuda_runtime_api.h>
#include <optix.h>

#include "Backend/CUDA/Unique/OptixDeviceContext.h"
#include "Backend/CUDA/Unique/Stream.h"

class VulkanSurfaceProvider
{
public:
    virtual ~VulkanSurfaceProvider() = default;
    virtual PFN_vkGetInstanceProcAddr getVulkanInstanceProcAddr() const = 0;
    virtual std::vector<const char*> getRequiredVulkanInstanceExtensions() const = 0;
    virtual vk::SurfaceKHR createVulkanSurface(vk::Instance instance) const = 0;
    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;
};

class Context {

    std::vector<const char*> RequiredDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#ifdef __APPLE__
        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif
    };

    bool headless = false;
    bool raytracingAvailable = false;
    VulkanSurfaceProvider* surfaceProvider{};
    bool validationEnabled = false;

    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;

    VmaAllocator allocator{};

    nr::cuda::UniqueOptixDeviceContext optixCtx;
    nr::cuda::UniqueStream cudaStream;
    
    vk::Queue graphicsQueue;
    uint32_t graphicsFamilyIndex = UINT32_MAX;
    
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descriptorPool;
    

    void createVulkanInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();
    explicit Context(VulkanSurfaceProvider* provider);
public:
    Context();
    explicit Context(VulkanSurfaceProvider& provider);
    ~Context();

    // Helper functions
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func);
    vk::PresentModeKHR chooseSwapPresentMode() const;

    // Static callback with corrected C++ types
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugUtilsMessengerCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

    // Getters
    const vk::Instance& getInstance() const { return instance.get(); }
    const vk::SurfaceKHR& getSurface() const { return surface.get(); }
    const vk::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
    const vk::Device& getDevice() const { return device.get(); }
    const vk::Queue& getGraphicsQueue() const { return graphicsQueue; }
    uint32_t getGraphicsFamilyIndex() const { return graphicsFamilyIndex; }
    const vk::CommandPool& getCommandPool() const { return commandPool.get(); }
    const vk::DescriptorPool& getDescriptorPool() const { return descriptorPool.get(); }
    VmaAllocator getAllocator() const { return allocator; }
    bool isHeadless() const { return headless; }
    // False when CUDA or OptiX could not be brought up on this machine - no
    // NVIDIA driver, no compatible device, or an OptiX version mismatch. Vulkan
    // is still fully usable in that state, so a host can keep its UI running and
    // present something other than a raytraced image. Anything that touches the
    // Raytracer, interop images or CUDA streams must check this first.
    bool supportsRaytracing() const { return raytracingAvailable; }

    OptixDeviceContext getOptixContext() const { return optixCtx.get(); }
    cudaStream_t getCudaStream() const { return cudaStream.get(); }

};
