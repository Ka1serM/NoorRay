#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_beta.h>
#include <vector>
#include <functional>
#include <mutex>

#include "SDL3/SDL_video.h"

class Context {

    std::vector<const char*> RequiredDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
#ifdef __APPLE__
        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif
    };


    std::vector<const char*> RayTracingExtensions = {
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    
    SDL_Window* window;
    int windowWidth;
    int windowHeight;
    float dpiScale;

    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;
    vk::Queue queue;
    std::vector<uint32_t> queueFamilyIndices;
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descriptorPool;

    bool rtxSupported = false;

    void createVulkanInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();

public:
    Context(int width, int height);
    ~Context();

    // Helper functions
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func);
    vk::PresentModeKHR chooseSwapPresentMode() const;
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat() const;

    // Static callback with corrected C++ types
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugUtilsMessengerCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

    // Getters
    SDL_Window* getWindow() const { return window; }
    uint32_t getWindowWidth() const { return windowWidth; }
    uint32_t getWindowHeight() const { return windowHeight; }
    float getDPIScale() const { return dpiScale; }
    
    const vk::Instance& getInstance() const { return instance.get(); }
    const vk::SurfaceKHR& getSurface() const { return surface.get(); }
    const vk::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
    const vk::Device& getDevice() const { return device.get(); }
    const vk::Queue& getQueue() const { return queue; }
    const std::vector<uint32_t>& getQueueFamilyIndices() const { return queueFamilyIndices; }
    const vk::CommandPool& getCommandPool() const { return commandPool.get(); }
    const vk::DescriptorPool& getDescriptorPool() const { return descriptorPool.get(); }

    void queryWindowSize();

    bool isRtxSupported() const { return rtxSupported; }
};
