#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vector>
#include <functional>

#include "SDL3/SDL_video.h"

class Context {
private:
    SDL_Window* window;
    vk::detail::DynamicLoader dl;

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
    bool checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const;
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const;
    vk::UniqueDescriptorSet allocateDescSet(vk::DescriptorSetLayout descSetLayout);
    vk::PresentModeKHR chooseSwapPresentMode() const;
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat() const; // New helper function

    // Static callback with corrected C++ types
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugUtilsMessengerCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

    // Getters
    SDL_Window* getWindow() const { return window; }
    const vk::Instance& getInstance() const { return instance.get(); }
    const vk::SurfaceKHR& getSurface() const { return surface.get(); }
    const vk::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
    const vk::Device& getDevice() const { return device.get(); }
    const vk::Queue& getQueue() const { return queue; }
    const std::vector<uint32_t>& getQueueFamilyIndices() const { return queueFamilyIndices; }
    const vk::CommandPool& getCommandPool() const { return commandPool.get(); }
    const vk::DescriptorPool& getDescriptorPool() const { return descriptorPool.get(); }

    bool isRtxSupported() const { return rtxSupported; }
};
