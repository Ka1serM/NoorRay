#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <functional>

class Context {
private:
    GLFWwindow* window;
    vk::detail::DynamicLoader dl;
    bool rtxSupported = false; // Flag to indicate if Ray Tracing is supported

    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;
    vk::Queue queue;
    std::vector<uint32_t> queueFamilyIndices;
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descriptorPool;

    // Helper functions for setup
    void pickPhysicalDevice();
    void createInstance();
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
    GLFWwindow* getWindow() const { return window; }
    bool isRtxSupported() const { return rtxSupported; } // Getter for the RTX support flag
    const vk::Instance& getInstance() const { return instance.get(); }
    const vk::SurfaceKHR& getSurface() const { return surface.get(); }
    const vk::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
    const vk::Device& getDevice() const { return device.get(); }
    const vk::Queue& getQueue() const { return queue; }
    const std::vector<uint32_t>& getQueueFamilyIndices() const { return queueFamilyIndices; }
    const vk::CommandPool& getCommandPool() const { return commandPool.get(); }
    const vk::DescriptorPool& getDescriptorPool() const { return descriptorPool.get(); }
};
