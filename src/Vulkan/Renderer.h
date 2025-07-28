#pragma once
#include "Context.h"
#include "Camera/PerspectiveCamera.h"
#include <vector>
#include <vulkan/vulkan.hpp>
class Renderer {
    Context& context;
    uint32_t width, height;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    vk::UniqueSemaphore imageAcquiredSemaphore;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

public:
    Renderer(Context& context, uint32_t width, uint32_t height);
    ~Renderer(); 
    
    // Getters
    const vk::CommandBuffer& getCommandBuffer(uint32_t imageIndex) const;
    const vk::SwapchainKHR& getSwapChain() const { return swapchain.get(); }
    const std::vector<vk::Image>& getSwapchainImages() const { return swapchainImages; }
};