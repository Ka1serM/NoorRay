#pragma once
#include "Context.h"
#include <vector>
#include <vulkan/vulkan.hpp>

class Renderer {
public:
    Renderer(Context& context, uint32_t width, uint32_t height);
    ~Renderer(); 
    
    void advanceFrame();
    
    // Getters for the current frame's sync objects
    vk::Semaphore getCurrentImageAcquiredSemaphore() const { return m_imageAcquiredSemaphores[m_currentFrame].get(); }
    vk::Semaphore getCurrentRenderFinishedSemaphore() const { return m_renderFinishedSemaphores[m_currentFrame].get(); }
    vk::Fence getCurrentInFlightFence() const { return m_inFlightFences[m_currentFrame].get(); }
    
    const vk::SwapchainKHR& getSwapChain() const { return swapchain.get(); }
    const std::vector<vk::Image>& getSwapchainImages() const { return swapchainImages; }
    const vk::CommandBuffer& getCommandBuffer(uint32_t imageIndex) const;

private:
    Context& context;
    uint32_t width, height;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    
    static const int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<vk::UniqueSemaphore> m_imageAcquiredSemaphores;
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores;
    std::vector<vk::UniqueFence> m_inFlightFences;
    uint32_t m_currentFrame = 0;
};