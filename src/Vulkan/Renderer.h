#pragma once
#include "Context.h"
#include <vulkan/vulkan.hpp>
#include <functional>
#include <vector>

class Renderer {
public:
    Renderer(Context& context, uint32_t width, uint32_t height);
    ~Renderer();

    vk::CommandBuffer beginFrame();
    void endFrame(bool waitForCompute);

    // --- MODIFIED: Public method for resizing ---
    void recreateSwapChain(uint32_t width, uint32_t height);

    // --- MODIFIED: Public Getters ---
    const std::vector<vk::Image>& getSwapchainImages() const { return swapchainImages; }
    uint32_t getCurrentSwapchainImageIndex() const { return m_imageIndex; }
    
    // --- Compute Methods (unchanged) ---
    bool isComputeWorkFinished();
    void submitCompute(const std::function<void(vk::CommandBuffer)>& recordComputeCommands);
    void waitForComputeIdle() const;

private:
    // --- ADDED: Helper methods for swapchain management ---
    void createSwapChain();
    void cleanupSwapChain();

    Context& context;
    uint32_t width, height;

    // --- Swapchain resources ---
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores; // Depends on swapchain image count

    // --- Frame resources (unchanged) ---
    std::vector<vk::UniqueCommandBuffer> frameCommandBuffers;
    std::vector<vk::UniqueSemaphore> imageAcquiredSemaphores;
    std::vector<vk::UniqueFence> frameFences;

    // --- Compute resources (unchanged) ---
    vk::UniqueCommandBuffer computeCommandBuffer;
    vk::UniqueFence computeFence;
    vk::UniqueSemaphore computeFinishedSemaphore;
    
    uint32_t m_currentFrame = 0;
    uint32_t m_imageIndex = 0;
};
