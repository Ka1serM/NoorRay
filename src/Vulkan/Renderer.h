#pragma once
#include "Context.h"
#include <functional>
#include <vector>

class Renderer {
public:
    Renderer(Context& context, uint32_t width, uint32_t height);
    ~Renderer();

    void recreateSwapChain(uint32_t newWidth, uint32_t newHeight);

    // Returns null on swapchain failure, indicating a recreate is needed.
    vk::CommandBuffer beginFrame();
    
    // Returns true if swapchain needs recreation.
    bool endFrame(bool waitForCompute);

    // --- Compute ---
    void submitCompute(const std::function<void(vk::CommandBuffer)>& recordComputeCommands);
    bool isComputeWorkFinished();
    void waitForComputeIdle() const;

    // --- Getters ---
    vk::SwapchainKHR getSwapchain() const { return swapchain.get(); }
    const std::vector<vk::Image>& getSwapchainImages() const { return swapchainImages; }
    uint32_t getCurrentSwapchainImageIndex() const { return m_imageIndex; }

private:
    void createSwapChain();

    Context& context;
    uint32_t width, height;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    
    // --- Per-frame-in-flight resources ---
    struct FrameData {
        vk::UniqueCommandBuffer commandBuffer;
        vk::UniqueSemaphore imageAcquiredSemaphore;
        // renderFinishedSemaphore is REMOVED from here.
        vk::UniqueFence inFlightFence;
    };
    std::vector<FrameData> frames;
    uint32_t m_currentFrame = 0;
    uint32_t m_imageIndex = 0;

    // --- Per-swapchain-image resources ---
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> imagesInFlightFences;

    // --- Asynchronous Compute Resources ---
    vk::UniqueCommandBuffer computeCommandBuffer;
    vk::UniqueFence computeFence;
    vk::UniqueSemaphore computeFinishedSemaphore;
    bool computeSubmitted = false;
};
