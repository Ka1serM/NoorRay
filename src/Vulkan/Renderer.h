#pragma once
#include "Vulkan/Context.h"
#include <vector>
#include <functional>

class Renderer {
public:
    Renderer(Context& context);
    ~Renderer();

    void recreateSwapChain();
    vk::CommandBuffer beginFrame();
    bool endFrame(bool waitForCompute);

    // --- Compute Methods ---
    bool isComputeWorkFinished();
    void submitCompute(const std::function<void(vk::CommandBuffer)>& recordComputeCommands);
    void waitForComputeIdle() const;

    // --- Synchronization ---
    vk::Fence getGraphicsWorkFence() const { return graphicsWorkFence.get(); }

    // --- Getters ---
    const std::vector<vk::Image>& getSwapchainImages() const { return swapchainImages; }
    uint32_t getCurrentSwapchainImageIndex() const { return m_imageIndex; }

private:
    void createSwapChain();

    Context& context;
    uint32_t m_currentFrame = 0;
    uint32_t m_imageIndex = 0;

    // --- Swapchain ---
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;

    // --- Per-frame Graphics Resources ---
    struct Frame {
        vk::UniqueCommandBuffer commandBuffer;
        vk::UniqueSemaphore imageAcquiredSemaphore;
        vk::UniqueFence inFlightFence;
    };
    std::vector<Frame> frames;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> imagesInFlightFences;

    // --- Asynchronous Compute Resources ---
    vk::UniqueCommandBuffer computeCommandBuffer;
    vk::UniqueFence computeFence;
    vk::UniqueSemaphore computeFinishedSemaphore;
    bool computeSubmitted = false;

    // --- Dedicated fence for one-off graphics/transfer work (e.g., saving images) ---
    vk::UniqueFence graphicsWorkFence;
};
