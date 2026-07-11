#pragma once
#include "Vulkan/Context.h"
#include <vector>

class Renderer {
public:
    explicit Renderer(Context& context, uint32_t initial_width, uint32_t initial_height);
    ~Renderer();

    bool beginFrame(); // returns false if swapchain is invalid or resized
    void endFrame();

    void notifyResize(uint32_t width, uint32_t height); // called from window resize event

    uint32_t getNumSwapchainImages() const { return swapchainImages.size(); }
    vk::Extent2D getSwapchainExtent() const { return swapchainExtent; }
    vk::Image getCurrentColorImage() const { return swapchainImages[m_imageIndex]; }
    vk::ImageView getCurrentColorImageView() const { return swapchainImageViews[m_imageIndex].get(); }
    vk::Format getColorImageFormat() const { return colorImageFormat; }
    vk::ImageView getDepthImageView() const { return depthImageView.get(); }
    vk::Format getDepthImageFormat() const { return depthImageFormat; }
    vk::CommandBuffer getCurrentCommandBuffer() const { return frames[m_currentFrame].commandBuffer.get(); }
    vk::Fence getCurrentInFlightFence() const { return frames[m_currentFrame].inFlightFence.get(); }

    void setExternalFrameSync(
        vk::Semaphore renderReady,
        vk::Semaphore bufferReleased,
        uint64_t value);

private:
    // Upper bound on frames-in-flight. The actual count is clamped to the swapchain
    // image count so that CPU submissions never lap a swapchain image (or an ImGui
    // vertex/index buffer, which cycles once per swapchain image) that the GPU is
    // still reading — otherwise the ImGui overlay shows torn/garbage geometry.
    static constexpr uint32_t kMaxFramesInFlight = 3;
    uint32_t queryDesiredSwapchainImageCount() const;

    void recreateSwapChain();

    Context& context;

    // Frame state
    uint32_t m_currentFrame = 0;
    uint32_t m_imageIndex = 0;
    bool m_framebufferResized = false;

    // Swapchain
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    vk::Format colorImageFormat{};
    vk::Extent2D swapchainExtent{};
    std::vector<vk::UniqueImageView> swapchainImageViews;

    // Depth Buffer
    vk::UniqueImage depthImage;
    vk::UniqueDeviceMemory depthImageMemory;
    vk::UniqueImageView depthImageView;
    vk::Format depthImageFormat{};

    // Per-frame Graphics Resources
    struct Frame {
        vk::UniqueCommandBuffer commandBuffer;
        vk::UniqueSemaphore imageAcquiredSemaphore;
        vk::UniqueSemaphore renderFinishedSemaphore; 
        vk::UniqueFence inFlightFence;
    };
    std::vector<Frame> frames;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;

    vk::Semaphore externalRenderReady{};
    vk::Semaphore externalBufferReleased{};
    uint64_t externalTimelineValue{};
};
