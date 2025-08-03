#include "Renderer.h"
#include <iostream>
#include <vector>
#include <algorithm> // For std::clamp

// --- Constructor ---
// Initializes non-swapchain resources and creates the initial swapchain.
Renderer::Renderer(Context& context, uint32_t width, uint32_t height)
    : context(context), width(width), height(height)
{
    // --- Frame resources (for CPU-GPU synchronization) ---
    constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    vk::CommandBufferAllocateInfo graphicsCmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);
    auto cmdBuffers = context.getDevice().allocateCommandBuffersUnique(graphicsCmdAllocInfo);

    frameCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    imageAcquiredSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    frameFences.resize(MAX_FRAMES_IN_FLIGHT);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frameCommandBuffers[i] = std::move(cmdBuffers[i]);
        imageAcquiredSemaphores[i] = context.getDevice().createSemaphoreUnique({});
        frameFences[i] = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    }

    // --- Asynchronous Compute Resources ---
    vk::CommandBufferAllocateInfo computeCmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
    computeCommandBuffer = std::move(context.getDevice().allocateCommandBuffersUnique(computeCmdAllocInfo).front());
    computeFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    computeFinishedSemaphore = context.getDevice().createSemaphoreUnique({});

    // --- Create initial swapchain ---
    createSwapChain();
}

// --- Destructor ---
Renderer::~Renderer()
{
    context.getDevice().waitIdle();
    // All vk::Unique... handles will be automatically destroyed.
    std::cout << "Destroying Renderer" << std::endl;
}

// --- Swapchain creation logic ---
// This function now handles both initial creation and recreation.
void Renderer::createSwapChain() {
    // Clean up old resources before creating new ones.
    // The vk::UniqueSwapchainKHR handle will be automatically released and replaced.
    renderFinishedSemaphores.clear();

    vk::SurfaceCapabilitiesKHR surfaceCapabilities = context.getPhysicalDevice().getSurfaceCapabilitiesKHR(context.getSurface());
    
    vk::Extent2D extent;
    // Check if the surface size is determined by the window manager
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = surfaceCapabilities.currentExtent;
    } else {
        // Otherwise, we choose the size within the supported limits.
        extent.width = std::clamp(width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
        extent.height = std::clamp(height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
    }
    
    // Update our internal width/height to match the actual swapchain size.
    this->width = extent.width;
    this->height = extent.height;

    // Determine the number of images in the swapchain
    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
        imageCount = surfaceCapabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    swapchainInfo.setMinImageCount(imageCount);
    auto swapSurfaceFormat = context.chooseSwapSurfaceFormat();
    swapchainInfo.setImageFormat(swapSurfaceFormat.format);
    swapchainInfo.setImageColorSpace(swapSurfaceFormat.colorSpace);
    swapchainInfo.setImageExtent(extent);
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(surfaceCapabilities.currentTransform);
    swapchainInfo.setPresentMode(context.chooseSwapPresentMode());
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context.getQueueFamilyIndices());
    swapchainInfo.setOldSwapchain(swapchain.get()); // Pass handle to old swapchain for efficient transition

    swapchain = context.getDevice().createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());

    // --- Per-swapchain-image semaphores ---
    renderFinishedSemaphores.resize(swapchainImages.size());
    for (uint32_t i = 0; i < swapchainImages.size(); ++i) {
        renderFinishedSemaphores[i] = context.getDevice().createSemaphoreUnique({});
    }

    std::cout << "Recreated swapchain with " << swapchainImages.size() << " images at " << extent.width << "x" << extent.height << std::endl;
}

// --- Public recreation function ---
// This is called from the Viewer when a resize event occurs.
void Renderer::recreateSwapChain(uint32_t newWidth, uint32_t newHeight) {
    context.getDevice().waitIdle();

    this->width = newWidth;
    this->height = newHeight;

    createSwapChain();

    // Reset frame index to prevent using out-of-bounds semaphores if image count changes.
    m_currentFrame = 0;
}

vk::CommandBuffer Renderer::beginFrame()
{
    // Wait for the fence of the frame we want to use, ensuring its command buffer is free.
    (void)context.getDevice().waitForFences(frameFences[m_currentFrame].get(), true, UINT64_MAX);

    // Acquire the next available image from the swapchain.
    // This can throw vk::OutOfDateKHRError, which we let propagate up to the main loop.
    auto result = context.getDevice().acquireNextImageKHR(
        swapchain.get(),
        UINT64_MAX,
        imageAcquiredSemaphores[m_currentFrame].get(),
        nullptr
    );
    m_imageIndex = result.value;

    // Now that we've successfully acquired an image, we can reset the fence for this frame.
    context.getDevice().resetFences(frameFences[m_currentFrame].get());

    // Begin recording commands for this frame.
    frameCommandBuffers[m_currentFrame]->reset();
    frameCommandBuffers[m_currentFrame]->begin(vk::CommandBufferBeginInfo{});

    return frameCommandBuffers[m_currentFrame].get();
}

void Renderer::endFrame(bool waitForCompute)
{
    frameCommandBuffers[m_currentFrame]->end();

    std::vector<vk::Semaphore> waitSemaphores;
    std::vector<vk::PipelineStageFlags> waitStages;

    // Always wait for the image to be acquired before rendering.
    waitSemaphores.push_back(imageAcquiredSemaphores[m_currentFrame].get());
    waitStages.push_back(vk::PipelineStageFlagBits::eColorAttachmentOutput);

    // If there's compute work, wait for it to finish before the graphics pass.
    if (waitForCompute) {
        waitSemaphores.push_back(computeFinishedSemaphore.get());
        waitStages.push_back(vk::PipelineStageFlagBits::eTransfer);
    }

    vk::SubmitInfo submitInfo{};
    submitInfo.setWaitSemaphores(waitSemaphores);
    submitInfo.setWaitDstStageMask(waitStages);
    submitInfo.setCommandBuffers(frameCommandBuffers[m_currentFrame].get());
    submitInfo.setSignalSemaphores(renderFinishedSemaphores[m_imageIndex].get());

    context.getQueue().submit(submitInfo, frameFences[m_currentFrame].get());

    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphores(renderFinishedSemaphores[m_imageIndex].get());
    presentInfo.setSwapchains(swapchain.get());
    presentInfo.setImageIndices(m_imageIndex);

    // Let this throw vk::OutOfDateKHRError to be caught by the main loop.
    (void)context.getQueue().presentKHR(presentInfo);

    m_currentFrame = (m_currentFrame + 1) % frameFences.size();
}

// --- Compute Queue Methods (Unchanged) ---
bool Renderer::isComputeWorkFinished() {
    return context.getDevice().getFenceStatus(computeFence.get()) == vk::Result::eSuccess;
}

void Renderer::submitCompute(const std::function<void(vk::CommandBuffer)>& recordComputeCommands)
{
    context.getDevice().resetFences(computeFence.get());
    computeCommandBuffer->begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    recordComputeCommands(computeCommandBuffer.get());
    computeCommandBuffer->end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(computeCommandBuffer.get());
    submitInfo.setSignalSemaphores(computeFinishedSemaphore.get());
    context.getQueue().submit(submitInfo, computeFence.get());
}

void Renderer::waitForComputeIdle() const {
    (void)context.getDevice().waitForFences(computeFence.get(), true, UINT64_MAX);
}
