#include "Renderer.h"
#include <iostream>
#include <vector>
#include <algorithm>

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;

Renderer::Renderer(Context& context, const uint32_t width, const uint32_t height)
    : context(context), width(width), height(height)
{
    // --- Graphics Frame Resources ---
    frames.resize(MAX_FRAMES_IN_FLIGHT);
    
    vk::CommandBufferAllocateInfo cmdAllocInfo(
        context.getCommandPool(), vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);
    auto cmdBuffers = context.getDevice().allocateCommandBuffersUnique(cmdAllocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frames[i].commandBuffer = std::move(cmdBuffers[i]);
        frames[i].imageAcquiredSemaphore = context.getDevice().createSemaphoreUnique({});
        frames[i].inFlightFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    }

    // --- Asynchronous Compute Resources ---
    vk::CommandBufferAllocateInfo computeCmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
    computeCommandBuffer = std::move(context.getDevice().allocateCommandBuffersUnique(computeCmdAllocInfo).front());
    computeFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    computeFinishedSemaphore = context.getDevice().createSemaphoreUnique({});
    computeSubmitted = false;

    // --- Create initial swapchain ---
    createSwapChain();
}

// --- Destructor ---
Renderer::~Renderer() {
    context.getDevice().waitIdle();
    std::cout << "Destroying Renderer" << std::endl;
}

// --- Swapchain creation logic ---
void Renderer::createSwapChain() {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = context.getPhysicalDevice().getSurfaceCapabilitiesKHR(context.getSurface());

    vk::Extent2D extent;
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = surfaceCapabilities.currentExtent;
    } else {
        extent.width = std::clamp(width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
        extent.height = std::clamp(height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
    }
    
    if (extent.width == 0 || extent.height == 0) {
        swapchain.reset();
        swapchainImages.clear();
        return; 
    }

    this->width = extent.width;
    this->height = extent.height;

    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount)
        imageCount = surfaceCapabilities.maxImageCount;

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
    swapchainInfo.setOldSwapchain(swapchain.get());

    swapchain = context.getDevice().createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());
    
    // --- Create per-image semaphores and fences ---
    renderFinishedSemaphores.resize(swapchainImages.size());
    for(size_t i = 0; i < swapchainImages.size(); ++i)
        renderFinishedSemaphores[i] = context.getDevice().createSemaphoreUnique({});
    
    imagesInFlightFences.resize(swapchainImages.size(), VK_NULL_HANDLE);
    
    std::cout << "Recreated swapchain with " << swapchainImages.size() << " images at " << extent.width << "x" << extent.height << std::endl;
}

// --- Public recreation function ---
void Renderer::recreateSwapChain(uint32_t newWidth, uint32_t newHeight) {
    context.getDevice().waitIdle();
    this->width = newWidth;
    this->height = newHeight;
    createSwapChain();
    m_currentFrame = 0;
    computeSubmitted = false; 
}

// --- Frame begin ---
vk::CommandBuffer Renderer::beginFrame() {
    // 1. Wait for a frame resource slot to be available.
    (void)context.getDevice().waitForFences(frames[m_currentFrame].inFlightFence.get(), true, UINT64_MAX);

    if (!swapchain) return nullptr;

    // 2. Acquire an image from the swapchain.
    auto result = context.getDevice().acquireNextImageKHR(
        swapchain.get(),
        UINT64_MAX,
        frames[m_currentFrame].imageAcquiredSemaphore.get(),
        nullptr
    );

    if (result.result == vk::Result::eErrorOutOfDateKHR || result.result == vk::Result::eSuboptimalKHR) {
        return nullptr; // Signal to Viewer that a recreate is needed.
    }
    m_imageIndex = result.value;

    // 3. Check if a previous frame is still using this image.
    if (imagesInFlightFences[m_imageIndex] != VK_NULL_HANDLE)
        (void)context.getDevice().waitForFences(1, &imagesInFlightFences[m_imageIndex], true, UINT64_MAX);
    
    // 4. Mark the image as now being in use by the current frame.
    imagesInFlightFences[m_imageIndex] = frames[m_currentFrame].inFlightFence.get();

    // 5. Now that all waiting is done, reset the fence and command buffer for this frame.
    context.getDevice().resetFences(frames[m_currentFrame].inFlightFence.get());
    
    frames[m_currentFrame].commandBuffer->reset();
    frames[m_currentFrame].commandBuffer->begin(vk::CommandBufferBeginInfo{});

    return frames[m_currentFrame].commandBuffer.get();
}

// --- Frame end ---
bool Renderer::endFrame(bool waitForCompute) {
    frames[m_currentFrame].commandBuffer->end();

    std::vector<vk::Semaphore> waitSemaphores;
    std::vector<vk::PipelineStageFlags> waitStages;

    waitSemaphores.push_back(frames[m_currentFrame].imageAcquiredSemaphore.get());
    waitStages.push_back(vk::PipelineStageFlagBits::eColorAttachmentOutput);

    if (waitForCompute && computeSubmitted) {
        waitSemaphores.push_back(computeFinishedSemaphore.get());
        waitStages.push_back(vk::PipelineStageFlagBits::eTransfer);
    }

    vk::SubmitInfo submitInfo{};
    submitInfo.setWaitSemaphores(waitSemaphores);
    submitInfo.setWaitDstStageMask(waitStages);
    submitInfo.setCommandBuffers(frames[m_currentFrame].commandBuffer.get());
    
    // --- Use the semaphore corresponding to the acquired image index ---
    submitInfo.setSignalSemaphores(renderFinishedSemaphores[m_imageIndex].get());

    context.getQueue().submit(submitInfo, frames[m_currentFrame].inFlightFence.get());

    if (!swapchain)
        return true; // Swapchain was destroyed (e.g., minimized), signal recreate.

    vk::PresentInfoKHR presentInfo{};
    // Wait on the semaphore corresponding to the acquired image index ---
    presentInfo.setWaitSemaphores(renderFinishedSemaphores[m_imageIndex].get());
    presentInfo.setSwapchains(swapchain.get());
    presentInfo.setImageIndices(m_imageIndex);

    vk::Result result = context.getQueue().presentKHR(presentInfo);
    
    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    
    if(waitForCompute) {
        computeSubmitted = false;
    }

    return (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR);
}

// --- Compute Methods ---
bool Renderer::isComputeWorkFinished() {
    return context.getDevice().getFenceStatus(computeFence.get()) == vk::Result::eSuccess;
}

void Renderer::submitCompute(const std::function<void(vk::CommandBuffer)>& recordComputeCommands) {
    context.getDevice().resetFences(computeFence.get());

    computeCommandBuffer->begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    recordComputeCommands(computeCommandBuffer.get());
    computeCommandBuffer->end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(computeCommandBuffer.get());
    submitInfo.setSignalSemaphores(computeFinishedSemaphore.get());

    context.getQueue().submit(submitInfo, computeFence.get());
    computeSubmitted = true;
}

void Renderer::waitForComputeIdle() const {
    (void)context.getDevice().waitForFences(computeFence.get(), true, UINT64_MAX);
}
