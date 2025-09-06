#include "Renderer.h"
#include <iostream>
#include <vector>
#include <algorithm>

#include "Log.h"

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

Renderer::Renderer(Context& context)
    : context(context)
{
    // Graphics Frame Resources 
    frames.resize(MAX_FRAMES_IN_FLIGHT);

    const vk::CommandBufferAllocateInfo cmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);
    auto cmdBuffers = context.getDevice().allocateCommandBuffersUnique(cmdAllocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frames[i].commandBuffer = std::move(cmdBuffers[i]);
        frames[i].imageAcquiredSemaphore = context.getDevice().createSemaphoreUnique({});
        frames[i].inFlightFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    }

    //  Asynchronous Compute Resources 
    const vk::CommandBufferAllocateInfo computeCmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
    computeCommandBuffer = std::move(context.getDevice().allocateCommandBuffersUnique(computeCmdAllocInfo).front());
    computeFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    computeFinishedSemaphore = context.getDevice().createSemaphoreUnique({});
    computeSubmitted = false;

    //  Create initial swapchain 
    createSwapChain();
}

//  Destructor 
Renderer::~Renderer() {
    context.getDevice().waitIdle();
    LOG_INFO("Destroying Renderer");
}

//  Swapchain creation 
void Renderer::createSwapChain() {
    const vk::SurfaceCapabilitiesKHR surfaceCapabilities = context.getPhysicalDevice().getSurfaceCapabilitiesKHR(context.getSurface());

    vk::Extent2D extent;
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = surfaceCapabilities.currentExtent;
    } else {
        extent.width = std::clamp(context.getWindowWidth(), surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
        extent.height = std::clamp(context.getWindowHeight(), surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
    }
    
    if (extent.width == 0 || extent.height == 0) {
        swapchain.reset();
        swapchainImages.clear();
        return; 
    }
    
    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount)
        imageCount = surfaceCapabilities.maxImageCount;

    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    swapchainInfo.setMinImageCount(imageCount);
    const auto swapSurfaceFormat = context.chooseSwapSurfaceFormat();
    swapchainInfo.setImageFormat(swapSurfaceFormat.format);
    swapchainInfo.setImageColorSpace(swapSurfaceFormat.colorSpace);
    swapchainInfo.setImageExtent(extent);
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(surfaceCapabilities.currentTransform);
    swapchainInfo.setPresentMode(context.chooseSwapPresentMode());
    swapchainInfo.setClipped(true);
    std::vector queueFamilyIndices{context.getQueueFamilyIndex()};
    swapchainInfo.setQueueFamilyIndices(queueFamilyIndices);
    swapchainInfo.setOldSwapchain(swapchain.get());

    swapchain = context.getDevice().createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());
    
    //  Create per-image semaphores and fences 
    renderFinishedSemaphores.resize(swapchainImages.size());
    for(size_t i = 0; i < swapchainImages.size(); ++i)
        renderFinishedSemaphores[i] = context.getDevice().createSemaphoreUnique({});
    
    imagesInFlightFences.assign(swapchainImages.size(), VK_NULL_HANDLE);
    
    LOG_INFO("Recreated swapchain with " << swapchainImages.size() << " images at " << extent.width << "x" << extent.height);
}

void Renderer::recreateSwapChain() {
    context.getDevice().waitIdle();
    createSwapChain();
    m_currentFrame = 0;
    computeSubmitted = false; 
}

//  Frame begin 
vk::CommandBuffer Renderer::beginFrame() {
    // 1. Wait for a frame resource slot to be available.
    (void)context.getDevice().waitForFences(frames[m_currentFrame].inFlightFence.get(), true, UINT64_MAX);

    if (!swapchain)
        return nullptr;

    // 2. Acquire an image from the swapchain.
    const auto result = context.getDevice().acquireNextImageKHR(swapchain.get(), UINT64_MAX, frames[m_currentFrame].imageAcquiredSemaphore.get(), nullptr);

    if (result.result == vk::Result::eErrorOutOfDateKHR || result.result == vk::Result::eSuboptimalKHR)
        return nullptr; // Signal to Viewer that a recreate is needed
    
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

//  Frame end 
// Renderer.cpp - Only the modified endFrame function is shown.
// The rest of the file remains the same.

/**
 * @brief Ends the graphics frame, submitting the recorded command buffer and presenting the image.
 * This version is decoupled from the compute queue.
 * @return True if the swapchain is out of date and needs to be recreated.
 */
bool Renderer::endFrame() { 
    frames[m_currentFrame].commandBuffer->end();

    // The graphics submission now only waits for the swapchain image to be acquired.
    // It no longer waits for the compute semaphore.
    vk::Semaphore waitSemaphores[] = { frames[m_currentFrame].imageAcquiredSemaphore.get() };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    vk::SubmitInfo submitInfo{};
    submitInfo.setWaitSemaphores(waitSemaphores);
    submitInfo.setWaitDstStageMask(waitStages);
    submitInfo.setCommandBuffers(frames[m_currentFrame].commandBuffer.get());
    
    // Signal the semaphore that presentation will wait on.
    submitInfo.setSignalSemaphores(renderFinishedSemaphores[m_imageIndex].get());

    try {
        context.getQueue().submit(submitInfo, frames[m_currentFrame].inFlightFence.get());
    } catch (vk::DeviceLostError& e) {
       LOG_ERROR("Device lost during submit: " << e.what());
       // A device lost is unrecoverable in this context, but we can try to signal a recreate.
       return true;
    }

    if (!swapchain)
        return true; // Swapchain was destroyed (e.g., minimized), signal recreate.

    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphores(renderFinishedSemaphores[m_imageIndex].get());
    presentInfo.setSwapchains(swapchain.get());
    presentInfo.setImageIndices(m_imageIndex);

    const vk::Result result = context.getQueue().presentKHR(presentInfo);
    
    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    return (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR);
}

//  Compute 
bool Renderer::isComputeWorkFinished() {
    return context.getDevice().getFenceStatus(computeFence.get()) == vk::Result::eSuccess;
}

void Renderer::waitForComputeIdle() {
    (void)context.getDevice().waitForFences(computeFence.get(), VK_TRUE, UINT64_MAX);
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