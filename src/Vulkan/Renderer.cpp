#include "Renderer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "Log.h"

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;

Renderer::Renderer(Context& context, const uint32_t initial_width, const uint32_t initial_height)
    : context(context)
{
    colorImageFormat = vk::Format::eB8G8R8A8Unorm;
    depthImageFormat = vk::Format::eD24UnormS8Uint;
    
    swapchainExtent = vk::Extent2D{ initial_width, initial_height };
    
    frames.resize(MAX_FRAMES_IN_FLIGHT);

    const vk::CommandBufferAllocateInfo cmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);
    auto cmdBuffers = context.getDevice().allocateCommandBuffersUnique(cmdAllocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frames[i].commandBuffer = std::move(cmdBuffers[i]);
        frames[i].imageAcquiredSemaphore = context.getDevice().createSemaphoreUnique({});
        frames[i].renderFinishedSemaphore = context.getDevice().createSemaphoreUnique({});
        frames[i].inFlightFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    }

    // Async Compute
    const vk::CommandBufferAllocateInfo computeCmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
    computeCommandBuffer = std::move(context.getDevice().allocateCommandBuffersUnique(computeCmdAllocInfo).front());
    computeFence = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    computeFinishedSemaphore = context.getDevice().createSemaphoreUnique({});
    computeSubmitted = false;

    recreateSwapChain();
}

Renderer::~Renderer() {
    context.getDevice().waitIdle();
    LOG_INFO("Destroying Renderer");
}

void Renderer::notifyResize(const uint32_t width, const uint32_t height) {
    m_framebufferResized = true;
    if (width > 0 && height > 0)
        swapchainExtent = vk::Extent2D{ width, height };
}

void Renderer::recreateSwapChain() {
    context.getDevice().waitIdle();
    
    const vk::SurfaceCapabilitiesKHR surfaceCapabilities = context.getPhysicalDevice().getSurfaceCapabilitiesKHR(context.getSurface());
    
    // Determine the actual extent to use for the new swapchain.
    vk::Extent2D actualExtent;
    if (surfaceCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
        actualExtent = vk::Extent2D{
            std::clamp(swapchainExtent.width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width),
            std::clamp(swapchainExtent.height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height)
        };
    } else
        actualExtent = surfaceCapabilities.currentExtent;

    // If the window is minimized, the extent will be 0x0.
    // We update our internal extent to reflect this and return.
    // We do NOT attempt to create a 0x0 swapchain. The beginFrame() function
    // will check the extent and skip rendering, preventing a crash.
    if (actualExtent.width == 0 || actualExtent.height == 0) {
        swapchainExtent = actualExtent;
        return;
    }
    
    swapchainExtent = actualExtent;
    
    const vk::SwapchainKHR oldSwapchain = swapchain.release();

    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
        imageCount = surfaceCapabilities.maxImageCount;
    }
    
    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    swapchainInfo.setMinImageCount(imageCount);
    swapchainInfo.setImageFormat(colorImageFormat);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent(actualExtent);
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(surfaceCapabilities.currentTransform);
    swapchainInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
    swapchainInfo.setPresentMode(context.chooseSwapPresentMode());
    swapchainInfo.setClipped(true);
    std::vector queueFamilyIndices{context.getGraphicsFamilyIndex()};
    swapchainInfo.setQueueFamilyIndices(queueFamilyIndices);
    swapchainInfo.setOldSwapchain(oldSwapchain);

    swapchain = context.getDevice().createSwapchainKHRUnique(swapchainInfo);
    if(oldSwapchain) {
        context.getDevice().destroySwapchainKHR(oldSwapchain);
    }
    
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());
    swapchainImageViews.clear();
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {
        vk::ImageViewCreateInfo viewInfo({}, swapchainImages[i], vk::ImageViewType::e2D, colorImageFormat, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        swapchainImageViews[i] = context.getDevice().createImageViewUnique(viewInfo);
    }
    
    depthImage.reset();
    depthImageMemory.reset();
    depthImageView.reset();
    
    const vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, depthImageFormat, { swapchainExtent.width, swapchainExtent.height, 1 }, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined);
    depthImage = context.getDevice().createImageUnique(imageInfo);

    vk::MemoryRequirements memRequirements = context.getDevice().getImageMemoryRequirements(depthImage.get());
    vk::MemoryAllocateInfo allocInfo(memRequirements.size, context.findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
    depthImageMemory = context.getDevice().allocateMemoryUnique(allocInfo);
    context.getDevice().bindImageMemory(depthImage.get(), depthImageMemory.get(), 0);

    const vk::ImageViewCreateInfo viewInfo({}, depthImage.get(), vk::ImageViewType::e2D, depthImageFormat, {}, { vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1 });
    depthImageView = context.getDevice().createImageViewUnique(viewInfo);
        
    m_currentFrame = 0;
    
    LOG_INFO("Recreated swapchain with " << swapchainImages.size() << " images at " << swapchainExtent.width << "x" << swapchainExtent.height);
}
bool Renderer::beginFrame() {
    // Wait for the fence of the current frame to ensure its resources are free to use.
    (void)context.getDevice().waitForFences(frames[m_currentFrame].inFlightFence.get(), true, UINT64_MAX);

    // Handle resize events at the start of the frame
    if (m_framebufferResized) {
        recreateSwapChain();
        // After attempting to recreate, check if the extent is valid.
        // If the window is still minimized, the extent will be 0x0.
        // We must skip rendering, but crucially, we leave m_framebufferResized as true
        // so we will try to recreate again on the next frame.
        if (swapchainExtent.width == 0 || swapchainExtent.height == 0) {
            return false;
        }

        // If we get here, the swapchain was recreated successfully with a valid size.
        // It is now safe to reset the flag.
        m_framebufferResized = false;
    }
    
    // This check is now redundant because of the logic above, but it's harmless to keep.
    if (swapchainExtent.width == 0 || swapchainExtent.height == 0) {
        return false;
    }

    try {
        const vk::ResultValue<uint32_t> result = context.getDevice().acquireNextImageKHR(
            swapchain.get(), 
            UINT64_MAX, 
            frames[m_currentFrame].imageAcquiredSemaphore.get(), 
            nullptr
        );

        if (result.result == vk::Result::eErrorOutOfDateKHR) {
            m_framebufferResized = true; // Signal to recreate on the next frame
            return false;
        }
        
        if (result.result == vk::Result::eSuboptimalKHR) {
            m_framebufferResized = true; // Signal to recreate for better performance
        }
        
        m_imageIndex = result.value;

    } catch (const vk::OutOfDateKHRError&) {
        m_framebufferResized = true;
        return false;
    } catch (const vk::Error& e) {
        LOG_ERROR("Failed to acquire swapchain image: %s");
        return false;
    }

    context.getDevice().resetFences(frames[m_currentFrame].inFlightFence.get());

    const vk::CommandBuffer cmd = getCurrentCommandBuffer();
    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});

    // Transition swapchain image to be a color attachment
    const vk::ImageMemoryBarrier color_barrier(
        {}, vk::AccessFlagBits::eColorAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        swapchainImages[m_imageIndex],
        { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    );
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, 
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, nullptr, nullptr, color_barrier
    );

    // Transition depth image to be a depth/stencil attachment
    const vk::ImageMemoryBarrier depth_barrier(
        {}, vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        depthImage.get(),
        { vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1 }
    );
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
        {}, nullptr, nullptr, depth_barrier
    );
    
    return true;
}

void Renderer::endFrame() { 
    vk::CommandBuffer cmd = getCurrentCommandBuffer();
    cmd.end();

    const vk::Semaphore signalSemaphores[] = { frames[m_currentFrame].renderFinishedSemaphore.get() };
    const vk::Semaphore waitSemaphores[] = { frames[m_currentFrame].imageAcquiredSemaphore.get() };
    constexpr vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    vk::SubmitInfo submitInfo{};
    submitInfo.setWaitSemaphores(waitSemaphores);
    submitInfo.setWaitDstStageMask(waitStages);
    submitInfo.setCommandBuffers(cmd);
    submitInfo.setSignalSemaphores(signalSemaphores);

    context.getGraphicsQueue().submit(submitInfo, frames[m_currentFrame].inFlightFence.get());
   
    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphores(signalSemaphores);
    presentInfo.setSwapchains(swapchain.get());
    presentInfo.setImageIndices(m_imageIndex);

    vk::Result result;
    try {
        result = context.getGraphicsQueue().presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    // FIX: Check if a resize occurred between beginFrame and endFrame in addition to presentation results.
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || m_framebufferResized) {
        m_framebufferResized = true;
    } else if (result != vk::Result::eSuccess)
        LOG_ERROR("Failed to present swap chain image!");

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

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
    
    context.getGraphicsQueue().submit(submitInfo, computeFence.get());
    computeSubmitted = true;
}