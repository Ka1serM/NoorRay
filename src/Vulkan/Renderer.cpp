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
    
    if (width > 0 && height > 0) {
        swapchainExtent = vk::Extent2D{ width, height };
    }
}

// You may need these headers for std::clamp and std::numeric_limits
#include <algorithm>
#include <limits>

void Renderer::recreateSwapChain() {
    // Wait until the device is idle before tearing down resources.
    context.getDevice().waitIdle();
    
    // 1. Get the LATEST surface capabilities directly from the physical device.
    // This is crucial as they can change (e.g., when the window is minimized).
    const vk::SurfaceCapabilitiesKHR surfaceCapabilities = context.getPhysicalDevice().getSurfaceCapabilitiesKHR(context.getSurface());

    // 2. Handle minimization explicitly. If the surface has no size, we cannot create a
    // swapchain for it. We abort here, and the m_framebufferResized flag will remain
    // true, causing us to try again on the next frame.
    if (surfaceCapabilities.currentExtent.width == 0 || surfaceCapabilities.currentExtent.height == 0) {
        return;
    }

    // 3. Determine the actual extent to use for the new swapchain.
    vk::Extent2D actualExtent;
    // The special value UINT32_MAX means the window manager allows us to choose an extent.
    // Otherwise, we MUST use the extent provided in the capabilities.
    if (surfaceCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
        actualExtent = vk::Extent2D{
            std::clamp(swapchainExtent.width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width),
            std::clamp(swapchainExtent.height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height)
        };
    } else
        actualExtent = surfaceCapabilities.currentExtent;

    // 4. Update the member variable. This is now the new canonical size for our renderer.
    swapchainExtent = actualExtent;
    
    // Release the old swapchain handle to prepare for creating a new one.
    const vk::SwapchainKHR oldSwapchain = swapchain.release();

    // Choose the number of images in the swapchain.
    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
        imageCount = surfaceCapabilities.maxImageCount;
    }
    
    // Create the new swapchain.
    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    swapchainInfo.setMinImageCount(imageCount);
    swapchainInfo.setImageFormat(colorImageFormat);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent(actualExtent); // ✔️ Use the validated, current extent
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
    
    // Recreate swapchain image views.
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());
    swapchainImageViews.clear();
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {
        vk::ImageViewCreateInfo viewInfo({}, swapchainImages[i], vk::ImageViewType::e2D, colorImageFormat, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        swapchainImageViews[i] = context.getDevice().createImageViewUnique(viewInfo);
    }
    
    // Recreate depth buffer resources matching the new swapchain size.
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
    if (m_framebufferResized) {
        recreateSwapChain();
        m_framebufferResized = false;
        return false;
    }
    
    (void)context.getDevice().waitForFences(frames[m_currentFrame].inFlightFence.get(), true, UINT64_MAX);
    context.getDevice().resetFences(frames[m_currentFrame].inFlightFence.get());

    try {
        const vk::ResultValue<uint32_t> result = context.getDevice().acquireNextImageKHR(
            swapchain.get(), 
            UINT64_MAX, 
            frames[m_currentFrame].imageAcquiredSemaphore.get(), 
            nullptr
        );

        // 2. Handle non-throwing error codes also inside the try block.
        if (result.result == vk::Result::eErrorOutOfDateKHR || result.result == vk::Result::eSuboptimalKHR) {
            m_framebufferResized = true;
            return false;
        }
        
        // 3. If successful, assign the image index.
        m_imageIndex = result.value;
    }
    catch (const vk::OutOfDateKHRError&) {
        m_framebufferResized = true;
        return false;
    }

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

    // The submission logic remains the same
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
    }
    catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
        m_framebufferResized = true; // Just flag for resize, don't recreate here.

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