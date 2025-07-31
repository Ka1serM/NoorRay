#include "Renderer.h"
#include <iostream>

Renderer::Renderer(Context& context, uint32_t width, uint32_t height)
: context(context), width(width), height(height)
{
    // --- Create Swapchain ---
    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    // Let's request 3 images for triple buffering
    swapchainInfo.setMinImageCount(3);
    auto swapSurfaceFormat = context.chooseSwapSurfaceFormat();
    swapchainInfo.setImageFormat(swapSurfaceFormat.format);
    swapchainInfo.setImageColorSpace(swapSurfaceFormat.colorSpace);
    swapchainInfo.setImageExtent({width, height});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    swapchainInfo.setPresentMode(context.chooseSwapPresentMode());
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context.getQueueFamilyIndices());

    swapchain = context.getDevice().createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());
    
    // --- Create one set of sync objects and a command buffer for each swapchain image ---
    size_t imageCount = swapchainImages.size();
    
    vk::CommandBufferAllocateInfo cmdAllocInfo(context.getCommandPool(), vk::CommandBufferLevel::ePrimary, imageCount);
    commandBuffers = context.getDevice().allocateCommandBuffersUnique(cmdAllocInfo);

    m_imageAcquiredSemaphores.resize(imageCount);
    m_renderFinishedSemaphores.resize(imageCount);
    m_inFlightFences.resize(imageCount);

    for (size_t i = 0; i < imageCount; ++i) {
        m_imageAcquiredSemaphores[i] = context.getDevice().createSemaphoreUnique({});
        m_renderFinishedSemaphores[i] = context.getDevice().createSemaphoreUnique({});
        // Fences are created in a signaled state so the first frame doesn't wait
        m_inFlightFences[i] = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    }
}

Renderer::~Renderer()
{
    context.getDevice().waitIdle();
    std::cout << "Destroying Renderer" << std::endl;
}

vk::CommandBuffer Renderer::beginFrame()
{
    // 1. Acquire an image from the swapchain.
    // The m_imageIndex we get back tells us which set of resources to use.
    try {
        auto result = context.getDevice().acquireNextImageKHR(swapchain.get(), UINT64_MAX, m_imageAcquiredSemaphores[m_imageIndex].get(), nullptr);
        m_imageIndex = result.value;
    } catch (const vk::OutOfDateKHRError& e) {
        // Window was resized. Returning nullptr signals the main loop to recreate the swapchain.
        return nullptr;
    }
    
    // 2. Wait for the fence associated with THIS image.
    // This ensures that the last time this specific image was used, the work has finished.
    // This is much more direct than the old method.
    (void)context.getDevice().waitForFences(m_inFlightFences[m_imageIndex].get(), true, UINT64_MAX);
    context.getDevice().resetFences(m_inFlightFences[m_imageIndex].get());

    // 3. Begin the command buffer for this image.
    const auto& cmd = commandBuffers[m_imageIndex];
    cmd->reset();
    cmd->begin(vk::CommandBufferBeginInfo{});
    
    return cmd.get();
}

void Renderer::endFrame()
{
    const auto& cmd = commandBuffers[m_imageIndex];
    cmd->end();

    // --- Submit the command buffer ---
    vk::SubmitInfo submitInfo{};
    vk::Semaphore waitSemaphores[] = { m_imageAcquiredSemaphores[m_imageIndex].get() };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    submitInfo.setWaitSemaphores(waitSemaphores);
    submitInfo.setWaitDstStageMask(waitStages);
    submitInfo.setCommandBuffers(cmd.get());
    
    vk::Semaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_imageIndex].get() };
    submitInfo.setSignalSemaphores(signalSemaphores);
    
    // This submission will signal the fence for this image when it's done.
    context.getQueue().submit(submitInfo, m_inFlightFences[m_imageIndex].get());

    // --- Present the image ---
    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphores(signalSemaphores);
    presentInfo.setSwapchains(swapchain.get());
    presentInfo.setImageIndices(m_imageIndex);
    
    try {
        (void)context.getQueue().presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError& e) {
        // Swapchain out of date. The next call to beginFrame() will handle it.
    }
}