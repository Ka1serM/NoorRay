#include "Renderer.h"
#include <iostream>

Renderer::Renderer(Context& context, uint32_t width, uint32_t height)
: context(context), width(width), height(height)
{
    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({width, height});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eMailbox);
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context.getQueueFamilyIndices());

    swapchain = context.getDevice().createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = context.getDevice().getSwapchainImagesKHR(swapchain.get());

    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(context.getCommandPool());
    commandBufferInfo.setCommandBufferCount(static_cast<uint32_t>(swapchainImages.size()));
    commandBuffers = context.getDevice().allocateCommandBuffersUnique(commandBufferInfo);

    m_imageAcquiredSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_imageAcquiredSemaphores[i] = context.getDevice().createSemaphoreUnique({});
        m_renderFinishedSemaphores[i] = context.getDevice().createSemaphoreUnique({});
        m_inFlightFences[i] = context.getDevice().createFenceUnique({ vk::FenceCreateFlagBits::eSignaled });
    }
}

Renderer::~Renderer()
{
    std::cout << "Destroying Renderer" << std::endl;
}

const vk::CommandBuffer& Renderer::getCommandBuffer(const uint32_t imageIndex) const
{
    return commandBuffers[imageIndex].get();
}

void Renderer::advanceFrame() 
{
    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}