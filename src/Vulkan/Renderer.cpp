#include "Renderer.h"
#include <iostream>
#include "Utils.h"

Renderer::Renderer(Context& context, uint32_t width, uint32_t height)
: width(width), height(height), context(context)
{
    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.getSurface());
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({width, height});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst |vk::ImageUsageFlagBits::eColorAttachment);
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
}

Renderer::~Renderer()
{
    std::cout << "Destroying Renderer" << std::endl;
}

const vk::CommandBuffer& Renderer::getCommandBuffer(const uint32_t imageIndex) const
{
    return commandBuffers[imageIndex].get();
}