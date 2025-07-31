#pragma once

#include "../Vulkan/Context.h"
#include <vector>

class Renderer {
public:
    Renderer(Context& context, uint32_t width, uint32_t height);
    ~Renderer();

    vk::CommandBuffer beginFrame();
    void endFrame();

    const std::vector<vk::Image>& getSwapchainImages() const { return swapchainImages; }
    uint32_t getCurrentIndex() const { return m_imageIndex; }

private:
    Context& context;
    uint32_t width;
    uint32_t height;
    uint32_t m_imageIndex = 0;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;

    std::vector<vk::UniqueCommandBuffer> commandBuffers;
    std::vector<vk::UniqueSemaphore> m_imageAcquiredSemaphores;
    std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphores;
    std::vector<vk::UniqueFence> m_inFlightFences;
};