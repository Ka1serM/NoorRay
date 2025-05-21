#pragma once

#include "Context.h"

class ImGuiOverlay {
public:
    ImGuiOverlay(Context &context, const std::vector<vk::Image> &swapchainImages);

    void Render(vk::CommandBuffer commandBuffer, uint32_t imageIndex);

    ~ImGuiOverlay();
private:
    void CreateDescriptorPool();

    void CreateRenderPass();

    void CreateFrameBuffers(const std::vector<vk::Image> &images);

    void SetDarkTheme();

    Context& context;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueFramebuffer> frameBuffers;
    std::vector<vk::UniqueImageView> imageViews;
    vk::UniqueRenderPass renderPass;
};
