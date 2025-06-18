#pragma once

#include <memory>
#include "ImGuiComponent.h"
#include <string>
#include "Vulkan/Context.h"

class ViewportPanel : public ImGuiComponent {
public:
    ViewportPanel(Context& context, vk::DescriptorPool descriptorPool, vk::ImageView imageView, uint32_t width, uint32_t height);

    void renderUi() override;
    std::string getType() const override { return "Viewport"; }

private:
    uint32_t width, height;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
};