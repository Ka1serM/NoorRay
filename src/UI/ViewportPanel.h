#pragma once

#include <memory>
#include "ImGuiComponent.h"
#include <string>
#include "Vulkan/Context.h"

class ViewportPanel : public ImGuiComponent {
public:
    ViewportPanel(const std::shared_ptr<Context>& context, vk::DescriptorPool descriptorPool, vk::ImageView imageView);
    virtual ~ViewportPanel() = default;

    void renderUi() override;
    std::string getType() const override { return "Viewport"; }

private:
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
};