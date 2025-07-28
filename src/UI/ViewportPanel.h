#pragma once

#include "ImGuiComponent.h"
#include <string>
#include "Vulkan/Context.h"
#include "Vulkan/Image.h"

class ViewportPanel : public ImGuiComponent {
public:
    ~ViewportPanel();


    ViewportPanel(Context& context, const Image& inputImage, uint32_t width, uint32_t height, const std::string& title);
    void renderUi() override;
    std::string getType() const override { return "Viewport"; }

private:
    uint32_t width, height;
    std::string title;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
};
