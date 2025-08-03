#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include "../UI/ImGuiComponent.h"
#include <memory>
#include <string>

class ViewportPanel : public ImGuiComponent {
public:
    ViewportPanel(Context& context, const Image& inputImage, uint32_t width, uint32_t height, const std::string& title);
    ~ViewportPanel() override;

    void renderUi() override;
    
    void recordCopy(vk::CommandBuffer cmd, Image& srcImage);

    std::string getType() const override { return title; }

private:
    uint32_t width;
    uint32_t height;
    std::string title;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
    
    Image displayImage;
};
