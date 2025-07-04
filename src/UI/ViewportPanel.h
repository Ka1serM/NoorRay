#pragma once

#include "ImGuiComponent.h"
#include <string>
#include "Vulkan/Context.h"

class ViewportPanel : public ImGuiComponent {
public:
    ViewportPanel(Context& context, vk::ImageView imageView, uint32_t width, uint32_t height);

    ~ViewportPanel(); // Ensure it's declared

    // --- RULE OF FIVE ---
    // Forbid copying and moving to ensure single ownership of the descriptor set.
    // This will turn any accidental copies into a compile-time error.
    ViewportPanel(const ViewportPanel&) = delete;
    ViewportPanel& operator=(const ViewportPanel&) = delete;
    ViewportPanel(ViewportPanel&&) = delete;
    ViewportPanel& operator=(ViewportPanel&&) = delete;


    void renderUi() override;
    std::string getType() const override { return "Viewport"; }

private:
    uint32_t width, height;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
};