#pragma once

#include <cfloat>
#include <imgui.h>
#include <memory>

#include "Context.h"
#include "Image.h"
#include "glm/vec3.hpp"
#include "glm/gtc/type_ptr.hpp"

class ImGuiOverlay {
public:
    ImGuiOverlay(std::shared_ptr<Context> context, const std::vector<vk::Image> &swapchainImages, Image &outputImage);

    void Render(vk::CommandBuffer commandBuffer, uint32_t imageIndex);

    static void tableRowLabel(const char *label);

    template <typename Setter>
    static void dragFloat3Row(const char *label, glm::vec3 value, float speed, Setter setter) {
        tableRowLabel(label);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::DragFloat3((std::string("##") + label).c_str(), value_ptr(value), speed))
            setter(value);
    }

    vk::UniqueDescriptorSet outputImageDescriptorSet;

    ~ImGuiOverlay();
private:
    void CreateDescriptorPool();

    void CreateRenderPass();

    void CreateFrameBuffers(const std::vector<vk::Image> &images);


    void CreateOutputImageDescriptor(Image &outputImage);

    void SetBlenderTheme();

    std::shared_ptr<Context> context;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueFramebuffer> frameBuffers;
    std::vector<vk::UniqueImageView> imageViews;
    vk::UniqueRenderPass renderPass;
private:
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueSampler sampler;
};
