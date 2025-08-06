#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include "../UI/ImGuiComponent.h"
#include <memory>
#include <string>
#include "imgui.h"
#include "ImGuizmo.h"
#include "Scene/Scene.h"

struct ImVec2;

class ViewportPanel : public ImGuiComponent {
public:
    ViewportPanel(Scene& scene, const Image& inputImage, uint32_t width, uint32_t height, const std::string& title);

    void renderUi() override;
    ~ViewportPanel();
    
    void recordCopy(vk::CommandBuffer cmd, Image& srcImage);
    void handleInput(bool isImageHovered);

    std::string getType() const override { return title; }


private:
    Scene& scene;
    uint32_t width;
    uint32_t height;
    std::string title;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
    
    Image displayImage;

    bool isCapturingMouse = false;
    float oldX = 0.f, oldY = 0.f;
    ImGuizmo::OPERATION currentOperation = ImGuizmo::TRANSLATE;
};
