#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include "../UI/ImGuiComponent.h"
#include <memory>
#include <string>
#include "imgui.h" //needed here for ImGuizmo!
#include "ImGuizmo.h"
#include "Scene/Scene.h"
#include "Vulkan/Buffer.h"

class ViewportPanel : public ImGuiComponent {
public:

    void renderUi() override;
    void handlePicking(ImVec2 imageSize) const;
    ~ViewportPanel() override;

    ViewportPanel(Scene& scene, const Image& outputColor, Image& outputCrypto, uint32_t width, uint32_t height, const std::string& title);
    void recordCopy(vk::CommandBuffer cmd, Image& srcImage);

    std::string getType() const override { return title; }


private:
    Scene& scene;
    Image& outputCrypto;
    
    uint32_t width;
    uint32_t height;
    std::string title;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;
    
    Image displayImage;

    Buffer pickingBuffer; // Staging buffer for picking
    void* pickingBufferMappedPtr = nullptr;

    bool isCapturingMouse = false;
    float oldX = 0.f, oldY = 0.f;
    ImGuizmo::OPERATION currentOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE currentMode = ImGuizmo::LOCAL;
};
