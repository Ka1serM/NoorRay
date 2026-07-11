#pragma once

#include "../Vulkan/Context.h"
#include "../Vulkan/Image.h"
#include "../UI/ImGuiComponent.h"
#include <string>
#include "imgui.h" //needed here for ImGuizmo!
#include "ImGuizmo.h"
#include "Scene/Scene.h"
#include "Vulkan/Buffer.h"

class ViewportPanel : public ImGuiComponent {
public:
    ivec2 screenToPixel() const;
    void renderUi() override;
    void handleObjectPicking() const;
    bool handleBillboardPicking() const;
    void handlePositionPicking() const;
    void onComputeFinished(vk::CommandBuffer cmd, Image& srcImage);
    void setAovImages(Image& crypto, Image& position);
    void resize(uint32_t width, uint32_t height, vk::Format imageFormat);
    ~ViewportPanel() override;

    ViewportPanel(const std::string& name, Context& context, Scene& scene, const Image& outputColor, Image& outputCrypto, Image& outputPosition, uint32_t width, uint32_t height);
    void updateLayout();

private:
    Context& context;
    Scene& scene;
    Image* outputCrypto;
    Image* outputPosition;
    
    uint32_t width;
    uint32_t height;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    vk::UniqueDescriptorSet outputImageDescriptorSet;

    Image displayImage;
    
    Buffer cryptoStagingBuffer; // Staging buffer for picking
    void* cryptoStagingBufferMappedPtr = nullptr;
    bool  pickingRequested = false;

    Buffer positionStagingBuffer; // Staging buffer for picking
    void* positionStagingBufferMappedPtr = nullptr;
    
    bool isCapturingMouse = false;
    float oldX = 0.f, oldY = 0.f;

    ImVec2 viewportPos{};   // Top-left corner of the viewport image on the screen
    ImVec2 viewportSize{};
    
    bool isViewportHovered{false};
    float uiScale{1.0f};
    // Precomputed baseGizmoSizeClipSpace * uiScale * referenceViewportWidth; only viewportSize.x varies per frame.
    float gizmoSizeClipSpaceScale{1.0f};

    void drawBackground() const;
    void drawImageAndUpdateState();
    void updateDisplayDescriptor();

    void handleInput();
    void handleScrollZoom();
    void handleTransformGizmo();
    void handleViewGizmo() const;
    void renderToolbar();

    void beginMouseCapture();
    void endMouseCapture();
    
    ImGuizmo::OPERATION currentOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE currentMode = ImGuizmo::LOCAL;
};
