#pragma once

#include "Vulkan/Context.h"
#include "Vulkan/Image.h"
#include "UI/ImGuiComponent.h"
#include <string>
#include "imgui.h" //needed here for ImGuizmo!
#include "ImGuizmo.h"
#include "Scene/Scene.h"
#include "Vulkan/Buffer.h"
#include <SDL3/SDL_events.h>

class Window;

class ViewportPanel : public ImGuiComponent {
public:
    ivec2 screenToPixel() const;
    void renderUi() override;
    void handleObjectPicking();
    bool handleBillboardPicking() const;
    void handlePositionPicking() const;
    void onComputeFinished(vk::CommandBuffer cmd, Image& srcImage);
    void setAovImages(Image& crypto, Image& position);
    void resize(uint32_t width, uint32_t height, vk::Format imageFormat);
    // Returns true when the viewport owns this event and it must not be sent to ImGui.
    bool processEvent(const SDL_Event& event);
    ~ViewportPanel() override;

    bool showOverlays() const { return m_showOverlays; }
    bool needsContinuousRedraw() const { return isCapturingMouse; }
    uint32_t getSelectedGaussianIndex() const { return selectedGaussianIndex; }

    ViewportPanel(const std::string& name, Window& window, Context& context, Scene& scene,
        const Image& outputColor, Image& outputCrypto, Image& outputPosition,
        uint32_t width, uint32_t height);
    void updateLayout();

private:
    Window& window;
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
    bool imguiMouseWasDisabled = false;
    bool rightButtonDown = false;
    bool rightButtonPressPending = false;
    float rightButtonPressX = 0.f;
    float rightButtonPressY = 0.f;
    float pendingMouseDeltaX = 0.f;
    float pendingMouseDeltaY = 0.f;
    uint64_t observedCameraRevision = 0;
    float oldX = 0.f, oldY = 0.f;

    ImVec2 viewportPos{};   // Top-left corner of the viewport image on the screen
    ImVec2 viewportSize{};
    
    bool isViewportHovered{false};
    bool m_showOverlays = true;
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
    void synchronizeCameraTransition();
    
    ImGuizmo::OPERATION currentOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE currentMode = ImGuizmo::LOCAL;
    uint32_t selectedGaussianIndex = ~0u;
};
