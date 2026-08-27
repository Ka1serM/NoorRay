#pragma once

#include <gpu/interop.hpp>
#include <vulkan/vulkan.hpp>
#include "UI/ImGuiComponent.h"
#include <memory>
#include <string>
#include "imgui.h" //needed here for ImGuizmo!
#include "ImGuizmo.h"
#include "Scene/Scene.h"
#include <SDL3/SDL_events.h>

class Window;
class VulkanRaytracer;
class Viewport;

class ViewportPanel : public ImGuiComponent {
public:
    ivec2 screenToPixel() const;
    void renderUi() override;
    void handleObjectPicking();
    bool handleBillboardPicking() const;
    void handlePositionPicking() const;
    // Returns true when the viewport owns this event and it must not be sent to ImGui.
    bool processEvent(const SDL_Event& event);
    ~ViewportPanel() override;

    bool showOverlays() const { return m_showOverlays; }
    bool needsContinuousRedraw() const { return isCapturingMouse; }
    uint32_t getSelectedGaussianIndex() const { return selectedGaussianIndex; }

    ViewportPanel(const std::string& name, Window& window, Scene& scene,
        VulkanRaytracer& raytracer);
    void updateLayout();
    void recordPresentation();

private:
    Window& window;
    Scene& scene;
    VulkanRaytracer& raytracer;
    std::unique_ptr<Viewport> compositor;
    
    uint32_t width;
    uint32_t height;

    vk::UniqueSampler sampler;
    VkDescriptorSet outputImageDescriptorSet{};
    vk::ImageView observedImageView{};
    
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
