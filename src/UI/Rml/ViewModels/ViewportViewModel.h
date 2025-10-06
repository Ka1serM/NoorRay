#pragma once
#include "ViewModelBase.h"
#include "Vulkan/Image.h"
#include <RmlUi/Core.h>

#include "imgui.h"
#include "Scene/Scene.h"
#include "UI/Rml/Interfaces/RmlRenderInterface.h"
#include "Vulkan/Buffer.h"
#include <string>
#include "ImGuizmo.h"
#include "glm/glm.hpp"
#include "SDL3/SDL_events.h"
#include "Vulkan/Context.h"

class ViewportViewModel : public ViewModelBase {
public:
    ViewportViewModel(Context& appContext, Scene& scene, Rml::Context* rmlContext, RmlRenderInterface& renderInterface, Image& renderImage, Image& cryptoImage, Image& positionImage, const std::string& elementId);
    void render(vk::CommandBuffer commandBuffer);
    void renderViewGizmo();
    void processEvent(const SDL_Event& event);
    ~ViewportViewModel() override;
private:
    void renderToolbar();
    void renderTransformGizmo();

    Rml::Context* rmlUIContext;
    RmlRenderInterface& renderInterface;
    Context& context;
    Scene& scene;
    Image& cryptoImage;
    Image& positionImage;

    Buffer cryptoStagingBuffer;
    void* cryptoStagingBufferMappedPtr = nullptr;
    
    Buffer positionStagingBuffer;
    void* positionStagingBufferMappedPtr = nullptr;
    
    Image imGuiImage;

    ImVec2 lastMousePosInsideViewport{ -FLT_MAX, -FLT_MAX };

    float uiScale{1.0f};
    
    ImGuizmo::OPERATION currentOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE currentMode = ImGuizmo::LOCAL;

    ImVec2 viewportPos{0,0}; 
    ImVec2 viewportSize{0,0}; 
    vk::Extent2D windowSize{0,0};
    
    Rml::Element* imGuiElement = nullptr;

    void handleObjectPicking(const ImVec2& mousePos);
    void handlePositionPicking(const ImVec2& mousePos);
    void applyImGuiTheme();

    bool isMouseOverViewport() const;
    void addMouseInput(ImGuiIO& io);
};
