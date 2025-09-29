#pragma once
#include "ViewModelBase.h"
#include "Vulkan/Image.h"
#include <RmlUi/Core.h>

#include "UI/ImGui/ImGuiManager.h"
#include "UI/Rml/Interfaces/RmlRenderInterface.h"

class ViewportViewModel : public ViewModelBase {
public:
    ViewportViewModel(Context& appContext, Rml::Context* rmlContext, RmlRenderInterface& renderInterface, const Image& renderImage, const std::string& elementId);
    void Update() override;
    void Render(vk::CommandBuffer cmd, vk::Extent2D windowSize);
    void processEvent(SDL_Event& event);

private:
    RmlRenderInterface& renderInterface; // reference first
    ImGuiManager imGuiManager;
    Image imGuiImage;

    vk::Offset2D viewportPos{0,0};      // Position of viewport in screen coordinates
    vk::Extent2D viewportSize{0,0};     // Size of the viewport
    Rml::Element* imGuiElement = nullptr;
};