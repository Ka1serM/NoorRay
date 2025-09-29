#include "ViewportViewModel.h"
#include <iostream>
#include "UI/ImGui/DebugPanel.h"
#include "UI/ImGui/RenderPanel.h"
#include "UI/ImGui/ViewportPanel.h"
#include "Vulkan/Context.h"

ViewportViewModel::ViewportViewModel(Context& appContext, Rml::Context* rmlContext, RmlRenderInterface& renderInterface, const Image& renderImage, const std::string& elementId)
    : ViewModelBase(rmlContext, elementId),
    renderInterface(renderInterface),
    imGuiManager(appContext, 2, vk::Format::eR8G8B8A8Unorm),
    imGuiImage(appContext, renderImage.getWidth(), renderImage.getHeight(), renderImage.getFormat(),vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled)
{
    imGuiManager.addComponent<DebugPanel>("Debug");
    //imGuiManager.addComponent<EnvironmentPanel>("Environment", scene);
    //imGuiManager.addComponent<SceneGraphPanel>("Scene Graph", scene);
    //imGuiManager.addComponent<DetailsPanel>("Details", scene);
    //imGuiManager.addComponent<RenderPanel>("Render", context, *raytracer, renderer, *tonemapper);
   // imGuiManager.addComponent<ViewportPanel>("Viewport", context, scene, tonemapper->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(), raytracer->getWidth(), raytracer->getHeight());

    const Rml::String elementName = "viewport-img";
    Rml::ElementDocument* doc = rmlContext->GetDocument("main");

    imGuiElement = doc->GetElementById(elementName);
    
    //renderInterface.registerVulkanTexture(elementName, imGuiImage.getView(), Rml::Vector2i{ static_cast<int>(imGuiImage.getWidth()), static_cast<int>(imGuiImage.getHeight()) });
    //imGuiElement->SetAttribute("src", "vulkan://" + elementName);

    renderInterface.registerVulkanTexture(elementId, renderImage.getView(), Rml::Vector2i{ static_cast<int>(renderImage.getWidth()), static_cast<int>(renderImage.getHeight()) });
    doc->GetElementById(elementId)->SetAttribute("src", "vulkan://" + elementId);
}

void ViewportViewModel::Update() {
    // Get RmlUi values
    const Rml::Vector2f pos = imGuiElement->GetAbsoluteOffset();
    const Rml::Vector2f size = imGuiElement->GetBox().GetSize();

    // Convert to Vulkan types
    const vk::Offset2D newPos{ static_cast<int32_t>(pos.x), static_cast<int32_t>(pos.y) };
    const vk::Extent2D newSize{static_cast<uint32_t>(size.x),static_cast<uint32_t>(size.y)};
    viewportPos = newPos;
    viewportSize = newSize;
}

void ViewportViewModel::Render(const vk::CommandBuffer cmd, const vk::Extent2D windowSize) {
    // Render ImGui to the offscreen image
    //imGuiManager.renderToTarget(cmd, imGuiImage.getView(), vk::Extent2D{imGuiImage.getWidth(), imGuiImage.getHeight()}, viewportPos, viewportSize, windowSize);
}


void ViewportViewModel::processEvent(SDL_Event& event) {
    // Forward event to ImGui first
    //imGuiManager.processEvent(event);

    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        int x = event.button.x;
        int y = event.button.y;

        if (x >= viewportPos.x && x <= viewportPos.x + static_cast<int32_t>(viewportSize.width) &&
            y >= viewportPos.y && y <= viewportPos.y + static_cast<int32_t>(viewportSize.height)) {
            std::cout << "Click inside viewport!" << std::endl;
            }
    }
}