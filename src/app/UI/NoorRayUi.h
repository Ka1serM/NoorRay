#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "NoorRaySession.h"
#include "UI/Window.h"

class ImGuiManager;
class DebugPanel;
class ViewportPanel;

class NoorRayUi
{
public:
    NoorRayUi(std::string scenePath = {}, uint32_t windowWidth = 0, uint32_t windowHeight = 0);
    ~NoorRayUi();

    void run();

private:
    Window window;
    // The UI owns presentation. The session renders on this device into
    // textures and never touches the swapchain.
    noorrhi::Device device;
    noorrhi::Swapchain swapchain;
    noorray::NoorRaySession session;
    std::unique_ptr<ImGuiManager> imGuiManager;
    DebugPanel* debugPanel{};
    ViewportPanel* viewportPanel{};
};
