#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "NoorRaySession.h"
#include "UI/Window.h"

class ImGuiManager;
class Viewport;

class NoorRayUi
{
public:
    NoorRayUi(std::string scenePath = {}, uint32_t windowWidth = 0, uint32_t windowHeight = 0);
    ~NoorRayUi();

    void run();

private:
    Window window;
    noorray::NoorRaySession session;
    std::unique_ptr<Viewport> viewport;
    std::unique_ptr<ImGuiManager> imGuiManager;
};
