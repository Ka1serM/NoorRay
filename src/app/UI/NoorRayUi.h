#pragma once

#include <memory>

#include "NoorRaySession.h"
#include "UI/Window.h"

class ImGuiManager;
class Viewport;

class NoorRayUi
{
public:
    NoorRayUi();
    ~NoorRayUi();

    void run();

private:
    Window window;
    noorray::NoorRaySession session;
    std::unique_ptr<Viewport> viewport;
    std::unique_ptr<ImGuiManager> imGuiManager;
};
