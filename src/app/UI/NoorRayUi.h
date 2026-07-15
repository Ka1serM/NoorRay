#pragma once

#include <memory>
#include <string>

#include "NoorRaySession.h"
#include "UI/Window.h"

class ImGuiManager;
class Viewport;

class NoorRayUi
{
public:
    explicit NoorRayUi(std::string scenePath = {});
    ~NoorRayUi();

    void run();

private:
    Window window;
    noorray::NoorRaySession session;
    std::unique_ptr<Viewport> viewport;
    std::unique_ptr<ImGuiManager> imGuiManager;
};
