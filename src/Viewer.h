#pragma once

#include "Vulkan/Context.h"
#include "Camera/PerspectiveCamera.h"
#include "Vulkan/Tonemapper.h"
#include "UI/ImGuiManager.h"
#include "Vulkan/Renderer.h"

class Viewer {
public:
    Viewer(int width, int height);
    // Forbid copying and moving to prevent resource management bugs
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;
    Viewer(Viewer&&) = delete;
    Viewer& operator=(Viewer&&) = delete;

    static void cleanup();

    void run();

private:
    void setupScene();
    void setupUI();

    int frame = 0;
    const int width;
    const int height;
    
    Context context;
    
    Renderer renderer;
    InputTracker inputTracker;

    Tonemapper tonemapper;
    ImGuiManager imGuiManager;
};