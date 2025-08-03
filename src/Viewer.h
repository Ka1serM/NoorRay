#pragma once
#include "UI/ImGuiManager.h"
#include <memory>
#include "Camera/InputTracker.h"
#include "Scene/Scene.h"
#include "Vulkan/Renderer.h"

class Raytracer;
class Tonemapper;

class Viewer
{
public:
    Viewer(int width, int height);
    ~Viewer();
    void run();

private:
    void setupUI();
    void setupScene();
    
    void recreateSwapChain();
    bool framebufferResized = false;

    int width;
    int height;
    
    Context context;
    Renderer renderer;
    ImGuiManager imGuiManager;
    Scene scene;
    InputTracker inputTracker;

    std::unique_ptr<Raytracer> raytracer;
    std::unique_ptr<Tonemapper> gpuImageTonemapper;
    
    int frame = 0;
};
