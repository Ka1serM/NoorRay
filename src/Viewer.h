#pragma once
#include "UI/ImGuiManager.h"
#include <memory>
#include "Scene/Scene.h"
#include "Vulkan/Renderer.h"

class GpuRaytracer;
class Tonemapper;

class Viewer
{
public:
    Viewer(int width, int height);
    void recreateSwapChain();
    ~Viewer();
    void run();

private:
    void setupUI();
    void setupScene();
    
    bool resized = false;

    int windowWidth;
    int windowHeight;
    int renderWidth;
    int renderHeight;
    
    Context context;
    Renderer renderer;
    ImGuiManager imGuiManager;
    Scene scene;

    std::unique_ptr<GpuRaytracer> raytracer;
    std::unique_ptr<Tonemapper> gpuImageTonemapper;
    
    int frame = 0;
};
