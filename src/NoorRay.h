#pragma once
#include "UI/ImGuiManager.h"
#include <memory>
#include "Scene/Scene.h"
#include "Vulkan/Renderer.h"

class GpuRaytracer;
class Tonemapper;

class NoorRay
{
public:
    NoorRay(int windowWidth, int windowHeight, int renderWidth, int renderHeight);
    void recreateSwapChain();
    ~NoorRay();
    void run();

private:
    void setupUI();
    void setupScene();
    
    int windowWidth;
    int windowHeight;
    
    Context context;
    Renderer renderer;
    ImGuiManager imGuiManager;
    Scene scene;

    std::unique_ptr<GpuRaytracer> raytracer;
    std::unique_ptr<Tonemapper> tonemapper;
};
