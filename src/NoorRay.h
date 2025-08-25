#pragma once
#include "UI/ImGuiManager.h"
#include <memory>
#include "Scene/Scene.h"
#include "Vulkan/Renderer.h"

class GpuRaytracer;
class Tonemapper;

class NoorRay
{
    Context context;
    Renderer renderer;
    ImGuiManager imGuiManager;
    Scene scene;

    std::unique_ptr<GpuRaytracer> raytracer;
    std::unique_ptr<Tonemapper> tonemapper;

    void setupUI();
    void setupScene();

public:
    
    NoorRay(int windowWidth, int windowHeight, int renderWidth, int renderHeight);
    ~NoorRay();
    void run();
};
