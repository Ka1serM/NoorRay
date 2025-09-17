#pragma once
#include "UI/ImGui/ImGuiManager.h"
#include <memory>
#include "Scene/Scene.h"
#include "UI/Rml/RmlUiManager.h"
#include "Vulkan/Renderer.h"

class GpuRaytracer;
class Tonemapper;

class NoorRay
{
    Context context;
    Renderer renderer;
    ImGuiManager imGuiManager;
    RmlUiManager rmlUiManager;
    Scene scene;

    std::unique_ptr<GpuRaytracer> raytracer;
    std::unique_ptr<Tonemapper> tonemapper;
public:
    
    NoorRay(int windowWidth, int windowHeight, int renderWidth, int renderHeight);
    ~NoorRay();
    void run();
};
