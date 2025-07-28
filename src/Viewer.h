#pragma once

#include "Vulkan/Context.h"
#include "Camera/PerspectiveCamera.h"
#include "Cpu/CpuRaytracer.h"
#include "Vulkan/Tonemapper.h"
#include "UI/ImGuiManager.h"
#include "Vulkan/GpuRaytracer.h"
#include "Vulkan/Renderer.h"

class Viewer {
private:
    void setupScene();
    void setupUI();

    int frame = 0;
    const int width;
    const int height;
    
    Context context;
    
    Renderer renderer;
    Scene scene;
    GpuRaytracer gpuRaytracer;
    CpuRaytracer cpuRaytracer;
    InputTracker inputTracker;

    Tonemapper gpuImageTonemapper;
    Tonemapper cpuImageTonemapper;
    ImGuiManager imGuiManager;
    
public:
    Viewer(int width, int height);
    ~Viewer();

    static void cleanup();

    void run();
};