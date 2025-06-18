#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <atomic>

#include "Vulkan/Context.h"
#include "Vulkan/Image.h"
#include "Camera/PerspectiveCamera.h"
#include "Vulkan/HdrToLdrCompute.h"
#include "UI/ImGuiManager.h"
#include "Vulkan/Renderer.h"
#include <Windows.h>

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
};

class Viewer {
public:
    Viewer(int width, int height);
    ~Viewer();

    void start();
    void stop();

    void runOnMainThread(std::function<void()> task);
    
    void addMesh(const MeshData& mesh);

    HWND getHwnd() const;

private:
    void setupScene();
    void setupUI();

    std::queue<std::function<void()>> mainThreadQueue;
    std::mutex mainThreadQueueMutex;

    std::atomic<bool> running{ false };
    
    const int width;
    const int height;

    Context context;
    Renderer renderer;
    Image inputImage;
    Image outputImage;
    
    InputTracker inputTracker;
    HdrToLdrCompute hdrToLdrCompute;
    ImGuiManager imGuiManager;

    bool isRayTracing = false;
    int frame = 0;
};
