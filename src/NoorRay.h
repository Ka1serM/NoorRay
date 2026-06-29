#pragma once
#include <memory>
#include <string>
#include "Vulkan/Context.h"
#include "Scene/Scene.h"

class Renderer;
class ImGuiManager;
class Raytracer;
class Viewport;

class NoorRay {
    Context context;
    Scene scene;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ImGuiManager> imGuiManager;
    std::unique_ptr<Raytracer> raytracer;
    std::unique_ptr<Viewport> viewport;

public:
    NoorRay(int windowWidth, int windowHeight);
    NoorRay(int argc, char* argv[]);   // headless: argv[1] = scene.json
    ~NoorRay();

    void runUi();
    void runCli(int spp, const std::string& outputPath);
};
