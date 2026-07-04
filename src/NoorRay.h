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

    int m_cliSpp = 64;
    std::string m_cliOutput = "output.exr";
    bool m_cliStats = false;

public:
    NoorRay(int windowWidth, int windowHeight);
    NoorRay(const std::string& scenePath, int spp,
            const std::string& outputPath, int width, int height, bool statsEnabled = false);
    ~NoorRay();

    void runUi();
    void runCli();
};
