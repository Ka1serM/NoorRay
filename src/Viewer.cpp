#include "Viewer.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <SDL3/SDL.h>
#include "imgui.h"
#include "UI/DebugPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/OutlinerDetailsPanel.h"
#include "UI/ViewportPanel.h"
#include "portable-file-dialogs.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Utils.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/PerspectiveCamera.h"
#include "Raytracing/Computeraytracer.h"
#include "Raytracing/Rtxraytracer.h"
#include "UI/EnvironmentPanel.h"
#include "Vulkan/Tonemapper.h"

Viewer::Viewer(const int width, const int height)
    : width(width),
      height(height),
      context(width, height),
      renderer(context, width, height),
      imGuiManager(context, renderer.getSwapchainImages(), width, height),
      scene(context)
{
    const int viewportWidth = width / 2;
    const int viewportHeight = height / 2;

    if (context.isRtxSupported())
        raytracer = std::make_unique<RtxRaytracer>(scene, viewportWidth, viewportHeight);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, viewportWidth, viewportHeight);
    
    gpuImageTonemapper = std::make_unique<Tonemapper>(context, viewportWidth, viewportHeight, raytracer->getOutputColor());
    
    setupScene();
    setupUI();
}

void Viewer::recreateSwapChain() {
    // Get the new window size in pixels.
    SDL_GetWindowSizeInPixels(context.getWindow(), &width, &height);

    // If the window is minimized, pause execution until it is restored.
    while (width == 0 || height == 0) {
        SDL_GetWindowSizeInPixels(context.getWindow(), &width, &height);
        SDL_WaitEvent(nullptr);
    }

    context.getDevice().waitIdle();
    
    renderer.recreateSwapChain(width, height);
    imGuiManager.recreateForSwapChain(renderer.getSwapchainImages(), width, height);

    std::cout << "Recreated swapchain to " << width << "x" << height << std::endl;
}


void Viewer::run() {
    using clock = std::chrono::high_resolution_clock;
    auto lastTime = clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;
    float deltaTime = 0.0f;

    auto recordComputeWork = [&](const vk::CommandBuffer cmd) {
        PushConstantsData pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.camera = scene.getActiveCamera()->getCameraData();
        if (const auto* environment = dynamic_cast<EnvironmentPanel*>(imGuiManager.getComponent("Environment")))
            pushConstantData.push.hdriTexture = environment->getHdriTexture();

        raytracer->render(cmd, pushConstantData);
        gpuImageTonemapper->dispatch(cmd);
    };

    bool newFrameReady = false;
    bool isRunning = true;
    bool isFullscreen = false;

    auto* gpuViewport = dynamic_cast<ViewportPanel*>(imGuiManager.getComponent("GPU Viewport"));
    auto* debugPanel = dynamic_cast<DebugPanel*>(imGuiManager.getComponent("Debug"));
    
    while (isRunning) {
        
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            
            switch (event.type) {
            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_F11) {
                    isFullscreen = !isFullscreen;
                    if (isFullscreen) {
                        SDL_SetWindowFullscreenMode(context.getWindow(), NULL);
                        SDL_SetWindowFullscreen(context.getWindow(), true);
                    } else
                        SDL_SetWindowFullscreen(context.getWindow(), false);
                }
                break;
                case SDL_EVENT_QUIT:
                    isRunning = false;
                    break;
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
                case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
                    framebufferResized = true;
                    break;
                default:
                    break;
            }
        }
        
        // --- Time/FPS ---
        auto currentTime = clock::now();
        deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        timeAccumulator += deltaTime;
        frameCounter++;
        if (timeAccumulator >= 1.0f) {
                debugPanel->setFps(static_cast<float>(frameCounter) / timeAccumulator);
            timeAccumulator = 0.0f;
            frameCounter = 0;
        }
        
        // --- Resize Handling ---
        // If a resize event occurred, recreate the swapchain and skip this frame.
        if (framebufferResized) {
            recreateSwapChain();
            framebufferResized = false;
            continue;
        }

        // --- UI and Scene ---
        imGuiManager.renderUi();

        if (scene.isTlasDirty() || scene.isMeshesDirty() || scene.isTexturesDirty()) {
            context.getDevice().waitIdle();
            if (scene.isMeshesDirty()) raytracer->updateMeshes();
            if (scene.isTexturesDirty()) raytracer->updateTextures();
            if (scene.isTlasDirty()) raytracer->updateTLAS();
        }

        if (scene.isAccumulationDirty())
            frame = 0;
        else
            frame++;

        scene.clearDirtyFlags();

        if (renderer.isComputeWorkFinished() && !newFrameReady) {
            newFrameReady = true;
            renderer.submitCompute(recordComputeWork);
        }

        // --- Begin Frame / Record / Draw ---
        vk::CommandBuffer commandBuffer;
        try {
            commandBuffer = renderer.beginFrame();
        } catch (const vk::OutOfDateKHRError&) {
            framebufferResized = true;
            continue;
        }

        try {
            if (newFrameReady)
                gpuViewport->recordCopy(commandBuffer, gpuImageTonemapper->getOutputImage());

            imGuiManager.Draw(commandBuffer, renderer.getCurrentSwapchainImageIndex(), width, height);

            renderer.endFrame(newFrameReady);
            newFrameReady = false;

        } catch (const vk::OutOfDateKHRError&) {
            framebufferResized = true;
        }
    }

    context.getDevice().waitIdle();
}

Viewer::~Viewer() {
    std::cout << "Destroying Viewer...." << std::endl;
}

void Viewer::setupUI() {
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto environmentPanel = std::make_unique<EnvironmentPanel>(scene);
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(scene);

    auto gpuViewportUniquePtr = std::make_unique<ViewportPanel>(scene, gpuImageTonemapper->getOutputImage(), raytracer->getOutputCrypto(), width/2, height/2, "GPU Viewport");

    mainMenuBar->setCallback("File.Quit", [&] {
        SDL_Event quitEvent;
        quitEvent.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&quitEvent);
    });

    mainMenuBar->setCallback("File.Import.Obj", [this] {
        const auto selection = pfd::open_file("Import OBJ Model", ".", { "OBJ Files", "*.obj", "All Files", "*" }).result();
        if (!selection.empty()) {
            
            try {
                std::vector<Vertex> vertices;
                std::vector<uint32_t> indices;
                std::vector<Face> faces;
                std::vector<Material> materials;
                Utils::loadObj(scene, selection[0], vertices, indices, faces, materials);
                auto meshAsset = std::make_shared<MeshAsset>(scene, selection[0], std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
                
                this->scene.add(meshAsset);
                auto instance = std::make_unique<MeshInstance>(this->scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
                int instanceIndex = this->scene.add(std::move(instance));
                
                    scene.setActiveObjectIndex(instanceIndex);

            } catch (const std::exception& e) {
                std::cerr << "Import failed: " << e.what() << std::endl;
            }
        }
    });

    mainMenuBar->setCallback("File.Import.CrtScene", [this] {
        const auto selection = pfd::open_file("Import CrtScene", ".", { "CRT Scene Files", "*.crtscene", "All Files", "*" }).result();
        if (!selection.empty()) {
            
        try {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            std::vector<Face> faces;
            std::vector<Material> materials;
            Utils::loadCrtScene(scene, selection[0], vertices, indices, faces, materials);
            auto meshAsset = std::make_shared<MeshAsset>(scene, selection[0], std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
            this->scene.add(meshAsset);
            auto instance = std::make_unique<MeshInstance>(this->scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
            int instanceIndex = this->scene.add(std::move(instance));
            
                scene.setActiveObjectIndex(instanceIndex);

        } catch (const std::exception& e) {
            std::cerr << "Import failed: " << e.what() << std::endl;
        }
    }
    });
    
    mainMenuBar->setCallback("File.Import.Texture", [this] {
      const auto selection = pfd::open_file(
          "Import Texture",
          ".",
          {
              "Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic",
              "All Files", "*"
          }).result();

      if (!selection.empty()) {
          try {
              scene.add(Texture(context, selection[0]));
          } catch (const std::exception& e) {
              std::cerr << "Import failed: " << e.what() << std::endl;
          }
      }
    });

    mainMenuBar->setCallback("Add.Cube", [this] {
        auto cube = MeshAsset::CreateCube(scene, "Cube", {});
        this->scene.add(cube);
        auto instance = std::make_unique<MeshInstance>(this->scene, "Cube Instance", cube, Transform(vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
            scene.setActiveObjectIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Plane", [this] {
        auto plane = MeshAsset::CreatePlane(this->scene, "Plane", {});
        this->scene.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->scene, "Plane Instance", plane, Transform(vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
            scene.setActiveObjectIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Sphere", [this] {
        auto sphere = MeshAsset::CreateSphere(this->scene, "Sphere", {}, 24, 48);
        this->scene.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->scene, "Sphere Instance", sphere, Transform(vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
            scene.setActiveObjectIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.RectLight", [this] {
        Material material{};
        material.emission = vec3(1, 1, 1);
        material.emissionStrength = 10.0f; // White light with intensity
        auto plane = MeshAsset::CreatePlane(this->scene, "RectLight", material);
        this->scene.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->scene, "RectLight Instance", plane, Transform(vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
            scene.setActiveObjectIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.SphereLight", [this] {
        Material material{};
        material.emission = vec3(1, 1, 1);
        material.emissionStrength = 10.0f; // White light with intensity
        auto sphere = MeshAsset::CreateSphere(this->scene, "SphereLight", material, 24, 48);
        this->scene.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->scene, "SphereLight Instance", sphere, Transform(vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        scene.setActiveObjectIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.DiskLight", [this] {
        Material material{};
        material.emission = vec3(1, 1, 1);
        material.emissionStrength = 10.0f; // White light with intensity
        auto disk = MeshAsset::CreateDisk(this->scene, "DiskLight", material, 48);
        this->scene.add(disk);
        auto instance = std::make_unique<MeshInstance>(this->scene, "DiskLight Instance", disk, Transform(vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        scene.setActiveObjectIndex(instanceIndex);
    });

    imGuiManager.add(std::move(mainMenuBar));
    imGuiManager.add(std::move(debugPanel));
    imGuiManager.add(std::move(environmentPanel));
    imGuiManager.add(std::move(outlinerDetailsPanel));
    imGuiManager.add(std::move(gpuViewportUniquePtr));
}

void Viewer::setupScene() {
    
    static constexpr unsigned char data[] = {
        #embed "../assets/Ultimate_Skies_4k_0036.hdr"
    };
    int imgWidth, imgHeight, channels;
    float* pixels = stbi_loadf_from_memory(data, static_cast<int>(sizeof(data)), &imgWidth, &imgHeight, &channels, 4);
    if (!pixels)
        throw std::runtime_error("Failed to load HDR texture from memory");

    scene.add(Texture(context, "HDRI Sky", pixels, imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));
    
    stbi_image_free(pixels);

    scene.add(Texture(context, "White", (const uint8_t[]){255, 255, 255, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));
    scene.add(Texture(context, "Black", (const uint8_t[]){0, 0, 0, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));
    
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    auto cam = std::make_unique<PerspectiveCamera>(scene, "Camera", Transform{vec3(0, 0, 5.0f), vec3(-180, 0, 180), vec3(0)}, aspectRatio, 36.0f, 24.0f, 30.0f, 0.0f, 10.0f, 2.0f);
    scene.add(std::move(cam));
}
