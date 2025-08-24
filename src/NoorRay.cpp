#include "NoorRay.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <SDL3/SDL.h>
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
#include "UI/RenderPanel.h"
#include "Vulkan/Tonemapper.h"

NoorRay::NoorRay(const int windowWidth, const int windowHeight, const int renderWidth, const int renderHeight)
    : windowWidth(windowWidth),
      windowHeight(windowHeight),
      context(windowWidth, windowHeight),
      renderer(context, windowWidth, windowHeight),
      imGuiManager(context, renderer.getSwapchainImages(), windowWidth, windowHeight),
      scene(context)
{
    if (context.isRtxSupported())
       raytracer = std::make_unique<RtxRaytracer>(scene, renderWidth, renderHeight);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, renderWidth, renderHeight);
    
    tonemapper = std::make_unique<Tonemapper>(context, raytracer->getWidth(), raytracer->getHeight(), raytracer->getOutputColor());
    
    setupScene();
    setupUI();
}

void NoorRay::recreateSwapChain() {
    SDL_GetWindowSizeInPixels(context.getWindow(), &windowWidth, &windowHeight);
    renderer.recreateSwapChain(windowWidth, windowHeight);
    imGuiManager.recreateForSwapChain(renderer.getSwapchainImages(), windowWidth, windowHeight);
}

void NoorRay::run()
{
    int frame = 0;
    
    using clock = std::chrono::high_resolution_clock;
    auto lastTime = clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;
    
    auto* debugPanel = dynamic_cast<DebugPanel*>(imGuiManager.getComponent("Debug"));
    auto* gpuViewport = dynamic_cast<ViewportPanel*>(imGuiManager.getComponent("Viewport"));
    const auto* environment = dynamic_cast<EnvironmentPanel*>(imGuiManager.getComponent("Environment"));
    auto* renderPanel = dynamic_cast<RenderPanel*>(imGuiManager.getComponent("Render"));
    
    bool isRunning = true;
    bool isFullscreen = false;
    bool framebufferResized = false;

    auto recordComputeCommands = [&](const vk::CommandBuffer cmd) {
        PushConstantsData pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.push.diffuseBounces = renderPanel->getDiffuseBounces();
        pushConstantData.push.specularBounces = renderPanel->getSpecularBounces();
        pushConstantData.push.transmissionBounces = renderPanel->getTransmissionBounces();
        pushConstantData.push.samples = renderPanel->getSamples();
        pushConstantData.camera = scene.getActiveCamera()->getCameraData();
        pushConstantData.environment = environment->getEnvironmentData();

        raytracer->render(cmd, pushConstantData);
        tonemapper->dispatch(cmd);
    };
    
    while (isRunning) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            
            switch (event.type) {
            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_F11) {
                    isFullscreen = !isFullscreen;
                    SDL_SetWindowFullscreen(context.getWindow(), isFullscreen);
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
        
        auto currentTime = clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        timeAccumulator += deltaTime;
        if (timeAccumulator >= 1.0f) {
            debugPanel->setFps(static_cast<float>(frameCounter) / timeAccumulator);
            timeAccumulator = 0.0f;
            frameCounter = 0;
        }

        if (framebufferResized) {
            recreateSwapChain();
            framebufferResized = false;
        }

        //skip frame
        if (windowWidth == 0 || windowHeight == 0)
            continue;

        try {
            imGuiManager.renderUi();
            
            if (scene.isTlasDirty() || scene.isMeshesDirty() || scene.isTexturesDirty()) {
                renderer.waitForComputeIdle();
                if (scene.isMeshesDirty()) raytracer->updateMeshes();
                if (scene.isTexturesDirty()) raytracer->updateTextures();
                if (scene.isTlasDirty()) raytracer->updateTLAS();
            }

            if (scene.isAccumulationDirty())
                frame = 0;
            else
                frame++;
            scene.clearDirtyFlags();

            bool computeWasSubmitted = false;
            if (renderer.isComputeWorkFinished()) {
                frameCounter++;
                // The previous frame's compute work is done. The output images are stable.
                // Now, we decide whether to save them or to render the next frame.
                if (renderPanel->isSaveRequested()) {
                    renderPanel->executeSave();
                } else {
                    renderer.submitCompute(recordComputeCommands);
                    computeWasSubmitted = true;
                }
            }

            vk::CommandBuffer commandBuffer = renderer.beginFrame();
            if (!commandBuffer) {
                framebufferResized = true;
                continue;
            }

            gpuViewport->recordCopy(commandBuffer, tonemapper->getOutputImage());
            imGuiManager.Draw(commandBuffer, renderer.getCurrentSwapchainImageIndex(), windowWidth, windowHeight);

            if (renderer.endFrame(computeWasSubmitted))
                framebufferResized = true;

        } catch (const vk::OutOfDateKHRError&) {
            framebufferResized = true;
        } catch (const std::exception& e) {
            std::cerr << "An exception occurred in the main loop: " << e.what() << std::endl;
            isRunning = false;
        }
    }

    context.getDevice().waitIdle();
}

NoorRay::~NoorRay() {
    std::cout << "Destroying NoorRay...." << std::endl;
}

void NoorRay::setupUI() {
    imGuiManager.addComponent<MainMenuBar>("Menu", context, scene);
    imGuiManager.addComponent<DebugPanel>("Debug");
    imGuiManager.addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager.addComponent<OutlinerDetailsPanel>("Outliner", scene);
    imGuiManager.addComponent<RenderPanel>("Render", context, *raytracer, renderer);
    imGuiManager.addComponent<ViewportPanel>("Viewport", context, scene, tonemapper->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getWidth(), raytracer->getHeight());
}

void NoorRay::setupScene() {
    int imgWidth, imgHeight, channels;
    static constexpr unsigned char hdriData[] = {
        //#embed "../assets/Ultimate_Skies_4k_0036.hdr"
        #embed "../assets/whipple_creek_regional_park_04_2k.hdr"
    };
    float* hdriPixels = stbi_loadf_from_memory(hdriData, static_cast<int>(sizeof(hdriData)), &imgWidth, &imgHeight, &channels, 4);
    if (!hdriPixels)
        throw std::runtime_error("Failed to load HDR texture from memory");
    scene.add(Texture(context, "HDRI Sky", hdriPixels, imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));
    stbi_image_free(hdriPixels);

    float aspectRatio = static_cast<float>(raytracer->getWidth()) / static_cast<float>(raytracer->getHeight());
    auto cam = std::make_unique<PerspectiveCamera>(scene, "Camera", Transform{vec3(0, 0, -5.0f), vec3(0), vec3(1)}, aspectRatio, 36.0f, 24.0f, 30.0f, 1.8f, 5.0f, 2.0f);
    scene.add(std::move(cam));
}
