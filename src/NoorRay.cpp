#include "NoorRay.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <SDL3/SDL.h>
#include "UI/DebugPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/SceneGraphPanel.h"
#include "UI/ViewportPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/PerspectiveCamera.h"
#include "Raytracing/ComputeRaytracer.h"
#include "Raytracing/RtxRaytracer.h"
#include "UI/DetailsPanel.h"
#include "UI/EnvironmentPanel.h"
#include "UI/RenderPanel.h"
#include "Vulkan/Tonemapper.h"

NoorRay::~NoorRay() = default;

NoorRay::NoorRay(const int windowWidth, const int windowHeight, const int renderWidth, const int renderHeight)
    : context(windowWidth, windowHeight),
      renderer(context),
      imGuiManager(context, renderer.getSwapchainImages()),
      scene(context)
{
    // Apply DPI scaling for the raytracer
    const float dpiScale = context.getDPIScale();
    int scaledRenderWidth  = static_cast<int>(static_cast<float>(renderWidth)  * dpiScale);
    int scaledRenderHeight = static_cast<int>(static_cast<float>(renderHeight) * dpiScale);

    if (context.isRtxSupported())
        raytracer = std::make_unique<RtxRaytracer>(scene, scaledRenderWidth, scaledRenderHeight);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, scaledRenderWidth, scaledRenderHeight);

    tonemapper = std::make_unique<Tonemapper>(context, raytracer->getWidth(), raytracer->getHeight(), raytracer->getOutputColor());

    
    imGuiManager.addComponent<MainMenuBar>("Menu", context, scene);
    imGuiManager.addComponent<DebugPanel>("Debug");
    imGuiManager.addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager.addComponent<SceneGraphPanel>("Scene Graph", scene);
    imGuiManager.addComponent<DetailsPanel>("Details", scene);
    imGuiManager.addComponent<RenderPanel>("Render", context, *raytracer, renderer, *tonemapper);
    imGuiManager.addComponent<ViewportPanel>("Viewport", context, scene, tonemapper->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(), raytracer->getWidth(), raytracer->getHeight());

    
    int imgWidth, imgHeight, channels;
    static constexpr unsigned char hdriData[] = {
        #embed "../assets/textures/whipple_creek_regional_park_04_2k.hdr"
    };
    float* hdriPixels = stbi_loadf_from_memory(hdriData, static_cast<int>(sizeof(hdriData)), &imgWidth, &imgHeight, &channels, 4);
    if (!hdriPixels)
        throw std::runtime_error("Failed to load HDR texture from memory" + std::string(stbi_failure_reason()));

    scene.add(Texture(context, "HDRI Sky", hdriPixels, imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));
    stbi_image_free(hdriPixels);

    auto sphereMaterial = Material{};
    sphereMaterial.metallic = 1.0;
    sphereMaterial.roughness = 0.2;
    auto sphere = MeshAsset::CreateSphere(scene, "Sphere", sphereMaterial);
    scene.add(sphere);
    auto instance = std::make_unique<MeshInstance>(scene, "Sphere Instance", sphere, Transform(vec3(0, 0, 0)));
    const uint32_t instanceIndex = scene.add(std::move(instance));
    scene.setActiveObjectIndex(instanceIndex);
    
    float aspectRatio = static_cast<float>(raytracer->getWidth()) / static_cast<float>(raytracer->getHeight());
    auto cam = std::make_unique<PerspectiveCamera>(scene, "Camera", Transform{vec3(4.15f,-3.25f,-3.75f), vec3(-30.f, -48.f, 0.f), vec3( 1)}, aspectRatio, 36.0f, 24.0f, 45.0f, 2.8, 5.0f, 2.0f);
    scene.add(std::move(cam));
}

void NoorRay::run() {
    auto* debugPanel = dynamic_cast<DebugPanel*>(imGuiManager.getComponent("Debug"));
    auto* viewportPanel = dynamic_cast<ViewportPanel*>(imGuiManager.getComponent("Viewport"));
    const auto* environmentPanel = dynamic_cast<EnvironmentPanel*>(imGuiManager.getComponent("Environment"));
    auto* renderPanel = dynamic_cast<RenderPanel*>(imGuiManager.getComponent("Render"));

    int frame = 0;
    bool isRunning = true;
    bool isFullscreen = false;
    bool framebufferResized = false;
    
    auto handleEvents = [&]() {
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
            default: ;
            }
        }
    };
    
    while (isRunning) {
        handleEvents();
        
        if (framebufferResized) {
            context.queryWindowSize();
            renderer.recreateSwapChain();
            imGuiManager.recreateForSwapChain(renderer.getSwapchainImages());
            framebufferResized = false;
        }

        if (context.getWindowWidth() == 0 || context.getWindowHeight() == 0)
            continue;

        try {
            imGuiManager.renderUi();
            vk::CommandBuffer cmd = renderer.beginFrame();
            if (!cmd) {
                framebufferResized = true;
                continue;
            }

            imGuiManager.Draw(cmd, renderer.getCurrentSwapchainImageIndex());
            
            if (renderer.isComputeWorkFinished()) {
                debugPanel->onComputeFinished();
                // A compute frame is ready. We can now safely access its output.
                context.oneTimeSubmit([&](const vk::CommandBuffer copyCmd) {
                    viewportPanel->updateDisplayImage(copyCmd, tonemapper->getOutputImage());
                });
                // Now that the result is safely copied, we can decide what to do next.
                if (renderPanel->isSaveRequested()) {
                    // The pipeline is paused. The save function can safely access the finished data.
                    renderPanel->executeSave();
                } else {
                    // The pipeline is running. Submit the next compute job.
                    if (scene.isAnyDirty()) {
                        if (scene.isDirty(Meshes)) raytracer->updateMeshes();
                        if (scene.isDirty(Textures)) raytracer->updateTextures();
                        if (scene.isDirty(TLAS)) raytracer->updateTLAS();
                    }

                    if (scene.isDirty(Accumulation)) frame = 0; else frame++;
                    
                    scene.clearDirtyFlags();
                    
                    renderer.submitCompute([&](const vk::CommandBuffer computeCmd) {
                        PushConstantsData pushConstants{};
                        pushConstants.push.frame = frame;
                        pushConstants.push.diffuseBounces = renderPanel->getDiffuseBounces();
                        pushConstants.push.specularBounces = renderPanel->getSpecularBounces();
                        pushConstants.push.transmissionBounces = renderPanel->getTransmissionBounces();
                        pushConstants.push.samples  = renderPanel->getSamples();
                        pushConstants.push.exposure = renderPanel->getExposure();
                        pushConstants.camera  = scene.getActiveCamera()->getCameraData();
                        pushConstants.environment = environmentPanel->getEnvironmentData();

                        raytracer->render(computeCmd, pushConstants);
                        tonemapper->dispatch(computeCmd);
                    });
                }
            }

            if (renderer.endFrame())
                framebufferResized = true;

        } catch (const vk::OutOfDateKHRError&) {
            framebufferResized = true;
        } catch (const std::exception& e) {
           LOG_ERROR("Main loop exception: " << e.what());
            isRunning = false;
        }
    }

    context.getDevice().waitIdle();
}