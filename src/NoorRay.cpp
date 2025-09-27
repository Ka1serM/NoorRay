#include "NoorRay.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <RmlUi/Debugger/Debugger.h>
#include <SDL3/SDL.h>
#include "UI/ImGui/ViewportPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/PerspectiveCamera.h"
#include "Raytracing/ComputeRaytracer.h"
#include "Raytracing/RtxRaytracer.h"
#include "UI/ImGui/DebugPanel.h"
#include "UI/ImGui/DetailsPanel.h"
#include "UI/ImGui/MainMenuBar.h"
#include "UI/ImGui/EnvironmentPanel.h"
#include "UI/ImGui/RenderPanel.h"
#include "UI/ImGui/SceneGraphPanel.h"
#include "Vulkan/Tonemapper.h"

NoorRay::~NoorRay() = default;

NoorRay::NoorRay(const int windowWidth, const int windowHeight, const int renderWidth, const int renderHeight)
    : context(windowWidth, windowHeight),
      scene(context),
      renderer(context, context.getWindowWidth(), context.getWindowHeight()),
      imGuiManager(context, renderer.getNumSwapchainImages(), renderer.getColorImageFormat()),
      rmlUiManager(context, scene, renderer)
{
    const float dpiScale = context.getDPIScale();
    int scaledRenderWidth  = static_cast<int>(static_cast<float>(renderWidth)  * dpiScale);
    int scaledRenderHeight = static_cast<int>(static_cast<float>(renderHeight) * dpiScale);

    if (context.isRtxSupported())
        raytracer = std::make_unique<RtxRaytracer>(scene, scaledRenderWidth, scaledRenderHeight);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, scaledRenderWidth, scaledRenderHeight);

    tonemapper = std::make_unique<Tonemapper>(context, raytracer->getWidth(), raytracer->getHeight(), raytracer->getOutputColor(), renderer.getColorImageFormat());

    rmlUiManager.bindViewportImage(tonemapper->getOutputImage());

    //imGuiManager.addComponent<MainMenuBar>("Menu", context, scene);
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
    auto cam = std::make_unique<PerspectiveCamera>(scene, "Camera", Transform{vec3(4.15f,-3.25f,-3.75f), vec3(-30.f, -48.f, 0.f), vec3( 1)}, aspectRatio, 36.0f, 24.0f, 45.0f, 2.8, 10.0f, 2.0f);
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
    
    while (isRunning) {
        
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            imGuiManager.processEvent(event);
            rmlUiManager.processEvent(context.getWindow(), event);
            
            if (event.type == SDL_EVENT_QUIT)
                isRunning = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F11) {
                isFullscreen = !isFullscreen;
                SDL_SetWindowFullscreen(context.getWindow(), isFullscreen);
            }
            if (event.type == SDL_EVENT_WINDOW_RESIZED)
            {
                renderer.notifyResize(event.window.data1, event.window.data2);
                rmlUiManager.resize(event.window.data1, event.window.data2);
            }
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_R && event.key.mod & SDL_KMOD_CTRL)
            {
                rmlUiManager.reload();
                rmlUiManager.bindViewportImage(tonemapper->getOutputImage());
            }
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_D && event.key.mod & SDL_KMOD_CTRL)
                Rml::Debugger::SetVisible(!Rml::Debugger::IsVisible());
        }

        if (renderer.beginFrame())
        {
            const vk::CommandBuffer cmd = renderer.getCurrentCommandBuffer();

            if (renderer.isComputeWorkFinished()) {
                debugPanel->onComputeFinished();
                
                viewportPanel->updateDisplayImage(cmd, tonemapper->getOutputImage());

                if (renderPanel->isSaveRequested()) {
                    renderPanel->executeSave();
                }
                else
                {
                    if (scene.isAnyDirty()) {
                        if (scene.isDirty(Meshes))
                            raytracer->updateMeshes();
                        if (scene.isDirty(Textures))
                            raytracer->updateTextures();
                        if (scene.isDirty(TLAS))
                            raytracer->updateTLAS();
                    }
                    if (scene.isDirty(Accumulation))
                        frame = 0;
                    else
                        frame++;

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

            rmlUiManager.render(cmd, renderer.getCurrentColorImage(), renderer.getCurrentColorImageView(), renderer.getDepthImageView(), renderer.getSwapchainExtent(), renderer.getCurrentInFlightFence());
            //imGuiManager.render(cmd, renderer.getCurrentColorImageView(), renderer.getSwapchainExtent());

            renderer.endFrame();
        }
    }
    context.getDevice().waitIdle();
}