#include "NoorRayUi.h"
#include "NoorRaySession.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <SDL3/SDL.h>
#include "Vulkan/Renderer.h"
#include "UI/ImGuiManager.h"
#include "UI/DebugPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/SceneGraphPanel.h"
#include "UI/ViewportPanel.h"
#include "UI/DetailsPanel.h"
#include "UI/EnvironmentPanel.h"
#include "UI/RenderPanel.h"
#include "UI/RenderSettingsPanel.h"
#include "UI/LensViewerPanel.h"
#include "UI/TrainingPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/Camera.h"
#include "Camera/CameraInstance.h"
#include "Camera/PerspectiveCamera.h"
#include "Raytracing/Raytracer.h"
#include "Vulkan/Viewport.h"
#include "Scene/LightInstance.h"

using namespace noorray;

NoorRayUi::NoorRayUi()
    : session(window)
{
    Context& context = session.context;
    Scene& scene = session.scene;
    Raytracer& raytracer = *session.raytracer;
    Renderer* renderer = session.renderer.get();
    viewport = std::make_unique<Viewport>(
        context, raytracer.getWidth(), raytracer.getHeight(),
        raytracer.getOutputColor(0), raytracer.getOutputColor(1),
        raytracer.getOutputAlbedo(0), raytracer.getOutputAlbedo(1),
        raytracer.getOutputNormal(0), raytracer.getOutputNormal(1),
        raytracer.getOutputCrypto(0), raytracer.getOutputCrypto(1),
        raytracer.getOutputPosition(0), raytracer.getOutputPosition(1),
        renderer->getColorImageFormat());
    Viewport* viewport = this->viewport.get();
    imGuiManager = std::make_unique<ImGuiManager>(
        window, context, renderer->getNumSwapchainImages(), renderer->getColorImageFormat());
    imGuiManager->addComponent<MainMenuBar>("Menu", context, scene, *imGuiManager);
    imGuiManager->addComponent<DebugPanel>("Timings");
    imGuiManager->addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager->addComponent<SceneGraphPanel>("Scene Graph", scene);
    imGuiManager->addComponent<DetailsPanel>("Details", scene);
    imGuiManager->addComponent<RenderSettingsPanel>("Render Settings", scene);
    imGuiManager->addComponent<RenderPanel>("Render", context, raytracer, *renderer, *viewport);
    imGuiManager->addComponent<LensViewerPanel>("Lens Viewer", scene);
    imGuiManager->addComponent<TrainingPanel>("Training", scene, raytracer);
    imGuiManager->addComponent<ViewportPanel>("Viewport", window, context, scene,
        viewport->getOutputImage(), raytracer.getOutputCrypto(), raytracer.getOutputPosition(),
        raytracer.getWidth(), raytracer.getHeight());
}

NoorRayUi::~NoorRayUi()
{
    imGuiManager.reset();
}

// ── runUi ─────────────────────────────────────────────────────────────────────

void NoorRayUi::run() {
    Context& context = session.context;
    Scene& scene = session.scene;
    Raytracer* raytracer = session.raytracer.get();
    Renderer* renderer = session.renderer.get();
    Viewport* viewport = this->viewport.get();
    auto* debugPanel       = dynamic_cast<DebugPanel*>(imGuiManager->getComponent("Timings"));
    auto* viewportPanel    = dynamic_cast<ViewportPanel*>(imGuiManager->getComponent("Viewport"));
    auto* renderPanel      = dynamic_cast<RenderPanel*>(imGuiManager->getComponent("Render"));
    auto* trainingPanel    = dynamic_cast<TrainingPanel*>(imGuiManager->getComponent("Training"));

    raytracer->setTimingEnabled(true);

    int frame = 0;
    int submittedSamples = 0;
    bool renderComplete = false;
    uint64_t displayedRenderValue = 0;
    uint32_t displayedBufferIndex = 0;
    uint32_t displayedSelectionIndex = ~0u;
    bool isRunning = true, isFullscreen = false, firstFrame = true;

    while (isRunning) {
        SDL_Event event{};
        while (window.pollEvent(event)) {
            imGuiManager->processEvent(event);
            if (event.type == SDL_EVENT_QUIT)
                isRunning = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F11) {
                isFullscreen = !isFullscreen;
                window.setFullscreen(isFullscreen);
            }
            if (event.type == SDL_EVENT_WINDOW_RESIZED)
                renderer->notifyResize(event.window.data1, event.window.data2);
        }

        if (renderer->beginFrame()) {
            const vk::CommandBuffer cmd = renderer->getCurrentCommandBuffer();

            // Run UI logic first. GPU-visible managed data is flushed below
            // only when no CUDA render is in flight.
            imGuiManager->updateUi();

            {
                CameraInstance* viewportCamera = scene.getRenderCamera();
                if (auto* cam = viewportCamera) {
                    const glm::uvec2 resolution = cam->getCamera()->getSensor().resolution();
                    if (resolution.x != raytracer->getWidth() || resolution.y != raytracer->getHeight()) {
                        context.getDevice().waitIdle();
                        raytracer->resize(resolution.x, resolution.y);
                        viewport->resize(
                            raytracer->getWidth(), raytracer->getHeight(),
                            raytracer->getOutputColor(0),    raytracer->getOutputColor(1),
                            raytracer->getOutputAlbedo(0),   raytracer->getOutputAlbedo(1),
                            raytracer->getOutputNormal(0),   raytracer->getOutputNormal(1),
                            raytracer->getOutputCrypto(0),   raytracer->getOutputCrypto(1),
                            raytracer->getOutputPosition(0), raytracer->getOutputPosition(1),
                            renderer->getColorImageFormat());
                        viewportPanel->resize(raytracer->getWidth(), raytracer->getHeight(),
                                              viewport->getOutputImage().getFormat());
                        frame = 0; firstFrame = true;
                    }
                }

                if (!viewportCamera) {
                    frame = 0;
                    firstFrame = true;
                } else if (renderPanel->isSaveRequested()) {
                    renderPanel->executeSave();
                } else {
                    const FrameInfo completedFrame = raytracer->getFrameInfo();
                    const auto dispatchViewportOverlay = [&](const uint32_t bufferIndex)
                    {
                        const RenderSettings& renderSettings = scene.getRenderSettings();
                        const bool proxyOverdraw =
                            renderSettings.gaussianProxyOverdrawVisualization;
                        const uint32_t selectedIndex = scene.getActiveMeshInstanceIndex();
                        const glm::mat4 viewProjection = viewportCamera
                            ? viewportCamera->getProjectionMatrix() * viewportCamera->getViewMatrix()
                            : glm::mat4(1.0f);
                        if (viewportPanel->showOverlays())
                            viewport->updateBillboards(scene);
                        viewport->dispatch(
                            cmd, bufferIndex, selectedIndex, viewProjection,
                            proxyOverdraw ? 0.0f : renderSettings.exposure,
                            proxyOverdraw ? 0 : static_cast<int>(renderSettings.bufferVisualization),
                            proxyOverdraw ? 0 : renderSettings.tonemappingEnabled,
                            viewportPanel->showOverlays());
                        viewportPanel->onComputeFinished(cmd, viewport->getOutputImage());
                        displayedSelectionIndex = selectedIndex;
                    };

                    if (raytracer->isFrameReady()
                        && completedFrame.readyValue > displayedRenderValue)
                    {
                        debugPanel->onComputeFinished(raytracer->getGpuTimeMs());
                        const RenderSettings& settings = scene.getRenderSettings();
                        constexpr int MinimumNoiseSamples = 16;
                        if (settings.noiseLimitEnabled
                            && submittedSamples >= MinimumNoiseSamples
                            && raytracer->getAverageNoiseVariance() <= settings.noiseLevel)
                        {
                            renderComplete = true;
                        }
                        displayedRenderValue = completedFrame.readyValue;
                        displayedBufferIndex = completedFrame.bufferIndex;
                        dispatchViewportOverlay(displayedBufferIndex);
                        viewportPanel->setAovImages(
                            raytracer->getOutputCrypto(completedFrame.bufferIndex),
                            raytracer->getOutputPosition(completedFrame.bufferIndex));
                        renderer->setExternalFrameSync(
                            completedFrame.renderReadySemaphore,
                            completedFrame.bufferReleasedSemaphore,
                            completedFrame.readyValue);
                    }
                    else if (displayedRenderValue != 0
                        && scene.getActiveMeshInstanceIndex() != displayedSelectionIndex)
                    {
                        dispatchViewportOverlay(displayedBufferIndex);
                    }

                    // Keep at most one CUDA render in flight. ImGui continues to
                    // present the last completed viewport image while it runs.
                    if (!raytracer->isRenderInFlight())
                    {
                        trainingPanel->tick();
                        if (scene.isDirty(Meshes))   raytracer->updateMeshes();
                        if (scene.isDirty(Textures)) raytracer->updateTextures();
                        else if (scene.isDirty(EnvironmentCdf)) raytracer->updateEnvironmentCdf();
                        if (scene.isDirty(Lights))   raytracer->updateLights();
                        if (scene.isDirty(TLAS))     raytracer->updateTLAS();
                        if (scene.isDirty(CameraState))
                            viewportCamera->rebuildCamera();

                        const bool resetAccumulation = firstFrame || scene.isDirty(Accumulation);
                        const RenderSettings& renderSettings = scene.getRenderSettings();
                        const bool proxyOverdraw =
                            renderSettings.gaussianProxyOverdrawVisualization;
                        const int spp = std::max(1, renderSettings.samples);
                        const int maxSamples = std::max(1, renderSettings.maxSamples);

                        if (proxyOverdraw) {
                            frame = 0;
                            submittedSamples = 0;
                            renderComplete = false;
                            scene.clearDirtyFlags();
                            firstFrame = false;
                            raytracer->renderFrame(PushData{.frame = 0});
                            debugPanel->setSampleInfo(0, 0);
                        } else {
                            if (resetAccumulation) {
                                frame = 0;
                                submittedSamples = 0;
                                renderComplete = false;
                                debugPanel->resetRenderTimer();
                            }

                            if (renderComplete) {
                                debugPanel->setSampleInfo(submittedSamples, maxSamples);
                            } else {
                                if (!resetAccumulation)
                                    ++frame;

                                scene.clearDirtyFlags();
                                firstFrame = false;
                                raytracer->renderFrame(PushData{
                                    .frame = frame,
                                    .accumulatedSampleOffset = static_cast<uint32_t>(submittedSamples)});

                                submittedSamples = std::min(submittedSamples + spp, maxSamples);
                                renderComplete = submittedSamples >= maxSamples;
                                debugPanel->setSampleInfo(
                                    submittedSamples, maxSamples);
                            }
                        }
                    }
                }
            }

            imGuiManager->renderDrawData(cmd, renderer->getCurrentColorImageView(), renderer->getSwapchainExtent());
            renderer->endFrame();
        }
    }
    context.getDevice().waitIdle();
}
