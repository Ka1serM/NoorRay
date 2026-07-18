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

NoorRayUi::NoorRayUi(std::string scenePath, const uint32_t windowWidth, const uint32_t windowHeight)
    : window(windowWidth, windowHeight)
    , session(window)
{
    Context& context = session.context;
    Scene& scene = session.scene;
    if (!scenePath.empty())
        scene.load(scenePath);
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
    imGuiManager->addComponent<MainMenuBar>(
        "Menu", context, scene, *imGuiManager, std::move(scenePath));
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

    uint32_t frameIndex = 0;
    int submittedSamples = 0;
    bool renderComplete = false;
    uint64_t displayedRenderValue = 0;
    uint32_t displayedBufferIndex = 0;
    uint32_t displayedSelectionIndex = ~0u;
    InteropFrame pendingDisplayFrame{};
    bool isRunning = true, isFullscreen = false, firstFrame = true;
    uint64_t lastResizeEventMs = 0;
    constexpr uint64_t LiveResizeCooldownMs = 100;

    while (isRunning) {
        SDL_Event event{};
        while (window.pollEvent(event)) {
            const bool viewportConsumedEvent = viewportPanel->processEvent(event);
            if (!viewportConsumedEvent)
                imGuiManager->processEvent(event);
            if (event.type == SDL_EVENT_QUIT)
                isRunning = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F11) {
                isFullscreen = !isFullscreen;
                window.setFullscreen(isFullscreen);
            }
            if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                renderer->notifyResize(
                    static_cast<uint32_t>(std::max(event.window.data1, 0)),
                    static_cast<uint32_t>(std::max(event.window.data2, 0)));
                lastResizeEventMs = SDL_GetTicks();
            }
        }

        const bool liveResize = lastResizeEventMs != 0
            && SDL_GetTicks() - lastResizeEventMs < LiveResizeCooldownMs;

        // Build ImGui and apply scene edits before submitting CUDA work. Vulkan
        // swapchain acquisition can block, so it happens only after the next
        // OptiX frame has been queued and can run concurrently with that wait.
        imGuiManager->updateUi();

        CameraInstance* viewportCamera = scene.getRenderCamera();
        if (viewportCamera) {
            const glm::uvec2 resolution =
                viewportCamera->getCamera()->getSensor().resolution();
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
                frameIndex = 0;
                firstFrame = true;
            }
        }

        const bool displayHandoffPending = pendingDisplayFrame.readyValue != 0;
        if (!viewportCamera) {
            frameIndex = 0;
            firstFrame = true;
        } else if (renderPanel->isSaveRequested()) {
            renderPanel->executeSave();
        } else {
            const InteropFrame completedFrame = raytracer->getInteropFrame();
            if (!displayHandoffPending
                && raytracer->isFrameReady()
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
                pendingDisplayFrame = completedFrame;
            }

            // Keep at most one CUDA render in flight. It is submitted before
            // beginFrame(), so Vulkan presentation cannot pace OptiX startup.
            // A newly completed buffer may overlap one more render using the
            // other buffer. If Vulkan skipped the previous frame, wait until
            // its pending handoff is submitted before queueing anything else.
            if (!liveResize && !raytracer->isRenderInFlight() && !displayHandoffPending)
            {
                trainingPanel->tick();
                const bool resetAccumulation = firstFrame || scene.isDirty(Accumulation);
                const RenderSettings& renderSettings = scene.getRenderSettings();
                const bool proxyOverdraw =
                    renderSettings.gaussianProxyOverdrawVisualization;
                const int spp = std::max(1, renderSettings.samples);
                const int maxSamples = std::max(1, renderSettings.maxSamples);

                if (proxyOverdraw) {
                    frameIndex = 0;
                    submittedSamples = 0;
                    renderComplete = false;
                    firstFrame = false;
                    raytracer->renderFrame();
                    debugPanel->setSampleInfo(0, 0);
                } else {
                    if (resetAccumulation) {
                        frameIndex = 0;
                        submittedSamples = 0;
                        renderComplete = false;
                        debugPanel->resetRenderTimer();
                    }

                    if (renderComplete) {
                        debugPanel->setSampleInfo(submittedSamples, maxSamples);
                    } else {
                        if (!resetAccumulation)
                            ++frameIndex;

                        firstFrame = false;
                        raytracer->renderFrame(
                            frameIndex, static_cast<uint32_t>(submittedSamples));

                        submittedSamples = std::min(submittedSamples + spp, maxSamples);
                        renderComplete = submittedSamples >= maxSamples;
                        debugPanel->setSampleInfo(submittedSamples, maxSamples);
                    }
                }
            }
        }

        if (renderer->beginFrame()) {
            const vk::CommandBuffer cmd = renderer->getCurrentCommandBuffer();
            const auto dispatchViewportOverlay = [&](const uint32_t bufferIndex)
            {
                const RenderSettings& renderSettings = scene.getRenderSettings();
                const float cameraExposure = viewportCamera
                    ? viewportCamera->getCamera()->exposure : 0.0f;
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
                    proxyOverdraw ? 0.0f : cameraExposure,
                    proxyOverdraw ? 0 : static_cast<int>(renderSettings.bufferVisualization),
                    proxyOverdraw ? 0 : renderSettings.tonemappingEnabled,
                    viewportPanel->showOverlays());
                viewportPanel->onComputeFinished(cmd, viewport->getOutputImage());
                displayedSelectionIndex = selectedIndex;
            };

            if (pendingDisplayFrame.readyValue != 0) {
                displayedRenderValue = pendingDisplayFrame.readyValue;
                displayedBufferIndex = pendingDisplayFrame.bufferIndex;
                dispatchViewportOverlay(displayedBufferIndex);
                viewportPanel->setAovImages(
                    raytracer->getOutputCrypto(displayedBufferIndex),
                    raytracer->getOutputPosition(displayedBufferIndex));
                renderer->setExternalFrameSync(
                    pendingDisplayFrame.renderReadySemaphore,
                    pendingDisplayFrame.bufferReleasedSemaphore,
                    pendingDisplayFrame.readyValue);
                pendingDisplayFrame = {};
            } else if (displayedRenderValue != 0
                && scene.getActiveMeshInstanceIndex() != displayedSelectionIndex)
            {
                dispatchViewportOverlay(displayedBufferIndex);
            }

            imGuiManager->renderDrawData(cmd, renderer->getCurrentColorImageView(), renderer->getSwapchainExtent());
            renderer->endFrame();
        }
    }
    context.getDevice().waitIdle();
}
