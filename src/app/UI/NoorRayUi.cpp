#include "NoorRayUi.h"
#include "Materials/MaterialX/MaterialXSceneRuntime.h"
#include "NoorRaySession.h"
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <SDL3/SDL.h>
#include "Backend/Vulkan/Runtime/Renderer.h"
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
#include "UI/MaterialXNodeEditorPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "backends/imgui_impl_sdl3.h"
#include "Rendering/Camera/Camera.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/PerspectiveCamera.h"
#include "Backend/OptiX/Runtime/Raytracer.h"
#include "Backend/Vulkan/Viewport/Viewport.h"
#include "Scene/Objects/LightInstance.h"

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
    materialxRuntime = std::make_unique<MaterialXSceneRuntime>();
    materialxRuntime->compilePending(scene, raytracer,
        std::filesystem::path(scenePath).parent_path().string());
    Renderer* renderer = session.renderer.get();
    // The editor always needs the AOVs: the viewport compute pass samples
    // albedo, normal and cryptomatte, and picking reads back cryptomatte and
    // position. Ask for them (and for the frame they are sized against) before
    // anything binds those images.
    raytracer.setAovEnabled(true);
    if (CameraInstance* camera = scene.getRenderCamera()) {
        const glm::uvec2 resolution = camera->getCamera()->getSensor().resolution();
        raytracer.resize(resolution.x, resolution.y);
    }
    viewport = std::make_unique<Viewport>(
        context, raytracer.getWidth(), raytracer.getHeight(),
        raytracer.getOutputColor(),
        raytracer.getOutputAlbedo(),
        raytracer.getOutputNormal(),
        raytracer.getOutputCrypto(),
        raytracer.getOutputPosition(),
        raytracer.getOutputDenoised(),
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
    imGuiManager->addComponent<MaterialXNodeEditorPanel>("MaterialX Node Editor", scene);
    imGuiManager->addComponent<ViewportPanel>("Viewport", window, context, scene,
        raytracer, viewport->getOutputImage(), raytracer.getOutputCrypto(),
        raytracer.getOutputPosition(), raytracer.getWidth(), raytracer.getHeight());
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
    auto* mainMenuBar      = dynamic_cast<MainMenuBar*>(imGuiManager->getComponent("Menu"));
    auto* viewportPanel    = dynamic_cast<ViewportPanel*>(imGuiManager->getComponent("Viewport"));
    auto* renderPanel      = dynamic_cast<RenderPanel*>(imGuiManager->getComponent("Render"));

    raytracer->setTimingEnabled(true);

    uint32_t frameIndex = 0;
    int submittedSamples = 0;
    bool renderComplete = false;
    uint64_t displayedRenderValue = 0;
    uint32_t displayedCryptomatteId = ~0u;
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
        // Finish the one frame that may have been submitted before this UI
        // edit. OptiX module creation and a launch on the same device context
        // otherwise serialize inside the driver and can strand the Vulkan
        // interop handoff, making the whole window appear frozen.
        if (materialxRuntime->needsCompilation(scene))
            raytracer->waitForRender();
        materialxRuntime->compilePending(scene, *raytracer,
            std::filesystem::path(mainMenuBar->getCurrentScenePath())
                .parent_path().string());
        const bool materialCompilationPending =
            materialxRuntime->hasPendingCompilations();

        CameraInstance* viewportCamera = scene.getRenderCamera();
        if (viewportCamera && !materialCompilationPending) {
            const glm::uvec2 resolution =
                viewportCamera->getCamera()->getSensor().resolution();
            if (resolution.x != raytracer->getWidth() || resolution.y != raytracer->getHeight()) {
                context.getDevice().waitIdle();
                raytracer->resize(resolution.x, resolution.y);
                viewport->resize(
                    raytracer->getWidth(), raytracer->getHeight(),
                    raytracer->getOutputColor(),    raytracer->getOutputAlbedo(),
                    raytracer->getOutputNormal(),   raytracer->getOutputCrypto(),
                    raytracer->getOutputPosition(),
                    raytracer->getOutputDenoised(),
                    renderer->getColorImageFormat());
                viewportPanel->resize(raytracer->getWidth(), raytracer->getHeight(),
                                      viewport->getOutputImage().getFormat());
                frameIndex = 0;
                firstFrame = true;
            }
        }

        static int nrOdFrames = 0;
        if (std::getenv("NR_PROXY_OVERDRAW") != nullptr && ++nrOdFrames > 300) {
            scene.getRenderSettings().gaussianProxyOverdrawVisualization = true;
            scene.getRenderSettings().bufferVisualization =
                BufferVisualization::ProxyOverdraw;
        }
        bool displayHandoffPending = pendingDisplayFrame.readyValue != 0;
        if (!viewportCamera) {
            frameIndex = 0;
            firstFrame = true;
        } else if (materialCompilationPending) {
            // OptiX serializes module creation and launches on a device
            // context. Calling any render/resize operation here would block
            // this UI thread behind the background material compiler. Keep
            // presenting the last completed viewport image until the prepared
            // callable is ready to install.
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
                pendingDisplayFrame = completedFrame;
                // The frame is now owned by the Vulkan handoff below. Do not
                // queue another CUDA render in this same UI iteration: the
                // local flag was computed before completedFrame was adopted.
                displayHandoffPending = true;
            }

            // Keep at most one CUDA render in flight. It is submitted before
            // beginFrame(), so Vulkan presentation cannot pace OptiX startup.
            // A newly completed buffer may overlap one more render using the
            // other buffer. If Vulkan skipped the previous frame, wait until
            // its pending handoff is submitted before queueing anything else.
            if (!liveResize && !raytracer->isRenderInFlight() && !displayHandoffPending)
            {
                const bool resetAccumulation = firstFrame || scene.isDirty(Accumulation);
                const RenderSettings& renderSettings = scene.getRenderSettings();
                const bool proxyOverdraw = rendersProxyOverdraw(renderSettings);
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
            const auto dispatchViewportOverlay = [&]()
            {
                const RenderSettings& renderSettings = scene.getRenderSettings();
                const float cameraExposure = viewportCamera
                    ? viewportCamera->getCamera()->exposure : 0.0f;
                const bool proxyOverdraw = rendersProxyOverdraw(renderSettings);
                const uint32_t selectedCryptomatteId =
                    scene.getActiveCryptomatteId(
                        viewportPanel->getSelectedGaussianIndex());
                const int bufferVisualization =
                    static_cast<int>(renderSettings.bufferVisualization);
                const glm::mat4 viewProjection = viewportCamera
                    ? viewportCamera->getProjectionMatrix() * viewportCamera->getViewMatrix()
                    : glm::mat4(1.0f);
                if (viewportPanel->showOverlays())
                    viewport->updateBillboards(scene);
                viewport->dispatch(
                    cmd, selectedCryptomatteId,
                    viewProjection,
                    proxyOverdraw ? 0.0f : cameraExposure,
                    proxyOverdraw ? 0 : bufferVisualization,
                    proxyOverdraw ? 0 : renderSettings.tonemappingEnabled,
                    viewportPanel->showOverlays());
                viewportPanel->onComputeFinished(cmd, viewport->getOutputImage());
                displayedCryptomatteId = selectedCryptomatteId;
            };

            if (pendingDisplayFrame.readyValue != 0) {
                displayedRenderValue = pendingDisplayFrame.readyValue;
                dispatchViewportOverlay();
                viewportPanel->setAovImages(
                    raytracer->getOutputCrypto(),
                    raytracer->getOutputPosition());
                renderer->setExternalFrameSync(
                    pendingDisplayFrame.renderReadySemaphore,
                    pendingDisplayFrame.bufferReleasedSemaphore,
                    pendingDisplayFrame.readyValue);
                pendingDisplayFrame = {};
            } else if (displayedRenderValue != 0
                && scene.getActiveCryptomatteId(
                        viewportPanel->getSelectedGaussianIndex())
                    != displayedCryptomatteId)
            {
                dispatchViewportOverlay();
            }

            imGuiManager->renderDrawData(cmd, renderer->getCurrentColorImageView(), renderer->getSwapchainExtent());
            renderer->endFrame();
        }
    }
    context.getDevice().waitIdle();
}
