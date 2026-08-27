#include "NoorRayUi.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "Backend/Vulkan/Raytracer/RaytracerRenderer.h"
#include "Log.h"
#include "UI/DebugPanel.h"
#include "UI/DetailsPanel.h"
#include "UI/EnvironmentPanel.h"
#include "UI/ImGuiManager.h"
#include "UI/LensViewerPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/MaterialXNodeEditorPanel.h"
#include "UI/RenderSettingsPanel.h"
#include "UI/SceneGraphPanel.h"
#include "UI/ViewportPanel.h"
#include "Rendering/Camera/CameraInstance.h"

using namespace noorray;

NoorRayUi::NoorRayUi(std::string scenePath,
    const uint32_t windowWidth, const uint32_t windowHeight)
    : window(windowWidth, windowHeight), session(window)
{
    if (!scenePath.empty())
        session.scene.load(scenePath);
    if (session.raytracer)
        session.rebuildNativeScene();

    imGuiManager = std::make_unique<ImGuiManager>(window, *session.device,
        session.swapchain->image_count(), session.swapchain->format());
    imGuiManager->addComponent<MainMenuBar>("Menu",
        session.scene, *imGuiManager, std::move(scenePath));
    imGuiManager->addComponent<DebugPanel>("Timings");
    imGuiManager->addComponent<EnvironmentPanel>("Environment", session.scene);
    imGuiManager->addComponent<SceneGraphPanel>("Scene Graph", session.scene);
    imGuiManager->addComponent<DetailsPanel>("Details", session.scene);
    imGuiManager->addComponent<RenderSettingsPanel>("Render Settings", session.scene);
    imGuiManager->addComponent<LensViewerPanel>("Lens Viewer", session.scene);
    imGuiManager->addComponent<MaterialXNodeEditorPanel>(
        "MaterialX Node Editor", session.scene);
    if (session.raytracer)
        imGuiManager->addComponent<ViewportPanel>("Viewport", window,
            session.scene, *session.raytracer);
}

NoorRayUi::~NoorRayUi()
{
    // ViewportPanel removes its ImGui texture descriptor and releases
    // compositor resources during ImGuiManager destruction. The last frame
    // may still reference both, so retire the shared NoorRay/gpu-api queue
    // before any UI-owned GPU state is torn down.
    if (session.raytracer)
    {
        try {
            session.raytracer->device().synchronize();
        } catch (const std::exception& error) {
            LOG_ERROR("GPU synchronization before UI shutdown failed: "
                << error.what());
        }
    }
    imGuiManager.reset();
}

void NoorRayUi::run()
{
    auto* viewportPanel = dynamic_cast<ViewportPanel*>(
        imGuiManager->getComponent("Viewport"));
    auto* timingsPanel = dynamic_cast<DebugPanel*>(
        imGuiManager->getComponent("Timings"));
    uint32_t frameIndex = 0;
    uint32_t accumulatedSamples = 0;
    auto frameStart = std::chrono::steady_clock::now();
    glm::mat4 observedCamera(0.0f);
    bool running = true;
    bool fullscreen = false;
    const char* frameLimitEnvironment = std::getenv("NR_GUI_FRAME_LIMIT");
    const uint32_t diagnosticFrameLimit = frameLimitEnvironment
        ? static_cast<uint32_t>(std::max(std::atoi(frameLimitEnvironment), 0)) : 0u;
    uint32_t renderedFrames = 0u;
    while (running)
    {
        SDL_Event event{};
        while (window.pollEvent(event))
        {
            const bool consumed = viewportPanel
                && viewportPanel->processEvent(event);
            if (!consumed)
                imGuiManager->processEvent(event);
            if (event.type == SDL_EVENT_QUIT)
                running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F11)
            {
                fullscreen = !fullscreen;
                window.setFullscreen(fullscreen);
            }
            if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
                session.swapchain->invalidate();
        }

        // A quit event ends the frame loop immediately. Submitting another
        // frame here used UI state that was already logically closing and
        // widened the teardown race above.
        if (!running)
            break;

        imGuiManager->updateUi();
        if (session.pollNativeScene())
            accumulatedSamples = 0;
        if (session.pollNativeMaterials())
            accumulatedSamples = 0;
        if (const CameraInstance* camera = session.scene.getRenderCamera())
        {
            const glm::mat4 currentCamera = camera->getCamera()->cameraToWorld;
            if (currentCamera != observedCamera)
            {
                observedCamera = currentCamera;
                session.updateNativeCamera();
                accumulatedSamples = 0;
            }
        }
        gpu::Frame frame = session.device->begin_frame(*session.swapchain);
        if (!frame)
            continue;
        uint32_t submittedSamples = 0;
        if (session.raytracer)
        {
            const CameraInstance* camera = session.scene.getRenderCamera();
            const glm::uvec2 resolution = camera
                ? camera->getCamera()->getSensor().resolution()
                : glm::uvec2(frame.width(), frame.height());
            if (session.raytracer->width() != resolution.x
                || session.raytracer->height() != resolution.y)
            {
                session.raytracer->resize(resolution.x, resolution.y);
                accumulatedSamples = 0;
            }
            // Every accumulation reset above lands here with a zeroed counter,
            // so one check restarts the render timer for all of them.
            if (timingsPanel && accumulatedSamples == 0)
                timingsPanel->resetRenderTimer();
            const RenderSettings& settings = session.scene.getRenderSettings();
            const uint32_t maximumSamples = static_cast<uint32_t>(
                std::max(settings.maxSamples, 1));
            const uint32_t samplesThisFrame = std::min(
                static_cast<uint32_t>(std::max(settings.samples, 1)),
                maximumSamples > accumulatedSamples
                    ? maximumSamples - accumulatedSamples : 0u);
            for (uint32_t sample = 0; sample < samplesThisFrame; ++sample)
                // Keep the Owen scramble stable when camera motion resets
                // accumulation. sampleIndex still supplies independent samples;
                // changing both values made the one-sample interactive result
                // flicker even while geometry and camera were unchanged.
                session.raytracer->render(0u,
                    accumulatedSamples + sample);
            submittedSamples = samplesThisFrame;

            if (viewportPanel) {
                viewportPanel->recordPresentation();
            } else {
                session.raytracer->copyColorTo(frame.target());
            }
        }
        imGuiManager->renderDrawData(frame);
        session.device->end_frame(std::move(frame));
        ++renderedFrames;
        if (diagnosticFrameLimit != 0u && renderedFrames >= diagnosticFrameLimit)
            running = false;
        frameIndex += submittedSamples;
        accumulatedSamples = std::min(accumulatedSamples + submittedSamples,
            static_cast<uint32_t>(std::max(
                session.scene.getRenderSettings().maxSamples, 1)));

        const auto frameEnd = std::chrono::steady_clock::now();
        if (timingsPanel)
        {
            // The dispatch timestamp is only resolved once its command buffer
            // has retired, so read it after the frame has been submitted. With
            // no dispatch this frame the query still holds the previous result,
            // hence the sample count gates it rather than the value itself.
            const double raytraceMs = submittedSamples > 0
                ? session.raytracer->lastDispatchMilliseconds() : 0.0;
            timingsPanel->onFrameCompleted(
                std::chrono::duration<double>(frameEnd - frameStart).count(),
                static_cast<float>(raytraceMs),
                static_cast<int>(submittedSamples));
            timingsPanel->setSampleInfo(static_cast<int>(accumulatedSamples),
                std::max(session.scene.getRenderSettings().maxSamples, 1));
        }
        frameStart = frameEnd;
    }
}
