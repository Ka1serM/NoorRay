#include "NoorRayUi.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include "Raytracing/Raytracer.h"
#include "Logging/Log.h"
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
#include "Camera/CameraInstance.h"

NoorRayUi::NoorRayUi(std::string scenePath,
    const uint32_t windowWidth, const uint32_t windowHeight)
    : window(windowWidth, windowHeight), session(window)
{
    if (!scenePath.empty())
        session.scene().load(scenePath);
    if (session.hasRenderer())
        session.rebuildNativeScene();

    imGuiManager = std::make_unique<ImGuiManager>(window, session.device(),
        session.swapchain().image_count(), session.swapchain().format());
    imGuiManager->addComponent<MainMenuBar>("Menu",
        session.scene(), *imGuiManager, std::move(scenePath));
    debugPanel = imGuiManager->addComponent<DebugPanel>("Timings");
    imGuiManager->addComponent<EnvironmentPanel>("Environment", session.scene());
    imGuiManager->addComponent<SceneGraphPanel>("Scene Graph", session.scene());
    imGuiManager->addComponent<DetailsPanel>("Details", session.scene());
    imGuiManager->addComponent<RenderSettingsPanel>("Render Settings", session.scene());
    imGuiManager->addComponent<LensViewerPanel>("Lens Viewer", session.scene());
    imGuiManager->addComponent<MaterialXNodeEditorPanel>(
        "MaterialX Node Editor", session.scene());
    if (session.hasRenderer())
        viewportPanel = imGuiManager->addComponent<ViewportPanel>("Viewport", window,
            session);
}

NoorRayUi::~NoorRayUi()
{
    // ViewportPanel removes its ImGui texture descriptor and releases
    // viewport resources during ImGuiManager destruction. The last frame
    // may still reference both, so retire the shared NoorRay/gpu-api queue
    // before any UI-owned GPU state is torn down.
    if (session.hasRenderer())
    {
        try {
            session.synchronize();
        } catch (const std::exception& error) {
            NR_LOG_ERROR("GPU synchronization before UI shutdown failed: "
                << error.what());
        }
    }
    imGuiManager.reset();
}

void NoorRayUi::run()
{
    uint32_t frameIndex = 0;
    uint32_t accumulatedSamples = 0;
    auto frameStart = std::chrono::steady_clock::now();
    bool running = true;
    bool fullscreen = false;
    const char* frameLimitEnvironment = std::getenv("NR_GUI_FRAME_LIMIT");
    const uint32_t diagnosticFrameLimit = frameLimitEnvironment
        ? static_cast<uint32_t>(std::max(std::atoi(frameLimitEnvironment), 0)) : 0u;
    uint32_t renderedFrames = 0u;
    const auto maxSamples = [&]() {
        return static_cast<uint32_t>(std::max(
            session.scene().getRenderSettings().maxSamples, 1));
    };
    const auto processEvent = [&](const SDL_Event& event) {
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
            session.swapchain().invalidate();
    };
    while (running)
    {
        SDL_Event event{};
        const bool rendering = session.hasRenderer()
            && accumulatedSamples < maxSamples();
        const bool interacting = viewportPanel
            && viewportPanel->needsContinuousRedraw();
        if (!rendering && !interacting)
        {
            if (!window.waitEvent(event))
                break;
            if (event.type != 0)
                processEvent(event);
        }
        while (window.pollEvent(event))
            processEvent(event);

        // A quit event ends the frame loop immediately. Submitting another
        // frame here used UI state that was already logically closing and
        // widened the teardown race above.
        if (!running)
            break;

        imGuiManager->updateUi();
        if (session.processNativeMaterials())
            accumulatedSamples = 0;
        if (session.pollNativeScene())
            accumulatedSamples = 0;
        // Finish resource publication after UI edits, so all changes made in
        // one event/UI pass are visible in the same presentation frame.
        if (session.hasRenderer()) {
            if (const CameraInstance* camera = session.scene().getRenderCamera()) {
                const auto resolution = camera->getCamera()->getSensor().resolution();
                if (session.outputWidth() != resolution.x
                    || session.outputHeight() != resolution.y) {
                    session.resize(resolution.x, resolution.y);
                    accumulatedSamples = 0;
                }
            }
            if (viewportPanel) viewportPanel->preparePresentation();
        }
        if (session.hasRenderer()) session.commit();
        gpu::Frame frame = session.beginFrame();
        if (!frame)
            continue;
        uint32_t submittedSamples = 0;
        if (session.hasRenderer())
        {
            // Every accumulation reset above lands here with a zeroed counter,
            // so one check restarts the render timer for all of them.
            if (debugPanel && accumulatedSamples == 0)
                debugPanel->resetRenderTimer();
            const RenderSettings& settings = session.scene().getRenderSettings();
            const uint32_t maximumSamples = maxSamples();
            const uint32_t samplesThisFrame = std::min(
                static_cast<uint32_t>(std::max(settings.samples, 1)),
                maximumSamples > accumulatedSamples
                    ? maximumSamples - accumulatedSamples : 0u);
            for (uint32_t sample = 0; sample < samplesThisFrame; ++sample)
                // Keep the Owen scramble stable when camera motion resets
                // accumulation. sampleIndex still supplies independent samples;
                // changing both values made the one-sample interactive result
                // flicker even while geometry and camera were unchanged.
                session.render(0u,
                    accumulatedSamples + sample);
            submittedSamples = samplesThisFrame;

            if (viewportPanel) {
                viewportPanel->recordPresentation();
            } else {
                // The session's public output is the final viewport image,
                // including the configured composite and scene overlays.
                session.renderViewport(~0u, false);
                session.copyOutputTo(frame.target());
            }
        }
        imGuiManager->renderDrawData(frame);
        session.endFrame(std::move(frame));
        ++renderedFrames;
        if (diagnosticFrameLimit != 0u && renderedFrames >= diagnosticFrameLimit)
            running = false;
        frameIndex += submittedSamples;
        accumulatedSamples = std::min(accumulatedSamples + submittedSamples,
            static_cast<uint32_t>(std::max(
                session.scene().getRenderSettings().maxSamples, 1)));

        const auto frameEnd = std::chrono::steady_clock::now();
        if (debugPanel)
        {
            // The dispatch timestamp is only resolved once its command buffer
            // has retired, so read it after the frame has been submitted. With
            // no dispatch this frame the query still holds the previous result,
            // hence the sample count gates it rather than the value itself.
            const double raytraceMs = submittedSamples > 0
                ? session.lastDispatchMilliseconds() : 0.0;
            debugPanel->onFrameCompleted(
                std::chrono::duration<double>(frameEnd - frameStart).count(),
                static_cast<float>(raytraceMs),
                static_cast<int>(submittedSamples));
            debugPanel->setSampleInfo(static_cast<int>(accumulatedSamples),
                std::max(session.scene().getRenderSettings().maxSamples, 1));
        }
        frameStart = frameEnd;
    }
}
