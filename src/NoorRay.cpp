#include "NoorRay.h"
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
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/Camera.h"
#include "Camera/CameraInstance.h"
#include "Camera/PerspectiveCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/ThinLensCamera.h"
#include "CUDA/rstd/Allocator.h"
#include "IO/BitmapWriter.h"
#include "Log.h"
#include "Mesh/MeshAsset.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneReader.h"
#include "Raytracing/Raytracer.h"
#include "Vulkan/Viewport.h"

// ── GUI constructor ───────────────────────────────────────────────────────────

NoorRay::NoorRay(const int windowWidth, const int windowHeight)
    : context(windowWidth, windowHeight)
    , scene(context)
    , renderer(std::make_unique<Renderer>(context, windowWidth, windowHeight))
    , imGuiManager(std::make_unique<ImGuiManager>(context, renderer->getNumSwapchainImages(), renderer->getColorImageFormat()))
{
    // Defaults to a realistic camera with an aspheric lens so lens-file/tracing issues around
    // aspheres are visible in the Lens Viewer panel right away.
    nr::rstd::allocator<RealisticCamera> cameraAllocator;
    RealisticCamera* realisticCamera = cameraAllocator.allocate(1);
    cameraAllocator.construct(realisticCamera);
    realisticCamera->sensor.setImageSensorPath(
        "/home/marcel/GitRepositories/ROSS/resources/sensors/onsemi_AR0237.json");
    realisticCamera->load(
        "/home/marcel/GitRepositories/ROSS/resources/lenses/canon_automotive_fisheye/canon_automotive_fisheye.zmx",
        "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/schott.AGF;"
        "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/ohara.AGF;"
        "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/misc.agf");
    scene.add(std::make_unique<CameraInstance>(
        scene, "Camera", Transform{glm::vec3(0.0f, 0.0f, 5.0f)}, Camera(realisticCamera)));

    raytracer = std::make_unique<Raytracer>(context, scene);

    viewport = std::make_unique<Viewport>(
        context, raytracer->getWidth(), raytracer->getHeight(),
        raytracer->getOutputColor(0),    raytracer->getOutputColor(1),
        raytracer->getOutputAlbedo(0),   raytracer->getOutputAlbedo(1),
        raytracer->getOutputNormal(0),   raytracer->getOutputNormal(1),
        raytracer->getOutputCrypto(0),   raytracer->getOutputCrypto(1),
        raytracer->getOutputPosition(0), raytracer->getOutputPosition(1),
        renderer->getColorImageFormat());

    imGuiManager->addComponent<MainMenuBar>("Menu", context, scene, *imGuiManager);
    imGuiManager->addComponent<DebugPanel>("Debug");
    imGuiManager->addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager->addComponent<SceneGraphPanel>("Scene Graph", scene);
    imGuiManager->addComponent<DetailsPanel>("Details", scene);
    imGuiManager->addComponent<RenderSettingsPanel>("Render Settings", scene);
    imGuiManager->addComponent<RenderPanel>("Render", context, *raytracer, *renderer, *viewport);
    imGuiManager->addComponent<LensViewerPanel>("Lens Viewer", scene);
    imGuiManager->addComponent<ViewportPanel>("Viewport", context, scene,
        viewport->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(),
        raytracer->getWidth(), raytracer->getHeight());
}

// ── Headless constructor ──────────────────────────────────────────────────────

NoorRay::NoorRay(const std::string& scenePath, const int spp,
                 const std::string& outputPath, const int width, const int height,
                 const bool statsEnabled)
    : context(1, 1, /*headless=*/true)
    , scene(context)
    , m_cliSpp(spp)
    , m_cliOutput(outputPath)
    , m_cliStats(statsEnabled)
{
    if (SceneImporter::IsSceneFile(scenePath))
    {
        SceneReader::Read(scene, scenePath);
    }
    else
    {
        SceneImporter::ImportFile(scene, scenePath);
        nr::rstd::allocator<PerspectiveCamera> alc;
        PerspectiveCamera* pc = alc.allocate(1);
        alc.construct(pc);
        Camera cam(pc);
        cam.setFocalLength(50.0f);
        scene.add(std::make_unique<CameraInstance>(
            scene, "Camera",
            Transform{glm::vec3(0.0f, 2.0f, 5.0f)},
            cam));
    }

    const uint32_t gaussianCount = scene.getGaussianCount();
    LOG_INFO("Scene loaded: " << scenePath << " (" << gaussianCount << " gaussians, "
             << scene.getMeshAssets().size() << " meshes)");
    raytracer = std::make_unique<Raytracer>(context, scene);
}

NoorRay::~NoorRay() = default;

// ── runUi ─────────────────────────────────────────────────────────────────────

void NoorRay::runUi() {
    auto* debugPanel       = dynamic_cast<DebugPanel*>(imGuiManager->getComponent("Debug"));
    auto* viewportPanel    = dynamic_cast<ViewportPanel*>(imGuiManager->getComponent("Viewport"));
    auto* renderPanel      = dynamic_cast<RenderPanel*>(imGuiManager->getComponent("Render"));

    raytracer->setTimingEnabled(true);

    int frame = 0;
    int submittedSamples = 0;
    bool renderComplete = false;
    uint64_t displayedRenderValue = 0;
    bool isRunning = true, isFullscreen = false, firstFrame = true;

    while (isRunning) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            imGuiManager->processEvent(event);
            if (event.type == SDL_EVENT_QUIT)
                isRunning = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F11) {
                isFullscreen = !isFullscreen;
                SDL_SetWindowFullscreen(context.getWindow(), isFullscreen);
            }
            if (event.type == SDL_EVENT_WINDOW_RESIZED)
                renderer->notifyResize(event.window.data1, event.window.data2);
        }

        if (renderer->beginFrame()) {
            const vk::CommandBuffer cmd = renderer->getCurrentCommandBuffer();

            {
                if (auto* cam = scene.getActiveCamera()) {
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

                if (!scene.getActiveCamera()) {
                    frame = 0;
                    firstFrame = true;
                } else if (renderPanel->isSaveRequested()) {
                    renderPanel->executeSave();
                } else {
                    const FrameInfo completedFrame = raytracer->getFrameInfo();
                    if (raytracer->isFrameReady()
                        && completedFrame.readyValue > displayedRenderValue)
                    {
                        displayedRenderValue = completedFrame.readyValue;
                        const RenderSettings& renderSettings = scene.getRenderSettings();
                        const bool proxyOverdraw =
                            renderSettings.gaussianProxyOverdrawVisualization != 0;
                        viewport->dispatch(
                            cmd, completedFrame.bufferIndex, scene.getActiveMeshInstanceIndex(),
                            proxyOverdraw ? 0.0f : renderSettings.exposure,
                            proxyOverdraw ? 0 : static_cast<int>(renderSettings.bufferVisualization),
                            proxyOverdraw ? 0 : renderSettings.tonemappingEnabled);
                        viewportPanel->onComputeFinished(cmd, viewport->getOutputImage());
                        viewportPanel->setAovImages(
                            raytracer->getOutputCrypto(completedFrame.bufferIndex),
                            raytracer->getOutputPosition(completedFrame.bufferIndex));
                        renderer->setExternalFrameSync(
                            completedFrame.renderReadySemaphore,
                            completedFrame.bufferReleasedSemaphore,
                            completedFrame.readyValue);
                    }

                    // Keep at most one CUDA render in flight. ImGui continues to
                    // present the last completed viewport image while it runs.
                    if (!raytracer->isRenderInFlight())
                    {
                        if (scene.isDirty(Meshes))   raytracer->updateMeshes();
                        if (scene.isDirty(Textures)) raytracer->updateTextures();
                        else if (scene.isDirty(EnvironmentCdf)) raytracer->updateEnvironmentCdf();
                        if (scene.isDirty(Lights))   raytracer->updateLights();
                        if (scene.isDirty(TLAS))     raytracer->updateTLAS();

                        const bool resetAccumulation = firstFrame || scene.isDirty(Accumulation);
                        const int spp = std::max(1, scene.getRenderSettings().samples);
                        const int maxSamples = std::max(1, scene.getRenderSettings().maxSamples);

                        if (resetAccumulation) {
                            frame = 0;
                            submittedSamples = 0;
                            renderComplete = false;
                        }

                        if (renderComplete) {
                            debugPanel->setSampleInfo(submittedSamples, maxSamples);
                        } else {
                            if (!resetAccumulation)
                                ++frame;

                            scene.clearDirtyFlags();
                            firstFrame = false;
                            raytracer->render(PushData{.frame = frame});
                            // render() harvests the completed frame's CUDA events
                            // before asynchronously submitting the next frame.
                            debugPanel->onComputeFinished(raytracer->getGpuTimeMs());

                            submittedSamples = std::min((frame + 1) * spp, maxSamples);
                            renderComplete = submittedSamples >= maxSamples;
                            debugPanel->setSampleInfo(
                                submittedSamples, maxSamples);
                        }
                    }
                }
            }

            imGuiManager->render(cmd, renderer->getCurrentColorImageView(), renderer->getSwapchainExtent());
            renderer->endFrame();
        }
    }
    context.getDevice().waitIdle();
}

// ── runCli ────────────────────────────────────────────────────────────────────

void NoorRay::runCli() {
    if (m_cliSpp <= 0)
        throw std::invalid_argument("Samples per pixel must be greater than zero");

    LOG_INFO("Rendering " << raytracer->getWidth() << "x" << raytracer->getHeight()
             << " @ " << m_cliSpp << " spp");
    raytracer->setAovEnabled(false);
    raytracer->setStatsEnabled(m_cliStats);
    raytracer->setTimingEnabled(m_cliStats);
    const Bitmap bitmap = raytracer->renderOffline(static_cast<uint32_t>(m_cliSpp));

    std::string writeError;
    const bool saved = BitmapWriter::write(m_cliOutput, bitmap, {}, &writeError);

    if (!saved)
        throw std::runtime_error("Failed to save bitmap: " + writeError);

    LOG_INFO("Saved: " << m_cliOutput);
    if (m_cliStats)
        raytracer->printKernelStats();
}
