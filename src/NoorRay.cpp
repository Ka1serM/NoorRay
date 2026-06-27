#include "NoorRay.h"
#include <cmath>
#include <cstring>
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
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/PerspectiveCamera.h"
#include "Camera/RealisticCamera.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneImporter.h"
#include "Raytracing/ComputeRaytracer.h"
#include "Raytracing/RtxRaytracer.h"
#include "Vulkan/Tonemapper.h"
#include "Vulkan/Buffer.h"
#include "Vulkan/Image.h"

// ── GUI constructor ───────────────────────────────────────────────────────────

NoorRay::NoorRay(const int windowWidth, const int windowHeight, const int renderWidth, const int renderHeight)
    : context(windowWidth, windowHeight)
    , scene(context)
    , renderer(std::make_unique<Renderer>(context, windowWidth, windowHeight))
    , imGuiManager(std::make_unique<ImGuiManager>(context, renderer->getNumSwapchainImages(), renderer->getColorImageFormat()))
{
    const float dpiScale = context.getDPIScale();
    const int scaledW = static_cast<int>(renderWidth  * dpiScale);
    const int scaledH = static_cast<int>(renderHeight * dpiScale);

    if (context.isRtxSupported())
        raytracer = std::make_unique<RtxRaytracer>(scene, scaledW, scaledH);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, scaledW, scaledH);

    tonemapper = std::make_unique<Tonemapper>(context, raytracer->getWidth(), raytracer->getHeight(),
                                             raytracer->getOutputColor(), renderer->getColorImageFormat());

    imGuiManager->addComponent<MainMenuBar>("Menu", context, scene, *imGuiManager);
    imGuiManager->addComponent<DebugPanel>("Debug");
    imGuiManager->addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager->addComponent<SceneGraphPanel>("Scene Graph", scene);
    imGuiManager->addComponent<DetailsPanel>("Details", scene);
    imGuiManager->addComponent<RenderPanel>("Render", context, *raytracer, *renderer, *tonemapper);
    imGuiManager->addComponent<ViewportPanel>("Viewport", context, scene,
        tonemapper->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(),
        raytracer->getWidth(), raytracer->getHeight());

    SceneImporter::ImportGltfScene(scene, "/home/marcel/GitRepositories/NoorRay/assets/slanted_edge_target.glb");

    CameraSettings cameraSettings{};
    cameraSettings.setFocalLength(45.0f);
    cameraSettings.fStop = 2.8f;
    cameraSettings.focusDistance = 39.0f;
    cameraSettings.bokehBias = 2.0f;
    scene.add(std::make_unique<RealisticCamera>(
        scene, "Camera",
        Transform{vec3(0.0f, 0.0f, 39.0f), vec3(0.0f), vec3(1.0f)},
        cameraSettings,
        "/home/marcel/GitRepositories/ROSS/resources/lenses/laikin/Wide1.zmx",
        "/home/marcel/GitRepositories/ROSS/resources/sensors/onsemi_AR0237.json",
        "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/schott.AGF;/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/ohara.AGF"));
}

// ── Headless constructor ──────────────────────────────────────────────────────

NoorRay::NoorRay(int /*argc*/, char* argv[])
    : context(1, 1, /*headless=*/true)
    , scene(context)
{
    SceneImporter::ImportJsonScene(scene, argv[1]);

    uint32_t w = 1920, h = 1080;
    scene.getActiveCamera()->getPreferredRenderSize(w, h);

    if (context.isRtxSupported())
        raytracer = std::make_unique<RtxRaytracer>(scene, w, h);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, w, h);

    tonemapper = std::make_unique<Tonemapper>(context, w, h,
                                             raytracer->getOutputColor(),
                                             vk::Format::eR8G8B8A8Unorm);
}

NoorRay::~NoorRay() = default;

// ── runUi ─────────────────────────────────────────────────────────────────────

void NoorRay::runUi() {
    auto* debugPanel       = dynamic_cast<DebugPanel*>(imGuiManager->getComponent("Debug"));
    auto* viewportPanel    = dynamic_cast<ViewportPanel*>(imGuiManager->getComponent("Viewport"));
    auto* renderPanel      = dynamic_cast<RenderPanel*>(imGuiManager->getComponent("Render"));

    int frame = 0;
    bool isRunning = true, isFullscreen = false, isMoving = false, firstFrame = true;
    mat4 prevCameraMatrix{1.0f};

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

            if (renderer->isComputeWorkFinished()) {
                if (auto* cam = scene.getActiveCamera()) {
                    uint32_t pw = 0, ph = 0;
                    if (cam->getPreferredRenderSize(pw, ph) &&
                        (pw != raytracer->getWidth() || ph != raytracer->getHeight())) {
                        context.getDevice().waitIdle();
                        raytracer->resize(pw, ph);
                        tonemapper->resize(raytracer->getWidth(), raytracer->getHeight(),
                                           raytracer->getOutputColor(), renderer->getColorImageFormat());
                        viewportPanel->resize(raytracer->getWidth(), raytracer->getHeight(),
                                              tonemapper->getOutputImage().getFormat());
                        frame = 0; firstFrame = true;
                    }
                }

                debugPanel->onComputeFinished();
                viewportPanel->onComputeFinished(cmd, tonemapper->getOutputImage());

                if (renderPanel->isSaveRequested()) {
                    renderPanel->executeSave();
                } else {
                    if (scene.isDirty(Meshes))   raytracer->updateMeshes();
                    if (scene.isDirty(Textures)) raytracer->updateTextures();
                    if (scene.isDirty(TLAS))     raytracer->updateTLAS();

                    const int pixelPct = std::max(renderPanel->getPixelSizePercent(),
                                                  viewportPanel->getViewportPixelSizePercent());
                    const EnvironmentSettings& env = scene.getEnvironment().settings;

                    SceneSettings ss{};
                    ss.renderSettings.samples               = renderPanel->getSamples();
                    ss.renderSettings.diffuseBounces        = renderPanel->getDiffuseBounces();
                    ss.renderSettings.specularBounces       = renderPanel->getSpecularBounces();
                    ss.renderSettings.transmissionBounces   = renderPanel->getTransmissionBounces();
                    ss.renderSettings.russianRouletteStartBounce = 3;
                    ss.renderSettings.exposure              = renderPanel->getExposure();
                    ss.renderSettings.transparentBackground = 0;
                    ss.environment = env;
                    const float rotRad = env.rotation * (3.14159265f / 180.0f);
                    ss.environment.rotationSin          = std::sin(rotRad);
                    ss.environment.rotationCos          = std::cos(rotRad);
                    ss.environment.lightingExposureScale = env.lightingExposure;
                    ss.environment.visibleExposureScale  = std::pow(2.0f, env.visibleExposure);
                    ss.environment.maxTextureLod         = 3.0f;

                    PushData push{};
                    push.frame            = frame;
                    push.pixelSizePercent = pixelPct;
                    auto* cam = scene.getActiveCamera();
                    cam->setRenderSize(raytracer->getWidth(), raytracer->getHeight());
                    push.cameraToWorld = cam->getCameraToWorld();
                    ss.camera = cam->getCameraData();
                    cam->populateRealisticCameraSettings(ss.realisticCamera);

                    if (scene.isDirty(RayLut)) raytracer->invalidateCameraRayLut();
                    const bool rayLutChanged = raytracer->prepareCameraRayLut(push, ss);
                    const bool renderChanged = renderPanel->consumeChanged();

                    if (firstFrame || scene.isDirty(Accumulation) || renderChanged || rayLutChanged)
                        frame = 0;
                    else
                        ++frame;

                    push.frame = frame;
                    scene.clearDirtyFlags();
                    firstFrame = false;

                    isMoving = (frame == 0) || memcmp(&push.cameraToWorld, &prevCameraMatrix, sizeof(mat4)) != 0;
                    prevCameraMatrix = push.cameraToWorld;
                    push.isMoving = isMoving ? 1 : 0;

                    if (frame == 0) raytracer->updateSceneSettings(ss);

                    renderer->submitCompute([&](vk::CommandBuffer computeCmd) {
                        raytracer->render(computeCmd, push);
                        tonemapper->dispatch(computeCmd);
                    });
                }
            }

            imGuiManager->render(cmd, renderer->getCurrentColorImageView(), renderer->getSwapchainExtent());
            renderer->endFrame();
        }
    }
    context.getDevice().waitIdle();
}

// ── runCli ────────────────────────────────────────────────────────────────────

void NoorRay::runCli(const int spp, const std::string& outputPath) {
    raytracer->updateMeshes();
    raytracer->updateTextures();
    raytracer->updateTLAS();

    auto* cam = scene.getActiveCamera();
    if (!cam) { std::cerr << "No camera in scene.\n"; return; }

    const uint32_t w = raytracer->getWidth(), h = raytracer->getHeight();
    cam->setRenderSize(w, h);
    cam->update();

    SceneSettings ss{};
    ss.renderSettings.samples                    = spp;
    ss.renderSettings.russianRouletteStartBounce = 3;
    ss.camera = cam->getCameraData();
    cam->populateRealisticCameraSettings(ss.realisticCamera);
    raytracer->updateSceneSettings(ss);

    std::cout << "Rendering " << w << "x" << h << " @ " << spp << " spp\n";

    for (int frame = 0; frame < spp; ++frame) {
        if (frame % std::max(1, spp / 20) == 0)
            std::cout << "  " << (frame * 100 / spp) << "%\r" << std::flush;

        PushData push{};
        push.frame            = frame;
        push.isMoving         = (frame == 0) ? 1 : 0;
        push.pixelSizePercent = 100;
        push.cameraToWorld    = cam->getCameraToWorld();
        raytracer->prepareCameraRayLut(push, ss);

        context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
            raytracer->render(cmd, push);
            tonemapper->dispatch(cmd);
        });
    }
    context.getDevice().waitIdle();
    std::cout << "  100%\n";

    const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(w) * h * 4;
    Buffer staging(context, Buffer::Type::Custom, bytes, nullptr,
                   vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    Image& out = tonemapper->getOutputImage();
    context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
        out.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        vk::BufferImageCopy region{};
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = vk::Extent3D{w, h, 1};
        cmd.copyImageToBuffer(out.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                              staging.getBuffer(), region);
    });

    stbi_write_png(outputPath.c_str(), static_cast<int>(w), static_cast<int>(h),
                   4, staging.getMappedData(), static_cast<int>(w) * 4);

    std::cout << "Saved: " << outputPath << "\n";
}
