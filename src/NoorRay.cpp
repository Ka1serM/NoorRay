#include "NoorRay.h"
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
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/CameraInstance.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneImporter.h"
#include "Raytracing/Raytracer.h"
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

    raytracer = std::make_unique<Raytracer>(context, scene, scaledW, scaledH);

    tonemapper = std::make_unique<Tonemapper>(
        context, raytracer->getWidth(), raytracer->getHeight(),
        raytracer->getOutputImage(0, Aov::Color), raytracer->getOutputImage(1, Aov::Color),
        renderer->getColorImageFormat());

    imGuiManager->addComponent<MainMenuBar>("Menu", context, scene, *imGuiManager);
    imGuiManager->addComponent<DebugPanel>("Debug");
    imGuiManager->addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager->addComponent<SceneGraphPanel>("Scene Graph", scene);
    imGuiManager->addComponent<DetailsPanel>("Details", scene);
    imGuiManager->addComponent<RenderPanel>("Render", context, scene, *raytracer, *renderer, *tonemapper);
    imGuiManager->addComponent<ViewportPanel>("Viewport", context, scene,
        tonemapper->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(),
        raytracer->getWidth(), raytracer->getHeight());

    SceneImporter::ImportGltfScene(scene, "/home/marcel/GitRepositories/NoorRay/assets/slanted_edge_target.glb");

    auto camera = std::make_unique<CameraInstance>(
        scene, "Camera",
        Transform{vec3(0.0f, 0.0f, 12.0f), vec3(0.0f), vec3(1.0f)},
        CameraProjectionType::ThinLens);
    camera->setFocalLength(45.0f);
    camera->getCamera()->DispatchCPU([](auto* managedCamera) {
        managedCamera->fStop = 2.8f;
        managedCamera->focusDistance = 39.0f;
        managedCamera->bokehBias = 2.0f;
    });
    //camera->loadRealisticLens(
    //    "/home/marcel/GitRepositories/ROSS/resources/lenses/laikin/Wide1.zmx",
    //    "/home/marcel/GitRepositories/ROSS/resources/sensors/onsemi_AR0237.json",
    //    "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/schott.AGF;/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/ohara.AGF");
    scene.add(std::move(camera));
}

// ── Headless constructor ──────────────────────────────────────────────────────

NoorRay::NoorRay(int /*argc*/, char* argv[])
    : context(1, 1, /*headless=*/true)
    , scene(context)
{
    SceneImporter::ImportJsonScene(scene, argv[1]);

    const glm::uvec2 resolution = scene.getActiveCamera()->getRenderResolution();
    const uint32_t w = resolution.x;
    const uint32_t h = resolution.y;

    raytracer = std::make_unique<Raytracer>(context, scene, w, h);

    tonemapper = std::make_unique<Tonemapper>(
        context, w, h, raytracer->getOutputImage(0, Aov::Color),
        raytracer->getOutputImage(1, Aov::Color), vk::Format::eR8G8B8A8Unorm);
}

NoorRay::~NoorRay() = default;

// ── runUi ─────────────────────────────────────────────────────────────────────

void NoorRay::runUi() {
    auto* debugPanel       = dynamic_cast<DebugPanel*>(imGuiManager->getComponent("Debug"));
    auto* viewportPanel    = dynamic_cast<ViewportPanel*>(imGuiManager->getComponent("Viewport"));
    auto* renderPanel      = dynamic_cast<RenderPanel*>(imGuiManager->getComponent("Render"));

    int frame = 0;
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
                    const glm::uvec2 resolution = cam->getRenderResolution();
                    if (resolution.x != raytracer->getWidth() || resolution.y != raytracer->getHeight()) {
                        context.getDevice().waitIdle();
                        raytracer->resize(resolution.x, resolution.y);
                        tonemapper->resize(
                            raytracer->getWidth(), raytracer->getHeight(),
                            raytracer->getOutputImage(0, Aov::Color),
                            raytracer->getOutputImage(1, Aov::Color),
                            renderer->getColorImageFormat());
                        viewportPanel->resize(raytracer->getWidth(), raytracer->getHeight(),
                                              tonemapper->getOutputImage().getFormat());
                        frame = 0; firstFrame = true;
                    }
                }

                debugPanel->onComputeFinished(raytracer->getGpuTimeMs());
                if (renderPanel->isSaveRequested()) {
                    renderPanel->executeSave();
                } else {
                    if (scene.isDirty(Meshes))   raytracer->updateMeshes();
                    if (scene.isDirty(Textures)) raytracer->updateTextures();
                    if (scene.isDirty(TLAS))     raytracer->updateTLAS();

                    const int pixelPct = std::max(renderPanel->getPixelSizePercent(),
                                                  viewportPanel->getViewportPixelSizePercent());
                    PushData push{};
                    push.frame            = frame;
                    push.pixelSizePercent = pixelPct;
                    auto* cam = scene.getActiveCamera();

                    if (firstFrame || scene.isDirty(Accumulation))
                        frame = 0;
                    else
                        ++frame;

                    push.frame = frame;
                    scene.clearDirtyFlags();
                    firstFrame = false;

                    push.isMoving = frame == 0 ? 1 : 0;

                    raytracer->render(push);
                    const FrameInfo frameInfo = raytracer->getFrameInfo();
                    tonemapper->dispatch(cmd, frameInfo.bufferIndex);
                    viewportPanel->onComputeFinished(cmd, tonemapper->getOutputImage());
                    viewportPanel->setAovImages(
                        raytracer->getOutputImage(frameInfo.bufferIndex, Aov::Cryptomatte),
                        raytracer->getOutputImage(frameInfo.bufferIndex, Aov::Position));
                    renderer->setExternalFrameSync(
                        frameInfo.renderReadySemaphore,
                        frameInfo.bufferReleasedSemaphore,
                        frameInfo.readyValue);
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
    cam->update();

    scene.getRenderSettings().samples = 1;

    std::cout << "Rendering " << w << "x" << h << " @ " << spp << " spp\n";

    for (int frame = 0; frame < spp; ++frame) {
        if (frame % std::max(1, spp / 20) == 0)
            std::cout << "  " << (frame * 100 / spp) << "%\r" << std::flush;

        PushData push{};
        push.frame            = frame;
        push.isMoving         = (frame == 0) ? 1 : 0;
        push.pixelSizePercent = 100;

        context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
            raytracer->render(push);
            tonemapper->dispatch(cmd, raytracer->getFrameInfo().bufferIndex);
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
