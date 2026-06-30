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
#include "UI/RenderSettingsPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/CameraInstance.h"
#include "IO/ExrWriter.h"
#include "Mesh/MeshAsset.h"
#include "Scene/SceneImporter.h"
#include "Raytracing/Raytracer.h"
#include "Vulkan/Viewport.h"
#include "Vulkan/Buffer.h"
#include "Vulkan/Image.h"

// ── GUI constructor ───────────────────────────────────────────────────────────

NoorRay::NoorRay(const int windowWidth, const int windowHeight)
    : context(windowWidth, windowHeight)
    , scene(context)
    , renderer(std::make_unique<Renderer>(context, windowWidth, windowHeight))
    , imGuiManager(std::make_unique<ImGuiManager>(context, renderer->getNumSwapchainImages(), renderer->getColorImageFormat()))
{
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
    imGuiManager->addComponent<ViewportPanel>("Viewport", context, scene,
        viewport->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(),
        raytracer->getWidth(), raytracer->getHeight());
}

// ── Headless constructor ──────────────────────────────────────────────────────

NoorRay::NoorRay(int /*argc*/, char* argv[])
    : context(1, 1, /*headless=*/true)
    , scene(context)
{
    SceneImporter::ImportJsonScene(scene, argv[1]);

    raytracer = std::make_unique<Raytracer>(context, scene);

    viewport = std::make_unique<Viewport>(
        context, raytracer->getWidth(), raytracer->getHeight(),
        raytracer->getOutputColor(0),    raytracer->getOutputColor(1),
        raytracer->getOutputAlbedo(0),   raytracer->getOutputAlbedo(1),
        raytracer->getOutputNormal(0),   raytracer->getOutputNormal(1),
        raytracer->getOutputCrypto(0),   raytracer->getOutputCrypto(1),
        raytracer->getOutputPosition(0), raytracer->getOutputPosition(1),
        vk::Format::eR8G8B8A8Unorm);
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
                    debugPanel->onComputeFinished(raytracer->getGpuTimeMs());
                    if (scene.isDirty(Meshes))   raytracer->updateMeshes();
                    if (scene.isDirty(Textures)) raytracer->updateTextures();
                    else if (scene.isDirty(EnvironmentCdf)) raytracer->updateEnvironmentCdf();
                    if (scene.isDirty(Lights))   raytracer->updateLights();
                    if (scene.isDirty(TLAS))     raytracer->updateTLAS();

                    PushData push{};
                    push.frame            = frame;
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
                    viewport->dispatch(
                        cmd, frameInfo.bufferIndex, scene.getActiveMeshInstanceIndex(),
                        scene.getRenderSettings().exposure,
                        static_cast<int>(scene.getRenderSettings().bufferVisualization),
                        scene.getRenderSettings().tonemappingEnabled);
                    viewportPanel->onComputeFinished(cmd, viewport->getOutputImage());
                    viewportPanel->setAovImages(
                        raytracer->getOutputCrypto(frameInfo.bufferIndex),
                        raytracer->getOutputPosition(frameInfo.bufferIndex));
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

        context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
            raytracer->render(push);
        });
    }
    context.getDevice().waitIdle();
    std::cout << "  100%\n";

    Image& out = raytracer->getOutputColor();
    const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(w) * h * 4 * sizeof(float);
    Buffer staging(context, Buffer::Type::Custom, bytes, nullptr,
                   vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    context.oneTimeSubmit([&](vk::CommandBuffer cmd) {
        out.setImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
        vk::BufferImageCopy region{};
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = vk::Extent3D{w, h, 1};
        cmd.copyImageToBuffer(out.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                              staging.getBuffer(), region);
    });

    const void* mapped = context.getDevice().mapMemory(staging.getMemory(), 0, bytes);
    std::string exrError;
    const bool saved = writeFloatExr(
        outputPath, static_cast<const float*>(mapped), w, h, &exrError);
    context.getDevice().unmapMemory(staging.getMemory());

    if (!saved)
        throw std::runtime_error("Failed to save EXR: " + exrError);

    std::cout << "Saved: " << outputPath << "\n";
}
