#include "NoorRay.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>
#include <SDL3/SDL.h>
#include "UI/DebugPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/SceneGraphPanel.h"
#include "UI/ViewportPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "backends/imgui_impl_sdl3.h"
#include "Camera/PerspectiveCamera.h"
#include "Camera/RealisticCamera.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "Raytracing/ComputeRaytracer.h"
#include "Raytracing/RtxRaytracer.h"
#include "UI/DetailsPanel.h"
#include "UI/EnvironmentPanel.h"
#include "UI/RenderPanel.h"
#include "Vulkan/Tonemapper.h"
#include "Scene/SceneImporter.h"

// ── CDF precomputation ────────────────────────────────────────────────────────
// Returns an RGBA32F image same size as input.
// R = conditional CDF row-by-row, G = marginal CDF (column 0 only), A = PDF
static std::vector<float> computeEnvCdf(const float* hdr, int w, int h)
{
    std::vector<float> out(w * h * 4, 0.0f);

    // Row integrals (weighted by sin(theta))
    std::vector<float> rowIntegrals(h, 0.0f);
    std::vector<std::vector<float>> rowWeights(h, std::vector<float>(w, 0.0f));

    for (int y = 0; y < h; ++y) {
        const float theta   = (y + 0.5f) / float(h) * 3.14159265f;
        const float sinTheta = std::sin(theta);
        for (int x = 0; x < w; ++x) {
            const int src = (y * w + x) * 4;
            const float lum = 0.2126f * hdr[src] + 0.7152f * hdr[src+1] + 0.0722f * hdr[src+2];
            rowWeights[y][x] = lum * sinTheta;
            rowIntegrals[y] += rowWeights[y][x];
        }
    }

    // Total integral for marginal PDF
    const float total = std::accumulate(rowIntegrals.begin(), rowIntegrals.end(), 0.0f);
    const float totalInv = (total > 0.0f) ? 1.0f / total : 0.0f;

    // Build marginal CDF (column 0, G channel)
    float margAcc = 0.0f;
    for (int y = 0; y < h; ++y) {
        margAcc += rowIntegrals[y] * totalInv;
        out[(y * w + 0) * 4 + 1] = margAcc;   // G = marginal CDF
    }

    // Build conditional CDF per row (R channel) and store PDF (A channel)
    for (int y = 0; y < h; ++y) {
        const float theta    = (y + 0.5f) / float(h) * 3.14159265f;
        const float sinTheta = std::max(std::sin(theta), 1e-6f);
        const float rowNorm  = (rowIntegrals[y] > 0.0f) ? 1.0f / rowIntegrals[y] : 0.0f;
        const float margPdf  = rowIntegrals[y] * totalInv;

        float condAcc = 0.0f;
        for (int x = 0; x < w; ++x) {
            const int dst = (y * w + x) * 4;
            condAcc += rowWeights[y][x] * rowNorm;
            out[dst + 0] = condAcc;   // R = conditional CDF

            // PDF = p(u,v) = p_marginal(v) * p_conditional(u|v)
            //             = (rowIntegral / total) * (weight / rowIntegral) * w * h / (2π²sinθ)
            const float pUV  = rowWeights[y][x] * totalInv;
            const float pdf  = pUV * float(w) * float(h) / (2.0f * 3.14159265f * 3.14159265f * sinTheta);
            out[dst + 3] = pdf;       // A = PDF
        }
    }
    return out;
}

NoorRay::~NoorRay() = default;

NoorRay::NoorRay(const int windowWidth, const int windowHeight, const int renderWidth, const int renderHeight)
    : context(windowWidth, windowHeight),
      renderer(context, windowWidth, windowHeight),
      imGuiManager(context, renderer.getNumSwapchainImages(), renderer.getColorImageFormat()),
      scene(context)
{
    // Apply DPI scaling for the raytracer
    const float dpiScale = context.getDPIScale();
    int scaledRenderWidth  = static_cast<int>(static_cast<float>(renderWidth)  * dpiScale);
    int scaledRenderHeight = static_cast<int>(static_cast<float>(renderHeight) * dpiScale);

    if (context.isRtxSupported())
        raytracer = std::make_unique<RtxRaytracer>(scene, scaledRenderWidth, scaledRenderHeight);
    else
        raytracer = std::make_unique<ComputeRaytracer>(scene, scaledRenderWidth, scaledRenderHeight);

    tonemapper = std::make_unique<Tonemapper>(context, raytracer->getWidth(), raytracer->getHeight(), raytracer->getOutputColor(), renderer.getColorImageFormat());

    imGuiManager.addComponent<MainMenuBar>("Menu", context, scene, imGuiManager);
    imGuiManager.addComponent<DebugPanel>("Debug");
    imGuiManager.addComponent<EnvironmentPanel>("Environment", scene);
    imGuiManager.addComponent<SceneGraphPanel>("Scene Graph", scene);
    imGuiManager.addComponent<DetailsPanel>("Details", scene);
    imGuiManager.addComponent<RenderPanel>("Render", context, *raytracer, renderer, *tonemapper);
    imGuiManager.addComponent<ViewportPanel>("Viewport", context, scene, tonemapper->getOutputImage(), raytracer->getOutputCrypto(), raytracer->getOutputPosition(), raytracer->getWidth(), raytracer->getHeight());

/*
    int imgWidth, imgHeight, channels;
    static constexpr unsigned char hdriData[] = {
        #embed "../assets/textures/whipple_creek_regional_park_04_2k.hdr"
    };
    float* hdriPixels = stbi_loadf_from_memory(hdriData, static_cast<int>(sizeof(hdriData)), &imgWidth, &imgHeight, &channels, 4);
    if (!hdriPixels)
        throw std::runtime_error("Failed to load HDR texture from memory" + std::string(stbi_failure_reason()));

    scene.add(Texture(context, "HDRI Sky", hdriPixels, imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));

    // Precompute environment CDF for importance sampling (index 1 after HDRI at index 0)
    std::vector<float> cdfData = computeEnvCdf(hdriPixels, imgWidth, imgHeight);
    scene.add(Texture(context, "HDRI CDF", cdfData.data(), imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));

    stbi_image_free(hdriPixels);
*/

    SceneImporter::ImportGltfScene(scene, "/home/marcel/GitRepositories/NoorRay/assets/slanted_edge_target.glb");

    CameraSettings cameraSettings{};
    cameraSettings.setFocalLength(45.0f);
    cameraSettings.fStop = 2.8f;
    cameraSettings.focusDistance = 39.0f;
    cameraSettings.bokehBias = 2.0f;
    auto cam = std::make_unique<RealisticCamera>(
        scene,
        "Camera",
        Transform{
            vec3(0.0f, 0.0f, 39.0f),
            vec3(0.0f, 0.0f, 0.0f),
            vec3(1.0f)
        },
        cameraSettings,
        "/home/marcel/GitRepositories/ROSS/resources/lenses/laikin/Wide1.zmx",
        "/home/marcel/GitRepositories/ROSS/resources/sensors/onsemi_AR0237.json",
        "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/schott.AGF;/home/marcel/GitRepositories/ROSS/resources/glasscatalogs/ohara.AGF");
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
    CameraData prevCameraData{};
    bool isMoving = false;
    bool hasLastRenderSettings = false;
    SceneSettings lastSceneSettings{};

    while (isRunning) {

        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            imGuiManager.processEvent(event);
            if (event.type == SDL_EVENT_QUIT)
                isRunning = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F11) {
                isFullscreen = !isFullscreen;
                SDL_SetWindowFullscreen(context.getWindow(), isFullscreen);
            }
            if (event.type == SDL_EVENT_WINDOW_RESIZED)
                renderer.notifyResize(event.window.data1, event.window.data2);
        }

        if (renderer.beginFrame())
        {
            const vk::CommandBuffer cmd = renderer.getCurrentCommandBuffer();

            if (renderer.isComputeWorkFinished()) {
                if (auto* activeCamera = scene.getActiveCamera()) {
                    uint32_t preferredWidth = 0;
                    uint32_t preferredHeight = 0;
                    if (activeCamera->getPreferredRenderSize(preferredWidth, preferredHeight) &&
                        (preferredWidth != raytracer->getWidth() || preferredHeight != raytracer->getHeight())) {
                        context.getDevice().waitIdle();
                        raytracer->resize(preferredWidth, preferredHeight);
                        tonemapper->resize(raytracer->getWidth(), raytracer->getHeight(), raytracer->getOutputColor(), renderer.getColorImageFormat());
                        viewportPanel->resize(raytracer->getWidth(), raytracer->getHeight(), tonemapper->getOutputImage().getFormat());
                        frame = 0;
                        hasLastRenderSettings = false;
                    }
                }

                debugPanel->onComputeFinished();
                viewportPanel->onComputeFinished(cmd, tonemapper->getOutputImage());

                if (renderPanel->isSaveRequested())
                    renderPanel->executeSave();
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
                    const int currentPixelSizePercent = std::max(renderPanel->getPixelSizePercent(), viewportPanel->getViewportPixelSizePercent());
                    const EnvironmentSettings& env = environmentPanel->getEnvironmentData();

                    SceneSettings sceneSettings{};
                    sceneSettings.renderSettings.samples               = renderPanel->getSamples();
                    sceneSettings.renderSettings.diffuseBounces        = renderPanel->getDiffuseBounces();
                    sceneSettings.renderSettings.specularBounces       = renderPanel->getSpecularBounces();
                    sceneSettings.renderSettings.transmissionBounces   = renderPanel->getTransmissionBounces();
                    sceneSettings.renderSettings.russianRouletteStartBounce = 3;
                    sceneSettings.renderSettings.exposure              = renderPanel->getExposure();
                    sceneSettings.renderSettings.transparentBackground = 0;
                    sceneSettings.environment = env;
                    const float rotRad = env.rotation * (3.14159265f / 180.0f);
                    sceneSettings.environment.rotationSin           = std::sin(rotRad);
                    sceneSettings.environment.rotationCos           = std::cos(rotRad);
                    sceneSettings.environment.lightingExposureScale = env.lightingExposure;
                    sceneSettings.environment.visibleExposureScale  = std::pow(2.0f, env.visibleExposure);
                    sceneSettings.environment.maxTextureLod         = 3.0f;

                    PushData push{};
                    push.frame            = frame;
                    push.pixelSizePercent = currentPixelSizePercent;
                    auto* activeCamera = scene.getActiveCamera();
                    activeCamera->setRenderSize(raytracer->getWidth(), raytracer->getHeight());
                    push.camera           = activeCamera->getCameraData();
                    activeCamera->populateRealisticCameraSettings(sceneSettings.realisticCamera);

                    const bool renderSettingsChanged =
                        !hasLastRenderSettings ||
                        std::memcmp(&sceneSettings, &lastSceneSettings, sizeof(SceneSettings)) != 0;

                    if (scene.isDirty(Accumulation) || renderSettingsChanged)
                        frame = 0;
                    else
                        frame++;

                    push.frame = frame;
                    scene.clearDirtyFlags();

                    // Detect camera movement for thin-lens / jitter switching
                    isMoving = (frame == 0) ||
                               memcmp(&push.camera, &prevCameraData, sizeof(CameraData)) != 0;
                    prevCameraData = push.camera;
                    push.isMoving = isMoving ? 1 : 0;

                    if (frame == 0)
                        raytracer->updateSceneSettings(sceneSettings);

                    lastSceneSettings = sceneSettings;
                    hasLastRenderSettings = true;

                    renderer.submitCompute([&](const vk::CommandBuffer computeCmd) {
                        raytracer->render(computeCmd, push);
                        tonemapper->dispatch(computeCmd);
                    });
                }
            }

            imGuiManager.render(cmd,renderer.getCurrentColorImageView(),  renderer.getSwapchainExtent());

            renderer.endFrame();
        }
    }
    context.getDevice().waitIdle();
}
