#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "Scene/Scene.h"
#include <gpu/gpu.hpp>
#include "Materials/MaterialX/MaterialXSceneRuntime.h"
#include "Raytracing/Raytracer.h"
#include "Viewport/Viewport.h"

namespace noorray
{

class NoorRaySession
{
public:
    NoorRaySession();
    explicit NoorRaySession(gpu::SurfaceProvider& surfaceProvider);
    ~NoorRaySession();

    // Publishes the current scene into the native raytracer. Hosts call
    // this after scene import or a batch of geometry edits; replacement is
    // immutable from the dispatcher's point of view.
    void rebuildNativeScene();
    // Publishes scene edits made by the editor after startup. Returns true
    // when a GPU snapshot changed and accumulation must restart.
    bool pollNativeScene();
    void rebuildNativeMaterials();
    // Processes MaterialX work after a scene/runtime completion notification
    // and publishes one immutable GPU snapshot when all changed materials are ready.
    bool processNativeMaterials();
    void updateNativeCamera();
    // Prepares the final viewport texture. Call this before opening a gpu
    // frame; it refreshes AOV handles, resizes the viewport output, and uploads
    // scene overlays when they changed.
    void prepareViewport();
    // Records the final composited viewport image into the current gpu frame,
    // or submits it independently when no frame is open. The output includes
    // AOV visualization, tonemapping, and optional scene billboards.
    void renderViewport(const glm::mat4& viewProjection,
        uint32_t selectedCryptomatteId = ~0u, bool showBillboards = true);
    void renderViewport(uint32_t selectedCryptomatteId = ~0u,
        bool showBillboards = true);

    gpu::ImageHandle outputImageHandle() const;
    gpu::TextureHandle outputTexture() const;
    gpu::TextureHandle outputStorageTexture() const;
    gpu::ImageFormat outputFormat() const;
    uint32_t outputWidth() const;
    uint32_t outputHeight() const;
    std::vector<gpu::float4> readOutput() const;
    // Creates the headless gpu::Device/raytracer pair used by embedded
    // hosts such as Python and Hydra. It is deliberately explicit so merely
    // constructing a scene remains GPU-free.
    void initializeHeadlessRenderer(uint32_t width, uint32_t height,
        bool exportColorMemory = false);
    void shutdownRenderer();

    bool hasRenderer() const noexcept { return raytracer_.has_value(); }
    bool isHeadless() const noexcept { return headless_; }
    Scene& scene() noexcept { return scene_; }
    const Scene& scene() const noexcept { return scene_; }

    // Small host integration surface. The renderer and presentation objects
    // remain owned by this session; consumers only borrow them for their own
    // frame/UI integration.
    gpu::Device& device();
    gpu::Swapchain& swapchain();
    void resize(uint32_t width, uint32_t height);
    void commit();
    void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0);
    double lastDispatchMilliseconds();
    void synchronize();
    gpu::Frame beginFrame();
    void endFrame(gpu::Frame&& frame);
    void copyOutputTo(gpu::ImageHandle target);
    std::vector<std::byte> readColor();
    std::vector<gpu::float4> readBeauty();
    std::vector<std::uint32_t> readCryptomatte();
    std::vector<gpu::float4> readPosition();

    NoorRaySession(const NoorRaySession&) = delete;
    NoorRaySession& operator=(const NoorRaySession&) = delete;

    // Member order is load-bearing: scene-owned GPU resources must be released
    // before the device. Members are destroyed in reverse declaration order.
private:
    std::optional<gpu::Device> device_;
    std::optional<gpu::Swapchain> swapchain_;
    std::optional<Raytracer> raytracer_;
    // Declared after raytracer so it is destroyed first; its inputs are views
    // into the raytracer's AOV resources.
    std::optional<Viewport> viewport_;
    Scene scene_;
    bool headless_{true};
    // Owns the immutable host-side MaterialX -> SVM compilation snapshot
    // consumed by the native raytracer dispatch.
    MaterialXSceneRuntime materialRuntime_;

private:
    gpu::ImageFormat viewportOutputFormat_{gpu::ImageFormat::Rgba32Float};
    bool exportViewportMemory_{};
    RenderSettings appliedRenderSettings{};
    bool renderSettingsInitialized{};
};

}
