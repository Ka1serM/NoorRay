#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "Scene/Scene.h"
#include <noorrhi/noorrhi.hpp>
#include "Materials/MaterialX/MaterialXSceneRuntime.h"
#include "Raytracing/Raytracer.h"
#include "Viewport/Viewport.h"

namespace noorray
{

class NoorRaySession
{
public:
    NoorRaySession();
    // Renders on a device the host owns, typically the one it also presents
    // with. The device must outlive the session. The session never presents:
    // its result is the viewport output texture. onMaterialWorkDone is called
    // from background threads when material compilation finishes, so a host
    // that waits for events knows to call processNativeMaterials().
    NoorRaySession(noorrhi::Device& device, uint32_t width = 1, uint32_t height = 1,
        std::function<void()> onMaterialWorkDone = {});
    ~NoorRaySession();

    // Publishes the current scene into the native raytracer. Hosts call
    // this after scene import or a batch of geometry edits; replacement is
    // immutable from the dispatcher's point of view.
    void rebuildNativeScene();
    // Publishes scene edits made by the editor after startup, including the
    // viewport's light icons. Returns true when a GPU snapshot changed and
    // accumulation must restart.
    bool pollNativeScene();
    void rebuildNativeMaterials();
    // Publishes the materials compiled since the last call and links the
    // renderer's shaders that finished compiling. Returns true when the
    // image changes and accumulation must restart.
    bool processNativeMaterials();
    void updateNativeCamera();
    // Prepares the renderer's images for the current size and settings and
    // the final viewport texture: it may replace images, which waits for the
    // device, so call it before opening a NoorRHI frame. It does not read the
    // scene, so hosts may call it without holding their scene lock. Returns
    // true when the images were replaced and accumulation must restart.
    bool prepareViewport();
    // Records the final composited viewport image into the current NoorRHI frame,
    // or submits it independently when no frame is open. The output includes
    // AOV visualization, tonemapping, and optional scene billboards.
    void renderViewport(const glm::mat4& viewProjection,
        uint32_t selectedCryptomatteId = ~0u, bool showBillboards = true);
    void renderViewport(uint32_t selectedCryptomatteId = ~0u,
        bool showBillboards = true);

    noorrhi::ImageHandle outputImageHandle() const;
    noorrhi::TextureHandle outputTexture() const;
    noorrhi::TextureHandle outputStorageTexture() const;
    noorrhi::ImageFormat outputFormat() const;
    uint32_t outputWidth() const;
    uint32_t outputHeight() const;
    // Allocated size of the output image, at least outputWidth() x
    // outputHeight(). Presenters sample the bottom-left logical rectangle,
    // i.e. UVs scaled by outputWidth() / outputImageWidth().
    uint32_t outputImageWidth() const;
    uint32_t outputImageHeight() const;
    std::vector<noorrhi::float4> readOutput() const;
    // Creates the headless noorrhi::Device/raytracer pair used by embedded
    // hosts such as Python and Hydra. It is deliberately explicit so merely
    // constructing a scene remains GPU-free.
    void initializeHeadlessRenderer(uint32_t width, uint32_t height,
        bool exportColorMemory = false);
    void shutdownRenderer();

    bool hasRenderer() const noexcept { return static_cast<bool>(raytracer_); }
    bool isHeadless() const noexcept { return headless_; }
    Scene& scene() noexcept { return scene_; }
    const Scene& scene() const noexcept { return scene_; }
    Raytracer& raytracer();
    Viewport& viewport();
    const Viewport& viewport() const;

    // The device the renderer runs on: owned by a headless session, borrowed
    // from the host otherwise. Presentation is entirely the host's business.
    noorrhi::Device& device();
    // Sets the render size and fits the camera film to it. The images and
    // the composited viewport follow in prepareViewport(), which replaces
    // the output image and all AOV bindings together, so hosts never need to
    // rebuild ViewportInputs themselves.
    void resizeViewport(uint32_t width, uint32_t height);
    // Allocates the render images at least this large so interactive resizes
    // up to it never wait on the GPU or replace images. Optional; without it
    // the allocation always matches the requested viewport size.
    void reserveViewport(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void commit();
    void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0);
    double lastDispatchMilliseconds();
    void synchronize();
    void copyOutputTo(noorrhi::ImageHandle target);
    std::vector<std::byte> readColor();
    std::vector<noorrhi::float4> readBeauty();
    std::vector<std::uint32_t> readCryptomatte();
    std::vector<noorrhi::float4> readPosition();

    // What lies under one pixel of the viewport output.
    struct ViewportPick
    {
        bool hit = false;
        SceneObjectHandle object{};
    };
    // Picks the object at render pixel (x, y): origin at the bottom-left of the
    // output, x < outputWidth(), y < outputHeight(). Light icons are drawn on
    // top of the render, so they are tested first when includeLights is set
    // (pass whether billboards are shown). Only the one id texel is read back.
    // Call outside an open frame.
    ViewportPick pick(uint32_t x, uint32_t y, bool includeLights = true);
    // World-space surface position at render pixel (x, y), or nothing where
    // the pixel shows the background.
    std::optional<glm::vec3> pickPosition(uint32_t x, uint32_t y);

    NoorRaySession(const NoorRaySession&) = delete;
    NoorRaySession& operator=(const NoorRaySession&) = delete;

    // Member order is load-bearing: scene-owned GPU resources must be released
    // before the device. Members are destroyed in reverse declaration order.
private:
    // Set only for headless sessions, which own their device. Hosts that
    // present pass in their own device instead.
    std::optional<noorrhi::Device> ownedDevice_;
    noorrhi::Device* device_{};
    std::unique_ptr<Raytracer> raytracer_;
    // Declared after raytracer so it is destroyed first; its inputs are views
    // into the raytracer's AOV resources.
    std::optional<Viewport> viewport_;
    // Set by a render() of sample 0 - the host restarted its accumulation -
    // and consumed by the next renderViewport(), whose outline history
    // restarts with it.
    bool accumulationRestarted_{true};
    // The session's last trace and composite. Only they read the scene's GPU
    // data, so publishing edits waits for them rather than for the whole
    // device, which also carries the host's own frames.
    noorrhi::GpuToken lastRender_{};
    noorrhi::GpuToken lastViewport_{};
    Scene scene_;
    Scene::ChangeState appliedSceneChanges_{};
    bool headless_{true};
    // Compiles the scene's MaterialX materials to the renderer's shaders.
    MaterialXSceneRuntime materialRuntime_;

private:
    noorrhi::ImageFormat viewportOutputFormat_{noorrhi::ImageFormat::Rgba32Float};
    bool exportViewportMemory_{};
    RenderSettings appliedRenderSettings{};
    bool renderSettingsInitialized{};
};

}
