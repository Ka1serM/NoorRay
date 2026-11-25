#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/CameraBuffers.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Render/Denoise/Relax.h"
#include "NoorRay/Render/GpuProfiler.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/Raster/GaussianSplats.h"
#include "NoorRay/Render/Raster/ToneMap.h"
#include "NoorRay/Render/Raster/VisibilityBuffer.h"
#include "NoorRay/Render/RendererSettings.h"
#include "NoorRay/Render/Upscale/DlssRr.h"
#include "NoorRay/Render/Upscale/Fsr.h"
#include "NoorRay/Render/Wavefront/WavefrontPathTracer.h"
#include "NoorRay/Scene/Camera.h"

namespace NoorRay::Render {

//! The one renderer: the settings choose the passes of a frame, and every frame ends in the sRGB display image.
class Renderer {
public:
    //! Without a shared cache the renderer compiles its own pipelines; renderers on one device can share one.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height,
                                                        std::shared_ptr<PipelineCache> pipelines = nullptr);
    [[nodiscard]] std::expected<void, Core::Error> resize(std::uint32_t width, std::uint32_t height);
    void setSettings(const RendererSettings& changed);
    void restartAccumulation() { restartPending = true; }
    //! Motion blur accumulates the steps of the shutter interval, so a scene change then keeps the film.
    void setAccumulatesChanges(const bool accumulates) { accumulatesChanges = accumulates; }
    //! A camera cut: RELAX, FSR and DLSS-RR drop their history at the next frame.
    void resetHistory() { historyPending = true; }
    //! Each traced frame also writes its linear display output to this storage image of display size, which another API shares; null stops it.
    void setSharedOutput(const NoorRHI::Image* image) { sharedOutput = image; }
    //! True when the last record wrote a new frame to the shared output.
    bool wroteSharedOutput() const { return sharedWritten; }
    //! Times the passes and kernels of each frame into the profiler; null stops the timing.
    void setProfiler(GpuProfiler* timing);

    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Scene::Camera& camera);

    const RendererSettings& getSettings() const { return settings; }
    //! The settings of the last frame after validation, which accumulate while a realtime view converges.
    const RendererSettings& getFrameSettings() const { return frameSettings; }
    //! True while a realtime view stands still and accumulates.
    bool isConverging() const { return converging; }
    const std::shared_ptr<PipelineCache>& getPipelines() const { return pipelines; }
    const NoorRHI::Image& getDisplay() const { return display; }
    const Targets& getTargets() const { return pathTracer.getTargets(); }
    std::uint32_t getWidth() const { return display.getWidth(); }
    std::uint32_t getHeight() const { return display.getHeight(); }
    //! The film resolution, below the display resolution while FSR upscales.
    std::uint32_t getRenderWidth() const { return pathTracer.getTargets().getWidth(); }
    std::uint32_t getRenderHeight() const { return pathTracer.getTargets().getHeight(); }
    //! The DLSS-RR inputs, output and call values of the last frame.
    const DlssRr& getDlssRr() const { return dlssRr; }
    //! The camera of the last rendered frame.
    const Shaders::CameraRecord& getCameraRecord() const { return lastCamera; }
    //! FSR writes an image, so the readback of the linear display output goes through this copy to a buffer of display size.
    [[nodiscard]] std::expected<void, Core::Error> recordUpscaledCopy(NoorRHI::CommandList& list, NoorRHI::Buffer<NoorRHI::float4>& destination);
    //! The accumulated samples, or the realtime frames since the last change when the renderer does not accumulate.
    std::uint32_t getSampleCount() const { return frameSettings.accumulate ? pathTracer.getAccumulatedFrames() : realtimeFrames; }
    //! True when accumulation reached the sample limit; later frames only refresh the display.
    bool isComplete() const;
    //! A final render with NRD denoises its film once, in the frame that reaches the sample limit; this denoises a film that stopped before it.
    [[nodiscard]] std::expected<void, Core::Error> recordFilmDenoise(NoorRHI::CommandList& list);
    //! What still compiles in the background and what the frames show until it is ready; empty when nothing does.
    std::string getCompileNotice() const;

private:
    [[nodiscard]] std::expected<void, Core::Error> resizeFilm(const RendererSettings& frame, bool fog);
    enum class DenoisePass { Prepare, Gather, Composite, Blend };
    //! An accumulation with NRD gathers the split of each frame and denoises the film once at the end, as Cycles does.
    bool denoisesFilm(const RendererSettings& frame) const { return frame.accumulate && !converging && frame.denoiser == Denoiser::Nrd; }
    [[nodiscard]] std::expected<void, Core::Error> recordLightingClear(NoorRHI::CommandList& list);
    [[nodiscard]] std::expected<void, Core::Error> recordFilmPasses(NoorRHI::CommandList& list, NoorRHI::Ptr<Shaders::WavefrontFrame> frame);
    [[nodiscard]] std::expected<void, Core::Error> recordDenoise(NoorRHI::CommandList& list, NoorRHI::Ptr<Shaders::WavefrontFrame> frame,
                                                                 const RendererSettings& settings, DenoisePass pass);
    //! The image that FSR or DLSS-RR resolves, or null when the frame has no temporal resolve.
    const NoorRHI::Image* getResolved(const RendererSettings& frame) const;
    //! Copies the image that FSR resolved to the shared output.
    [[nodiscard]] std::expected<void, Core::Error> recordSharedCopy(NoorRHI::CommandList& list, const NoorRHI::Image& resolved);
    [[nodiscard]] std::expected<void, Core::Error> recordPost(NoorRHI::CommandList& list, NoorRHI::Ptr<Shaders::WavefrontFrame> frame,
                                                              const RendererSettings& settings, NoorRHI::float2 jitter);

    NoorRHI::Device* device = nullptr;
    RendererSettings settings;
    RendererSettings frameSettings;
    bool restartPending = true;
    bool accumulatesChanges = false;
    //! The last frame with a change of the image, from which a realtime view waits stillSettle before it converges.
    std::chrono::steady_clock::time_point lastChange;
    bool converging = false;
    WavefrontPathTracer pathTracer;
    VisibilityBuffer visibility;
    GaussianSplats splats;
    //! The splat image of the last traced frame goes over the film, also after accumulation completes.
    bool splatted = false;
    const NoorRHI::Image* sharedOutput = nullptr;
    bool sharedWritten = false;
    ToneMap toneMap;
    Gpu::CameraBuffers camera;
    std::shared_ptr<PipelineCache> pipelines;
    NoorRHI::Image display;
    Relax relax;
    //! True while the film output holds the denoised film; the next traced frame clears it.
    bool filmDenoised = false;
    bool volumes = false;
    Fsr fsr;
    DlssRr dlssRr;
    //! Set for each DLSS-RR runtime that failed; RELAX takes its place until the denoiser setting changes, so the view keeps rendering.
    bool dlssRrFailed = false;
    //! The camera of the previous frame, for the motion of FSR.
    NoorRHI::Buffer<Shaders::CameraRecord> previousCamera;
    Shaders::CameraRecord lastCamera{};
    bool hasLastCamera = false;
    //! RELAX, FSR and DLSS-RR drop their history after a resize, a settings change or a camera cut; a camera move keeps it.
    bool historyPending = true;
    std::chrono::steady_clock::time_point lastFrame;
    std::uint32_t frameIndex = 0;
    std::uint32_t realtimeFrames = 0;
    GpuProfiler* profiler = nullptr;
};

}
