#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "Raytracer.h"
#include "Realtime/Accumulator.h"
#include "Realtime/Denoiser.h"
#include "Realtime/FrameContext.h"
#include "Realtime/RadianceCache.h"
#include "Realtime/RenderTargets.h"
#include "Realtime/Restir.h"
#include "Realtime/Upscaler.h"

namespace nr::graphics { struct RealtimeArgs; }

// Interactive biased RGB renderer. It shares the scene/resource contract with
// the spectral renderer but owns a separate, cheaper shader pipeline: meshes
// only, shaded with a fallback material, one path per pixel.
//
// A frame is a fixed sequence of stages around the image pass, each of which
// passes the frame through unchanged in its Off mode:
//
//   Restir::presample -> RadianceCache -> image pass -> Restir::resample
//   -> Denoiser -> composite -> Upscaler -> output AOVs -> Accumulator
//
// The image pass writes RenderTargets; the stages read them. Each stage owns
// its resources and settings translation; this class owns the targets, the
// frame's view and the one decision of when temporal history restarts.
class RealtimeRaytracer final : public Raytracer
{
public:
    RealtimeRaytracer(noorrhi::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory = false);
    ~RealtimeRaytracer() override;

    RaytracerType type() const noexcept override
    {
        return RaytracerType::Realtime;
    }

    bool supportsMeshLights() const noexcept override { return false; }
    bool needsSvmPrograms() const noexcept override { return false; }

    // Rays are traced at the resolution the upscaler mode derives; the
    // upscaler brings the result back to the logical size. Hosts and the
    // viewport composite need the trace resolution to reason about how sharp
    // the AOVs behind the beauty are.
    uint32_t traceWidth() const override;
    uint32_t traceHeight() const override;

    void prepareFrameResources() override;

protected:
    void renderImpl() override;
    void onImageAllocationChanged() override;
    void onLightsUploaded() override;
    void onRenderSettingsApplied(const RenderSettings& settings) override;
    void onMaterialShadersChanged(std::span<const noorrhi::Shader> shaders) override;

private:
    // History restarts whenever any of these changes.
    struct HistoryKey
    {
        Extent render;
        Extent output;
        RadianceCacheMode radianceCache{};
        RealtimeLightingMode lighting{};
        DenoiserMode denoiser{};
        UpscalerMode upscaler{};

        bool operator==(const HistoryKey&) const = default;
    };

    Extent outputExtent() const;
    Extent renderExtent() const;
    // Reallocates everything sized to the render resolution when it moved.
    // Waits for the device when it does.
    void ensureRenderResolution();
    void resizeRenderResolution();
    // Builds this frame's context and fills args->view, then records the
    // frame as the previous one.
    FrameContext beginFrame();

    // The realtime miss and hit stages every ray-tracing pass shares; each
    // pass supplies its own ray generation.
    noorrhi::RayTracingPipelineDesc traceStages;
    noorrhi::Shader imageRaygen;
    RadianceCache radianceCache;
    Restir restir;
    Denoiser denoiser;
    Upscaler upscaler;
    Accumulator accumulator;
    RenderTargets targets;
    noorrhi::ComputePipeline compositePipeline;
    noorrhi::ComputePipeline outputAovsPipeline;
    std::unique_ptr<nr::graphics::RealtimeArgs> args;

    HistoryKey historyKey;
    bool hasHistory{};
    std::array<float, 16> previousWorldToView{};
    std::array<float, 16> previousViewToClip{};
    std::array<float, 2> previousJitter{};
    std::array<float, 3> previousCameraPosition{};
    uint32_t frameIndex{};
};
