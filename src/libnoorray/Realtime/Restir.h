#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Shared/Light.h"
#include "Shared/RealtimeArgs.h"

namespace rtxdi { class ImportanceSamplingContext; }
namespace rtxdi { class ReSTIRPTContext; }

// Light sampling through NVIDIA's RTXDI (external/RTXDI-Library): ReSTIR DI
// for direct light on primary surfaces and ReSTIR PT for the multi-bounce
// paths leaving them, with RTXDI's hybrid shifts and ReGIR-driven RIS at
// every later vertex. Primary surfaces come in two sets, each with its own
// reservoirs: the first opaque surfaces and the nearest fractional layers.
class Restir
{
public:
    explicit Restir(noorrhi::Device& device);
    ~Restir();

    // Reallocates the reservoirs and surfaces for lighting rectangles up to
    // `lighting`, with the translucent layer set only when `layers`. The GPU
    // must be idle.
    void resize(Extent lighting, bool layers);
    // Local lights are the point, spot and rect records in that order, as
    // RtxdiBridge.slang addresses them in place. The GPU must be idle.
    void uploadLights(std::span<const nr::graphics::PointLight> points,
        std::span<const nr::graphics::SpotLight> spots,
        std::span<const nr::graphics::RectLight> rects,
        std::span<const nr::graphics::DirectionalLight> directionals);

    // Fills args.lighting for this frame, for the opaque surface set. Reads
    // args.view.
    void prepare(const FrameContext& frame, nr::graphics::RealtimeArgs& args);
    // `lighting`, as prepare() filled it, pointed at the translucent layer
    // set's surfaces and reservoirs. Only with the layer set allocated.
    nr::graphics::RealtimeLighting layerLighting(
        const nr::graphics::RealtimeLighting& lighting) const;
    // Draws environment directions and local lights into RTXDI's RIS tiles,
    // and the local lights into ReGIR cells, which every pass samples from.
    void presample(const nr::graphics::RealtimeArgs& args, nr::graphics::RealtimeRoot root) const;
    // The passes that replay paths through the materials or trace final
    // visibility, so they are ray-generation stages of the realtime trace
    // pipeline.
    std::vector<noorrhi::Shader> raygens() const;
    // After the lighting pass, which records the surfaces: ReSTIR DI and PT's
    // initial samples and reuse, which shade the lobes and pack them for the
    // denoiser. Each root is one surface set's arguments; the sets share
    // nothing they write, so each step runs for all of them between the same
    // barriers. `tracePipeline` must contain raygens(). Ordered after the
    // lighting pass's trace; the caller orders what reads the lobes after it.
    void resample(const nr::graphics::RealtimeArgs& args,
        std::span<const nr::graphics::RealtimeRoot> surfaceSets, Extent lighting,
        const noorrhi::RayTracingPipeline& tracePipeline) const;

private:
    // One set of surfaces ReSTIR resamples, and its reservoirs.
    struct SurfaceSet
    {
        // RTXDI_PackedDIReservoir (24 bytes).
        noorrhi::Buffer<std::uint32_t> diReservoirs;
        noorrhi::Buffer<std::uint32_t> ptReservoirs;
        // RealtimeSurface records, alternating between frames.
        std::array<noorrhi::Buffer<std::uint32_t>, 2> surfaces;
    };

    SurfaceSet surfaceSet(std::size_t pixels) const;

    noorrhi::Device& device_;
    std::unique_ptr<rtxdi::ImportanceSamplingContext> context_;
    std::unique_ptr<rtxdi::ReSTIRPTContext> ptContext_;
    noorrhi::ComputePipeline presampleLightsPipeline_;
    noorrhi::ComputePipeline presampleReGIRPipeline_;
    noorrhi::ComputePipeline presampleEnvironmentPipeline_;
    noorrhi::Shader initialRaygen_;
    noorrhi::ComputePipeline diTemporalPipeline_;
    noorrhi::ComputePipeline diBoilingPipeline_;
    noorrhi::ComputePipeline diSpatialPipeline_;
    noorrhi::Shader ptTemporalRaygen_;
    noorrhi::ComputePipeline ptBoilingPipeline_;
    noorrhi::Shader ptSpatialRaygen_;
    noorrhi::Shader shadeRaygen_;
    // uint2 entries: light index and inverse source pdf.
    noorrhi::Buffer<std::uint32_t> risBuffer_;
    noorrhi::Buffer<float> neighborOffsets_;
    SurfaceSet opaque_;
    std::optional<SurfaceSet> layers_;
    // RealtimeLightAlias records (3 words) over the local lights.
    noorrhi::Buffer<std::uint32_t> lightAlias_;
    uint32_t localLightCount_{};
    uint32_t infiniteLightCount_{};
    // Largest extent of the local lights, which sizes the ReGIR cells.
    float lightExtent_{1.0f};
    uint32_t frameIndex_{};
    // Index of this frame's surfaces in each SurfaceSet.
    uint32_t surfaceParity_{};
    // Temporal reuse also restarts when the lights change, or after a resize.
    bool historyValid_{};
};
