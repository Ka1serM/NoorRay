#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Shared/Light.h"
#include "Shared/RealtimeArgs.h"

namespace rtxdi { class ImportanceSamplingContext; }

// Analytic light sampling through NVIDIA's RTXDI (external/RTXDI-Library):
// ReSTIR DI on primary surfaces, ReGIR-driven RIS at every later vertex and,
// in ReSTIRGI mode, ReSTIR GI for the primary surfaces' diffuse indirect
// light. SingleSample leaves RTXDI unused: every vertex takes one
// power-weighted light sample in the hit shader.
class Restir
{
public:
    // `stages` holds the realtime miss and hit stages the screen-space passes
    // trace visibility rays through.
    Restir(noorrhi::Device& device, noorrhi::RayTracingPipelineDesc stages);
    // Rebuilds the ray-tracing pipelines around new miss, hit and callable stages.
    void setTraceStages(noorrhi::RayTracingPipelineDesc stages);
    ~Restir();

    void setMode(RealtimeLightingMode mode) { mode_ = mode; }
    RealtimeLightingMode mode() const { return mode_; }

    // Reallocates the reservoirs and G-buffer for this render resolution.
    // The GPU must be idle.
    void resize(Extent render);
    // Local lights are the point, spot and rect records in that order, as
    // RtxdiBridge.slang addresses them in place. The GPU must be idle.
    void uploadLights(std::span<const nr::graphics::PointLight> points,
        std::span<const nr::graphics::SpotLight> spots,
        std::span<const nr::graphics::RectLight> rects,
        std::span<const nr::graphics::DirectionalLight> directionals);

    // Fills args.lighting for this frame. Reads args.view.
    void prepare(const FrameContext& frame, nr::graphics::RealtimeArgs& args);
    // Draws the local lights into RTXDI's RIS tiles and ReGIR cells, which the
    // radiance cache update and the image pass sample from.
    void presample(const nr::graphics::RealtimeArgs& args, nr::graphics::RealtimeRoot root) const;
    // After the image pass: ReSTIR DI and, in ReSTIRGI mode, ReSTIR GI, which
    // add their light to the lobes in the render targets and pack them for
    // the denoiser.
    void resample(const nr::graphics::RealtimeArgs& args, nr::graphics::RealtimeRoot root,
        Extent render) const;

private:
    noorrhi::Device& device_;
    RealtimeLightingMode mode_{RenderSettings{}.realtimeLighting};
    std::unique_ptr<rtxdi::ImportanceSamplingContext> context_;
    noorrhi::ComputePipeline presampleLightsPipeline_;
    noorrhi::ComputePipeline presampleReGIRPipeline_;
    noorrhi::RayTracingPipeline diInitialPipeline_;
    noorrhi::RayTracingPipeline diTemporalPipeline_;
    noorrhi::ComputePipeline diBoilingPipeline_;
    noorrhi::RayTracingPipeline diSpatialPipeline_;
    noorrhi::RayTracingPipeline diShadePipeline_;
    noorrhi::RayTracingPipeline giTemporalPipeline_;
    noorrhi::ComputePipeline giBoilingPipeline_;
    noorrhi::RayTracingPipeline giSpatialPipeline_;
    noorrhi::RayTracingPipeline giShadePipeline_;
    // uint2 entries: light index and inverse source pdf.
    noorrhi::Buffer<std::uint32_t> risBuffer_;
    // RTXDI_PackedDIReservoir (24 bytes) and RTXDI_PackedGIReservoir (32 bytes).
    noorrhi::Buffer<std::uint32_t> diReservoirs_;
    noorrhi::Buffer<std::uint32_t> giReservoirs_;
    noorrhi::Buffer<float> neighborOffsets_;
    // RealtimeSurface records (16 words), current and previous frame.
    std::array<noorrhi::Buffer<std::uint32_t>, 2> surfaces_;
    // RealtimeLightAlias records (4 words) over the local lights.
    noorrhi::Buffer<std::uint32_t> lightAlias_;
    uint32_t localLightCount_{};
    uint32_t infiniteLightCount_{};
    // Largest extent of the local lights, which sizes the ReGIR cells.
    float lightExtent_{1.0f};
    uint32_t frameIndex_{};
    uint32_t surfaceParity_{};
    // Temporal reuse also restarts when the lights or the mode change, or
    // after a resize.
    bool historyValid_{};
    RealtimeLightingMode historyMode_{};
};
