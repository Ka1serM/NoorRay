#pragma once

#include "Frame.h"

// The realtime renderer's RTXDI (external/RTXDI-Library) state, shared by the
// host and the Slang passes. RTXDI's parameter records are plain 32-bit
// fields in both languages, so they are embedded as they are.
#include "Rtxdi/DI/ReSTIRDIParameters.h"
#include "Rtxdi/GI/ReSTIRGIParameters.h"
#include "Rtxdi/ReGIR/ReGIRParameters.h"

#ifdef __cplusplus
#include <cstddef>
namespace nr::graphics {
#endif

// RenderSettings::realtimeLighting, as the shaders see it.
static const uint RealtimeLightingReSTIRDI = 0u;
static const uint RealtimeLightingReSTIRGI = 1u;

// One primary surface of the realtime G-buffer, as the ReSTIR passes
// reconstruct it. Directions are RTXDI octahedral snorm2x16 words. A zero
// record (viewDepth 0) is background.
struct RealtimeSurface
{
    float3 position;
    // Distance along the camera's view axis; ReSTIR's "linear depth".
    float viewDepth;
    uint normal;
    uint geometricNormal;
    uint view;
    float roughness;
    float3 diffuse;
    // Probability the BSDF sampler picks its diffuse lobe.
    float diffuseProbability;
    float3 specularF0;
    // The camera ray's weight, which the primary pass already applied to the
    // radiance it left for the shading passes.
    float cameraWeight;
    float3 specularF90;
    float padding;
};

// Vose alias table over the local lights, weighted by power, from which the
// presampling pass fills RTXDI's RIS tiles.
struct RealtimeLightAlias
{
    float probability;
    uint alias;
    // Selection pdf of this entry's own light.
    float pdf;
    uint padding;
};

struct RealtimeLighting
{
    // RTXDI buffers. Reservoirs and the G-buffer use RTXDI's block-linear
    // reservoir addressing; the G-buffer is ping-ponged so the temporal passes
    // see the previous frame's surfaces.
    GpuPtr(uint2) risBuffer;
    GpuPtr(RTXDI_PackedDIReservoir) diReservoirs;
    GpuPtr(RTXDI_PackedGIReservoir) giReservoirs;
    GpuPtr(float2) neighborOffsets;
    GpuPtr(RealtimeSurface) surfaces;
    GpuPtr(RealtimeSurface) previousSurfaces;
    GpuPtr(RealtimeLightAlias) localLightAlias;

    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;
    RTXDI_RuntimeParameters runtimeParams;
    RTXDI_Parameters restirDI;
    RTXDI_GIParameters restirGI;
    ReGIR_Parameters regir;
    // World-space resampling for path vertices beyond the G-buffer: the SHaRC
    // update paths and the indirect bounces.
    RTXDI_DIInitialSamplingParameters secondaryInitialSamplingParams;

    // Nonzero once RTXDI owns analytic-light sampling. The environment is
    // always sampled separately.
    uint enabled;
    uint mode;
    // Zero when the previous G-buffer does not describe the previous frame
    // (first frame, resize), which disables temporal reuse.
    uint previousSurfacesValid;
    uint reservoirBlockRowPitch;
};

#ifdef __cplusplus
static_assert(sizeof(RealtimeSurface) == 80);
static_assert(sizeof(RTXDI_PackedGIReservoir) == 32);
static_assert(sizeof(RTXDI_PackedDIReservoir) == 24);
} // namespace nr::graphics
#endif
