#pragma once

#include "Frame.h"

// The realtime renderer's RTXDI (external/RTXDI-Library) state, shared by the
// host and the Slang passes. RTXDI's parameter records are plain 32-bit
// fields in both languages, so they are embedded as they are.
#include "Rtxdi/DI/ReSTIRDIParameters.h"
#include "Rtxdi/PT/ReSTIRPTParameters.h"
#include "Rtxdi/ReGIR/ReGIRParameters.h"

#ifdef __cplusplus
namespace nr::graphics {
#endif

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
    // The material's closure, packed as in RealtimeHitPayload.
    uint closure[5];
    // The surface's ray differential, packed as in RealtimeHitPayload.
    uint differential[9];
    // The Unreal lighting channels of the surface's instance.
    uint lightingChannels;
};

// Vose alias table over the local lights, weighted by power, from which the
// presampling pass fills RTXDI's RIS tiles.
struct RealtimeLightAlias
{
    float probability;
    uint alias;
    // Selection pdf of this entry's own light.
    float pdf;
};

struct RealtimeLighting
{
    // RTXDI buffers. Reservoirs and the G-buffer use RTXDI's block-linear
    // reservoir addressing; the G-buffer is ping-ponged so the temporal passes
    // see the previous frame's surfaces. The resampling passes run once per
    // surface set, each with its own surfaces and reservoirs here.
    GpuPtr(uint2) risBuffer;
    GpuPtr(RTXDI_PackedDIReservoir) diReservoirs;
    GpuPtr(RTXDI_PackedPTReservoir) ptReservoirs;
    GpuPtr(float2) neighborOffsets;
    GpuPtr(RealtimeSurface) surfaces;
    GpuPtr(RealtimeSurface) previousSurfaces;
    // This frame's surfaces of the translucent layer set, which the lighting
    // pass records alongside `surfaces`.
    GpuPtr(RealtimeSurface) layerSurfaces;
    GpuPtr(RealtimeLightAlias) localLightAlias;

    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;
    RTXDI_RuntimeParameters runtimeParams;
    RTXDI_Parameters restirDI;
    RTXDI_PTParameters restirPT;
    ReGIR_Parameters regir;
    // World-space resampling for surfaces beyond the G-buffer: ReSTIR PT's
    // path vertices.
    RTXDI_DIInitialSamplingParameters secondaryInitialSamplingParams;

    // Zero when the previous G-buffer does not describe the previous frame
    // (first frame, resize), which disables temporal reuse.
    uint previousSurfacesValid;
    uint reservoirBlockRowPitch;
    uint padding;
};

#ifdef __cplusplus
} // namespace nr::graphics
#endif
