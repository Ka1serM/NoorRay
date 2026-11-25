import NoorRHI;
import Lights.SgLighting;
import Restir.Bridge;
import Wavefront.Records;
import Wavefront.PathState;
import Common.Math;
import Common.Sampling;
import Lights.Lights;
import Lights.Environment;
import Wavefront.SurfaceShading;
import Bsdf.Principled;
import Materials.Closure;
import Spectral.Wavelengths;

static const uint restirDirectSamplePass = 7u;
static const uint restirDirectSpatialPass = 8u;

//! The vertex stamp of this frame: the host counts frames from 0 and the stamp of a cleared vertex is 0.
uint directStamp(RestirFrame frame) { return frame.runtime.frameIndex + 1; }

//! One light candidate of the light tree at the vertex, streamed into the reservoir over the uv square of its light.
void streamDirectCandidate(inout RTXDI_DIReservoir state, inout RAB_LightSample selected, RAB_Surface surface, PrincipledBsdf bsdf,
                           LightReceiver receiver, inout RTXDI_RandomSamplerState random) {
    const float4 u = float4(RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random));
    float selection;
    uint choice;
    const uint light = selectLight(restirScene, restirLightSet(surface), surface.position, receiver, u.w, selection, choice);
    if (light == invalidIndex || !(selection > 0.0)) {
        state.M += 1;
        return;
    }
    float2 uv = u.xy;
    if (light == environmentLight) {
        float density;
        uv = environmentUv(restirScene, sampleEnvironment(restirScene, u.xyz, density));
    }
    //! The reservoir keeps uv in 16 bits, so the candidate is the quantized point, which every later evaluation sees.
    const uint2 quantized = uint2(saturate(uv) * 65535.0);
    uv = float2(quantized) / 65535.0;
    const float sourcePdf = selection * (light == environmentLight ? environmentUvPdf(uv) : 1.0);
    const RAB_LightSample sample = RAB_SamplePolymorphicLight(RAB_LoadLightInfo(rtxdiLight(light), false), surface, uv);
    const float target = lightSampleTarget(sample, surface, bsdf);
    if (!RTXDI_StreamSample(state, rtxdiLight(light), uv, RTXDI_GetNextRandom(random), target, sourcePdf > 0.0 ? 1.0 / sourcePdf : 0.0))
        return;
    state.uvData = quantized.x | quantized.y << 16;
    selected = sample;
}

//! The initial candidates of a pixel, then its temporal resampling; a pixel whose path found no surface this frame clears its surface and reservoir.
void sampleDirect(RestirFrame frame, uint2 pixel) {
    const RestirDirectParameters direct = frame.direct;
    const uint index = restirPixelIndex(pixel);
    const uint surfaceIndex = restirSurfaceIndex(index, false);
    const DirectVertex vertex = frame.vertices[index];
    if (vertex.frame != directStamp(frame)) {
        frame.surfaces[surfaceIndex] = packRestirSurface(RAB_EmptySurface(), invalidIndex);
        RTXDI_StoreDIReservoir(RTXDI_EmptyDIReservoir(), direct.reservoirBuffer, pixel, direct.bufferIndices.temporalResamplingOutputBufferIndex);
        return;
    }
    const RAB_Surface surface = unpackRestirSurface(frame.surfaces[surfaceIndex], surfaceIndex);
    useRestirWavelengths(0.5);
    const LightReceiver receiver = unpackReceiver(vertex.receiver, restirVisibility);
    RTXDI_RandomSamplerState random = RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, restirDirectSamplePass);
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RAB_LightSample selected = RAB_EmptyLightSample();
    const PrincipledBsdf bsdf = restirBsdf(surface);
    for (uint candidate = 0; candidate < direct.candidates; ++candidate)
        streamDirectCandidate(state, selected, surface, bsdf, receiver, random);
    RTXDI_FinalizeResampling(state, 1.0, state.M);
    state.M = 1;
    if (RTXDI_IsValidDIReservoir(state) && !RAB_GetConservativeVisibility(surface, selected))
        RTXDI_StoreVisibilityInDIReservoir(state, float3(0.0, 0.0, 0.0), true);
    if (direct.temporal != 0) {
        int2 temporalPixel;
        state = RTXDI_DITemporalResampling(pixel, surface, state, random, frame.runtime, direct.reservoirBuffer, restirMotion(frame, surface, pixel),
                                           direct.bufferIndices.temporalResamplingInputBufferIndex, direct.temporalResampling, temporalPixel, selected);
    }
    RTXDI_StoreDIReservoir(state, direct.reservoirBuffer, pixel, direct.bufferIndices.temporalResamplingOutputBufferIndex);
}

//! The light of the reservoir sample at the primary vertex of the path, with its own wavelengths and the MIS weight against the BSDF hits of the path.
void shadeDirect(RestirFrame frame, uint index, RAB_Surface surface, RTXDI_DIReservoir reservoir) {
    const DirectVertex vertex = frame.vertices[index];
    if (all(vertex.throughput <= 0.0))
        return;
    const bool terminated = (vertex.flags & directVertexSecondaryTerminated) != 0;
    restirWavelengths =
        (vertex.flags & directVertexRgb) != 0 ? rgbWavelengths() : sampleWavelengths(restirScene.spectral, unpackHero(vertex.hero), terminated);
    const RAB_LightSample sample =
        RAB_SamplePolymorphicLight(RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(reservoir), false), surface, RTXDI_GetDIReservoirSampleUV(reservoir));
    const uint set = restirLightSet(surface);
    if (sample.solidAnglePdf <= 0.0 || !lightReachesBounce(restirScene, sample.light, 0u) || !reachesSet(restirScene, sample.light, set)
        || !RAB_GetConservativeVisibility(surface, sample))
        return;
    const float3 offset = sample.position - surface.position;
    const float distance = length(offset);
    const float3 direction = offset / distance;
    const PrincipledBsdf bsdf = makePrincipled(unpackClosure(frame.closures[surface.closure]), surface.view, surface.normal,
                                               (surface.flags & restirSurfaceBackfacing) != 0, restirWavelengths, sceneEnergyLut(restirScene),
                                               (vertex.flags & directVertexDispersion) != 0);
    float diffuseFraction;
    const BsdfValue value = bsdf.evaluate(direction, diffuseFraction);
    if (all(value.value <= 0.0))
        return;
    const LightReceiver receiver = unpackReceiver(vertex.receiver, restirVisibility);
    const float weight = sample.delta || (vertex.flags & directVertexLast) != 0
                             ? 1.0
                             : powerHeuristic(lightSelection(restirScene, set, sample.light, surface.position, receiver) * sample.solidAnglePdf, value.pdf);
    const float4 contribution = vertex.throughput * value.value * sample.radiance * restirTransmittance(surface.position, direction, distance)
                                * (weight * sample.jacobian * restirLightShare(sample.light, surface, direction, diffuseFraction) * RTXDI_GetDIReservoirInvPdf(reservoir));
    const WavefrontFrame wavefront = frame.wavefront[0];
    const float3 film = filmValue(wavefront, restirWavelengths, contribution, false);
    splatRadiance(wavefront.film, index, film);
    splitPathLight(wavefront.film, index, film, diffuseFraction, direction);
}

//! The spatial resampling of a pixel, whose result the next frame reuses, then its shading.
void resampleDirect(RestirFrame frame, uint2 pixel) {
    const RestirDirectParameters direct = frame.direct;
    const uint index = restirPixelIndex(pixel);
    const uint surfaceIndex = restirSurfaceIndex(index, false);
    const RAB_Surface surface = unpackRestirSurface(frame.surfaces[surfaceIndex], surfaceIndex);
    RTXDI_DIReservoir state =
        RTXDI_LoadDIReservoir(direct.reservoirBuffer, pixel, direct.bufferIndices.spatialResamplingInputBufferIndex);
    if (RAB_IsSurfaceValid(surface)) {
        useRestirWavelengths(0.5);
        RTXDI_RandomSamplerState random = RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, restirDirectSpatialPass);
        RAB_LightSample selected = RAB_EmptyLightSample();
        state = RTXDI_DISpatialResampling(pixel, surface, state, random, frame.runtime, direct.reservoirBuffer,
                                          direct.bufferIndices.spatialResamplingInputBufferIndex, direct.spatialResampling, selected);
    }
    RTXDI_StoreDIReservoir(state, direct.reservoirBuffer, pixel, direct.bufferIndices.spatialResamplingOutputBufferIndex);
    if (RAB_IsSurfaceValid(surface) && RTXDI_IsValidDIReservoir(state))
        shadeDirect(frame, index, surface, state);
}
