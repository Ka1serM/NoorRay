import NoorRHI;
import Restir.Bridge;
import Wavefront.Records;
import Wavefront.PathState;
import Common.Math;
import Common.RayCone;
import Lights.Environment;
import Bsdf.Principled;
import Materials.Material;
import Scene.Camera;
import Scene.SceneSurface;
import Spectral.Wavelengths;

static const uint restirTile = 8u;

uint2 restirPixel(RestirFrame frame, uint index) {
    const uint bandPixels = frame.width * restirTile;
    const uint band = index / bandPixels;
    const uint rows = min(restirTile, frame.height - band * restirTile);
    const uint inBand = index - band * bandPixels;
    const uint column = inBand / (restirTile * rows);
    const uint columns = min(restirTile, frame.width - column * restirTile);
    const uint inTile = inBand - column * restirTile * rows;
    return uint2(column * restirTile + inTile % columns, band * restirTile + inTile / columns);
}

RTXDI_PTInitialSamplingRuntimeParameters initialRuntime() {
    RTXDI_PTInitialSamplingRuntimeParameters runtime = RTXDI_EmptyPTInitialSamplingRuntimeParameters();
    runtime.cameraPos = cameraPosition(restirCamera);
    runtime.prevCameraPos = previousCameraPosition(restirCamera);
    runtime.prevPrevCameraPos = runtime.prevCameraPos;
    return runtime;
}

float3 restirMotion(RestirFrame frame, RAB_Surface surface, uint2 pixel) {
    float2 film;
    if (!projectToFilm(restirCamera, restirCamera.previousWorldToCamera, surface.position, film))
        return float3(0.0, 0.0, 0.0);
    const float2 previousPixel = film * float2(frame.width, frame.height) - 0.5;
    return float3(previousPixel - float2(pixel), length(surface.position - previousCameraPosition(restirCamera)) - surface.depth);
}

void traceRestirPrimary(RestirFrame frame, uint index) {
    const uint2 pixel = restirPixel(frame, index);
    beginRestir(frame, pixel);
    RTXDI_RandomSamplerState random = RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, 1u);
    const float4 u = float4(RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random), RTXDI_GetNextRandom(random));
    const float2 film = (float2(pixel) + u.xy) / float2(frame.width, frame.height);
    const CameraRay camera = generateCameraRay(restirCamera, film, u.zw, 550.0);
    RAB_Surface surface = RAB_EmptySurface();
    surface.view = -camera.direction;
    RayDesc ray;
    ray.Origin = camera.origin;
    ray.Direction = camera.direction;
    ray.TMin = 0.0;
    ray.TMax = floatMax;
    RayHit hit;
    if (camera.valid && camera.weight > 0.0 && traceRestirSurface(ray, instanceMaskCamera, 0u, hit)) {
        const RayCone cone = propagate(cameraRayCone(restirCamera), hit.t);
        const Surface loaded = loadSurface(restirScene, hit.instanceId, hit.primitiveIndex, hit.barycentrics, ray.Direction, hit.t, cone);
        surface = makeRestirSurface(loaded, hit.barycentrics, hit.t, cone);
    }
    frame.surfaces[restirPixelIndex(pixel)] = packRestirSurface(surface);
}

void sampleRestirInitial(RestirFrame frame, uint index) {
    const uint2 pixel = restirPixel(frame, index);
    beginRestir(frame, pixel);
    const RTXDI_PTParameters pt = frame.pt;
    const RAB_Surface surface = RAB_GetGBufferSurface(int2(pixel), false);
    RTXDI_PTReservoir reservoir = RTXDI_EmptyPTReservoir();
    if (RAB_IsSurfaceValid(surface)) {
        RTXDI_PathTracerRandomContext random = RTXDI_InitializePathTracerRandomContext(pixel, frame.runtime.frameIndex, 2u, 3u);
        useRestirWavelengths(RTXDI_GetNextRandom(random.initialRandomSamplerState));
        RAB_PathTracerUserData data = RAB_EmptyPathTracerUserData();
        reservoir = GenerateInitialSamples(pt.initialSampling, initialRuntime(), pt.reconnection, random, surface, data);
    }
    RTXDI_StorePTReservoir(reservoir, pt.reservoirBuffer, pixel, pt.bufferIndices.initialPathTracerOutputBufferIndex);
}

void resampleRestirTemporal(RestirFrame frame, uint index) {
    const uint2 pixel = restirPixel(frame, index);
    beginRestir(frame, pixel);
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_PTTemporalResamplingRuntimeParameters runtime = RTXDI_EmptyPTTemporalResamplingRuntimeParameters();
    runtime.pixelPosition = pixel;
    runtime.reservoirPosition = pixel;
    runtime.motionVector = restirMotion(frame, RAB_GetGBufferSurface(int2(pixel), false), pixel);
    runtime.cameraPos = cameraPosition(restirCamera);
    runtime.prevCameraPos = previousCameraPosition(restirCamera);
    runtime.prevPrevCameraPos = runtime.prevCameraPos;
    bool selectedPrevious = false;
    RAB_PathTracerUserData data = RAB_EmptyPathTracerUserData();
    const RTXDI_PTReservoir reservoir =
        RTXDI_PTTemporalResampling(pt.temporalResampling, runtime, pt.hybridShift, pt.reconnection, frame.runtime, pt.reservoirBuffer,
                                   RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, 4u), pt.bufferIndices, selectedPrevious, data);
    RTXDI_StorePTReservoir(reservoir, pt.reservoirBuffer, pixel, pt.bufferIndices.temporalResamplingOutputBufferIndex);
}

void resampleRestirSpatial(RestirFrame frame, uint index) {
    const uint2 pixel = restirPixel(frame, index);
    beginRestir(frame, pixel);
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_PTSpatialResamplingRuntimeParameters runtime = RTXDI_EmptyPTSpatialResamplingRuntimeParameters();
    runtime.pixelPosition = pixel;
    runtime.reservoirPosition = pixel;
    runtime.cameraPos = cameraPosition(restirCamera);
    runtime.prevCameraPos = previousCameraPosition(restirCamera);
    runtime.prevPrevCameraPos = runtime.prevCameraPos;
    runtime.viewportSize = uint2(frame.width, frame.height);
    bool resampled = false;
    RAB_PathTracerUserData data = RAB_EmptyPathTracerUserData();
    const RTXDI_PTReservoir reservoir =
        RTXDI_PTSpatialResampling(runtime, pt.spatialResampling, pt.hybridShift, pt.reconnection, pt.reservoirBuffer, pt.bufferIndices, frame.runtime,
                                  RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, 5u), resampled, data);
    RTXDI_StorePTReservoir(reservoir, pt.reservoirBuffer, pixel, pt.bufferIndices.spatialResamplingOutputBufferIndex);
}

float3 restirDirectLight(RAB_Surface surface, inout RTXDI_RandomSamplerState random) {
    if (!RAB_IsSurfaceValid(surface))
        return restirWavelengths.toFilm(restirWavelengths.upsample(backgroundRadiance(restirScene, -surface.view)));
    const Surface loaded = loadRestirSurface(surface);
    const BsdfInputs inputs = restirMaterial(loaded);
    float4 radiance = emittedRadiance(inputs, restirWavelengths);
    RestirLightSample light;
    if (sampleRestirLight(surface, makeRestirBsdf(loaded, inputs), random, light))
        radiance += light.radiance;
    return restirWavelengths.toFilm(radiance);
}

void shadeRestir(RestirFrame frame, uint index) {
    const uint2 pixel = restirPixel(frame, index);
    beginRestir(frame, pixel);
    const RTXDI_PTParameters pt = frame.pt;
    RTXDI_RandomSamplerState random = RTXDI_InitRandomSampler(pixel, frame.runtime.frameIndex, 6u);
    useRestirWavelengths(RTXDI_GetNextRandom(random));
    const uint pixelIndex = restirPixelIndex(pixel);
    float3 value = restirDirectLight(RAB_GetGBufferSurface(int2(pixel), false), random);
    const RTXDI_PTReservoir reservoir = RTXDI_LoadPTReservoir(pt.reservoirBuffer, pixel, pt.bufferIndices.finalShadingInputBufferIndex);
    if (RTXDI_IsValidPTReservoir(reservoir)) {
        const Wavelengths wavelengths = sampleWavelengths(restirScene.spectral, reservoir.wavelengthSample, false);
        value += wavelengths.toFilm(reservoir.targetFunction * reservoir.weightSum);
    }
    const FilmRecord film = frame.wavefront[0].film;
    splatRadiance(film, pixelIndex, max(sanitize(value), 0.0));
    atomicAdd(film.frame + pixelIndex * 4 + 3, 1.0);
}
