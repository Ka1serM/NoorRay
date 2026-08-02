#pragma once

#include <cuda_fp16.h>
#include <optix.h>
#include <optix_device.h>

#include "Rendering/Camera/Camera.h"
#include "Backend/OptiX/ABI/Geometry.h"
#include "Rendering/MisHeuristic.h"
#include "Rendering/ShadowTerminator.h"
#include "Rendering/Sampling/OwenSobolSampler.h"
#include "Materials/Shading/CompositeBsdf.h"
#include "Materials/Shading/MaterialEvaluation.h"
#include "Materials/SVM/SvmEval.h"
#include "Backend/OptiX/Kernels/AnyHitGaussian.h"
#include "Backend/OptiX/Kernels/ClosestHitGaussian.h"
#include "Backend/OptiX/Kernels/ClosestHitMesh.h"
#include "Backend/OptiX/Kernels/AnalyticLight.h"
#include "Backend/OptiX/Kernels/GaussianProxyOverdraw.h"

// OptiX modules cannot link against CUDA device code from libross. The
// ray-generation kernel owns the camera-lens implementations it dispatches.
#include "libross/imaging/cameralens/raytracing/exitpupil/ExitPupil.cpp"
#include "libross/imaging/cameralens/CameraLens.cpp"
#include "libross/imaging/cameralens/LensSurface.cpp"
#include "libross/imaging/cameralens/raytracing/sequential/SequentialRaytracer.cpp"
#include "libross/imaging/cameralens/raytracing/sequential/FromFilmToWorldRaytracer.cpp"

extern "C"
{
extern __constant__ KernelParams params;
}

NR_GPU inline ushort4 packAovHalf4(const glm::vec3 value, const float w)
{
    return make_ushort4(
        __half_as_ushort(__float2half(value.x)),
        __half_as_ushort(__float2half(value.y)),
        __half_as_ushort(__float2half(value.z)),
        __half_as_ushort(__float2half(w)));
}


namespace nr::aov
{

static constexpr uint32_t CoherenceHintMiss = 0;
static constexpr uint32_t CoherenceHintGaussian = 0x80u;
static constexpr uint32_t CoherenceHintMaterialBase = 0;
static constexpr uint32_t CoherenceHintBits = 8;

NR_GPU inline bool gaussianEnabled()
{
    return (params.frame.visibilityMask & GaussianVisibility) != 0;
}

NR_GPU inline uint32_t coherenceHint(const RayHit& hit)
{
    if (hit.instanceIndex == InvalidIndex)
        return CoherenceHintMiss;
    if (hit.primitiveIndex == InvalidIndex)
        return CoherenceHintGaussian;
    const GpuInstance instance = params.scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = params.scene.meshes[instance.meshIndex];
    const int materialSlot = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    return CoherenceHintMaterialBase
        | (mesh.getMaterialIds()[materialSlot] & 0x7Fu);
}

NR_GPU inline RayHit intersect(
    const Ray& ray,
    const float tMin,
    const float tMax,
    const uint32_t sampleIndex,
    const uint32_t excludedGaussianId = InvalidIndex,
    const bool terminateOnFirstGaussianHit = false,
    const bool reorder = true,
    const bool terminateOnFirstMeshHit = false)
{
    RayHit hit{};
    if (!ray.isValid() || !nr::isFinite(tMin) || !nr::isFinite(tMax)
        || tMin < 0.0f || tMax <= tMin)
        return hit;
    // Any-hit is disabled for traversal-only mesh queries. Beauty-path
    // traces use the separate closest-hit SBT and are launched by
    // tracePrimary() below; shadow/AOV queries continue to use this
    // hit-object path so they can inspect a hit without running shading.
    const unsigned int anyHitFlags =
        (reorder ? OPTIX_RAY_FLAG_DISABLE_ANYHIT : OPTIX_RAY_FLAG_NONE)
        | (terminateOnFirstMeshHit ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT : 0u);
    const bool gaussian = gaussianEnabled();
    if (gaussian)
    {
        RayHit meshHit{};
        if (params.scene.meshInstanceCount > 0)
        {
            optixTraverse(
                params.scene.tlasHandle,
                make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
                make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
                tMin,
                tMax,
                0.0f,
                MeshVisibility,
                anyHitFlags,
                QuerySbtRecordOffset,
                1,
                0);
            // Hit-object state must be read out into meshHit before the
            // gaussian traversal below overwrites the current hit object;
            // the reorder itself is deferred to the end of this function,
            // where the gaussian result is known too and the hint can
            // describe what will actually be shaded.
            if (optixHitObjectIsHit())
            {
                const float hitT = optixHitObjectGetRayTmax();
                if (nr::isFinite(hitT) && hitT >= tMin && hitT <= tMax)
                {
                    meshHit.t = hitT;
                    meshHit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                    meshHit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                    meshHit.instanceIndex = optixHitObjectGetInstanceIndex();
                    meshHit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
                }
            }
        }

        const float gaussianTMax = meshHit.instanceIndex != InvalidIndex
            ? meshHit.t
            : tMax;
        uint32_t payload0 = sampleIndex;
        uint32_t payload1 = __float_as_uint(gaussianTMax);
        uint32_t payload2 = excludedGaussianId;
        const unsigned int gaussianRayFlags = OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
            | (terminateOnFirstGaussianHit
                ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                : OPTIX_RAY_FLAG_NONE);
        optixTraverse(
            params.scene.tlasHandle,
            make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
            make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
            tMin,
            gaussianTMax,
            0.0f,
            GaussianVisibility,
            gaussianRayFlags,
            QuerySbtRecordOffset,
            1,
            0,
            payload0,
            payload1,
            payload2);

        const float gaussianT = __uint_as_float(payload1);
        if (nr::isFinite(gaussianT) && gaussianT >= tMin
            && gaussianT < gaussianTMax)
        {
            hit.instanceIndex = payload2;
            hit.t = gaussianT;
        }
        else if (meshHit.instanceIndex != InvalidIndex)
        {
            hit = meshHit;
        }
    }
    else
    {
        optixTraverse(
            params.scene.tlasHandle,
            make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
            make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
            tMin,
            tMax,
            0.0f,
            MeshVisibility,
            anyHitFlags,
            QuerySbtRecordOffset,
            1,
            0);
        if (optixHitObjectIsHit())
        {
            const float hitT = optixHitObjectGetRayTmax();
            if (nr::isFinite(hitT) && hitT >= tMin && hitT <= tMax)
            {
                hit.t = hitT;
                hit.u = __uint_as_float(optixHitObjectGetAttribute_0());
                hit.v = __uint_as_float(optixHitObjectGetAttribute_1());
                hit.instanceIndex = optixHitObjectGetInstanceIndex();
                hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
            }
        }
    }
    // Reorder once, here, where the full hit is known: threads that
    // scattered onto unrelated materials and directions after the first
    // bounce would otherwise stay locked to whatever lane they started
    // in, serializing the SVM interpreter across every distinct material
    // a warp's rays landed on. Bounce 0 stays coherent for free (adjacent
    // pixels start off hitting similar geometry); every bounce past that
    // is where this earns its keep. See this function's reorder parameter
    // comment for why this must not run for shadow rays. hit is held in
    // thread state, which SER migrates with the thread.
    if (reorder)
        optixReorder(coherenceHint(hit), CoherenceHintBits);
    return hit;
}

}

NR_GPU inline uchar4 packAovUnorm4(const glm::vec3 value, const float w)
{
    const glm::vec3 clamped(
        fminf(fmaxf(value.x, 0.0f), 1.0f),
        fminf(fmaxf(value.y, 0.0f), 1.0f),
        fminf(fmaxf(value.z, 0.0f), 1.0f));
    return make_uchar4(
        static_cast<unsigned char>(clamped.x * 255.0f + 0.5f),
        static_cast<unsigned char>(clamped.y * 255.0f + 0.5f),
        static_cast<unsigned char>(clamped.z * 255.0f + 0.5f),
        static_cast<unsigned char>(w * 255.0f + 0.5f));
}

// AOVs use a stable, center-sampled camera ray. Each thread traces two queries
// inline before its beauty path: a transparent surface query for data passes
// and an all-opaque Gaussian query for Cryptomatte IDs.
namespace nr::aov
{

NR_GPU inline void writeAovs(
    const uint32_t x, const uint32_t y, const uint32_t pixel,
    const nr::rstd::optional<CameraSample>& cameraSample,
    const SampledWavelengths& wavelengths)
{
    uint32_t gaussianSampleIndex = hashCombine32(pixel, 0u);
    // Keep the opaque-query sentinel unambiguous for the transparent query.
    if (gaussianSampleIndex == OpaqueAovGaussianSample)
        gaussianSampleIndex = 0u;
    const RayHit hit = cameraSample
        ? nr::aov::intersect(cameraSample->ray, Ray::DefaultMinDistance,
            Ray::DefaultMaxDistance, gaussianSampleIndex,
            InvalidIndex, false, false)
        : RayHit{};
    const RayHit cryptomatteHit = cameraSample
        ? nr::aov::intersect(cameraSample->ray, Ray::DefaultMinDistance,
            Ray::DefaultMaxDistance, OpaqueAovGaussianSample,
            OpaqueAovQueryMarker, false, false)
        : RayHit{};
    const bool valid = cameraSample.has_value() && hit.instanceIndex != InvalidIndex;
    const bool gaussianHit = valid && hit.primitiveIndex == InvalidIndex;
    const bool cryptomatteValid = cameraSample.has_value()
        && cryptomatteHit.instanceIndex != InvalidIndex;
    const bool cryptomatteGaussian = cryptomatteValid
        && cryptomatteHit.primitiveIndex == InvalidIndex;

    glm::vec3 albedo(0.0f);
    glm::vec3 normal(0.0f);
    glm::vec3 position(0.0f);
    if (valid)
    {
        if (gaussianHit)
        {
            // An albedo pass must remain view-independent, so Gaussian splats
            // use their DC SH coefficient rather than the beauty SH order.
            albedo = gaussianAlbedoRgb(
                params.scene.gaussianShCoeffs,
                params.scene.gaussianShCoefficientCount,
                hit.instanceIndex,
                SphericalHarmonicsOrder::Degree0,
                -cameraSample->ray.direction());
            position = cameraSample->ray.at(hit.t);
        }
        else
        {
            const Surface surface = Surface::fromHit(params.scene, hit);
            const glm::vec3 geometricNormal =
                glm::dot(surface.geometricNormal, -cameraSample->ray.direction()) < 0.0f
                ? -surface.geometricNormal : surface.geometricNormal;
            MaterialEvaluation evaluation{};
            nr::shading::NoorRayCompositeBsdf bsdf(
                geometricNormal, surface.normal, -cameraSample->ray.direction());
            // Raw geometric normal, not the view-oriented one above, which
            // would make this always false (see the beauty closest-hit path).
            const bool exiting = glm::dot(
                -cameraSample->ray.direction(), surface.geometricNormal) < 0.0f;
            bool svmEvaluated = false;
            if (surface.material->svmBytecodeLength != 0) {
                MaterialShadingContext shadingContext{};
                shadingContext.position = surface.position;
                shadingContext.geometricNormal = geometricNormal;
                shadingContext.interpolatedNormal = surface.normal;
                shadingContext.tangent = surface.tangent;
                shadingContext.bitangent = glm::cross(surface.normal, surface.tangent)
                    * surface.tangentSign;
                shadingContext.uv = surface.uv;
                shadingContext.vertexColor = surface.color;
                shadingContext.viewDirection = -cameraSample->ray.direction();
                shadingContext.primitiveId = surface.primitiveIndex;
                const GpuInstance surfaceInstance = params.scene.instances[surface.instanceIndex];
                shadingContext.objectToWorld = surfaceInstance.objectToWorld;
                shadingContext.worldToObject = surfaceInstance.worldToObject;
                shadingContext.normalToWorld = surfaceInstance.normalToWorld;
                svmEvaluated = nr::svm::svmEvalNodes(params.scene,
                    surface.material->svmBytecodeOffset, surface.material->svmBytecodeLength,
                    surface.material->svmTextureOffset, surface.material->svmTextureCount,
                    shadingContext, wavelengths, exiting, evaluation, bsdf);
            }
            if (!svmEvaluated) {
                evaluation.opacity = 1.0f;
                bsdf.prepare();
            }
            albedo = evaluation.albedo;
            normal = bsdf.shadingNormal();
            if (glm::dot(normal, cameraSample->ray.direction()) > 0.0f)
                normal = -normal;
            position = surface.position;
        }
    }
    const uchar4 packedAlbedo = packAovUnorm4(albedo, valid ? 1.0f : 0.0f);
    const ushort4 packedNormal = packAovHalf4(normal,
        valid && !gaussianHit ? 1.0f : 0.0f);
    const ushort4 packedPosition = packAovHalf4(position, valid ? 1.0f : 0.0f);
    const uint32_t objectId = cryptomatteGaussian
        ? params.scene.meshInstanceCount + cryptomatteHit.instanceIndex
        : (cryptomatteValid ? cryptomatteHit.instanceIndex : InvalidIndex);
    surf2Dwrite(packedAlbedo, params.output.albedo, x * sizeof(uchar4), y);
    surf2Dwrite(packedNormal, params.output.normal, x * sizeof(ushort4), y);
    surf2Dwrite(packedPosition, params.output.position, x * sizeof(ushort4), y);
    surf2Dwrite(objectId, params.output.cryptomatte, x * sizeof(uint32_t), y);
    if (params.denoiserAlbedoGuide != nullptr)
        params.denoiserAlbedoGuide[pixel] = make_float3(albedo.x, albedo.y, albedo.z);
    if (params.denoiserNormalGuide != nullptr)
        params.denoiserNormalGuide[pixel] = make_float3(normal.x, normal.y, normal.z);
}

}

namespace nr::path_trace
{

static constexpr uint32_t CoherenceHintMiss = 0;
static constexpr uint32_t CoherenceHintGaussian = 0x80u;
static constexpr uint32_t CoherenceHintMaterialBase = 0;
static constexpr uint32_t CoherenceHintBits = 8;

NR_GPU inline bool gaussianEnabled()
{
    return (params.frame.visibilityMask & GaussianVisibility) != 0;
}

NR_GPU inline uint32_t coherenceHint(const RayHit& hit)
{
    if (hit.instanceIndex == InvalidIndex)
        return CoherenceHintMiss;
    if (hit.primitiveIndex == InvalidIndex)
        return CoherenceHintGaussian;
    const GpuInstance instance = params.scene.instances[hit.instanceIndex];
    const MeshAsset& mesh = params.scene.meshes[instance.meshIndex];
    const int materialSlot = mesh.getFaces()[hit.primitiveIndex].materialIndex;
    return CoherenceHintMaterialBase
        | (mesh.getMaterialIds()[materialSlot] & 0x7Fu);
}

NR_GPU inline void tracePrimary(PathTracePayload& payload)
{
    if (!payload.ray.isValid() || !nr::isFinite(payload.gaussianTMax)
        || payload.gaussianTMax <= Ray::DefaultMinDistance)
    {
        payload.status = PathTraceStatus::Terminate;
        return;
    }
    const bool gaussian = gaussianEnabled();
    // Analytic finite lights are terminal geometry for indirect path rays.
    // Keep them out of camera rays so the existing renderer semantics remain
    // unchanged: analytic lights are sampled by NEE rather than shown as
    // visible camera/AOV geometry. Traversal-only queries use their own masks.
    const bool analyticLights = payload.state->depth > 0;
    const uint8_t visibilityMask =
        (gaussian ? SceneVisibility : MeshVisibility)
        | (analyticLights ? AnalyticLightVisibility : 0u)
        | (payload.includeMeshLightHits ? MeshLightVisibility : 0u);
    uint32_t payload0 = packPathPointer0(&payload);
    uint32_t payload1 = packPathPointer1(&payload);
    uint32_t payload2 = PathTracePayloadMarker;
    optixTraverse(
        params.scene.tlasHandle,
        make_float3(payload.ray.origin().x, payload.ray.origin().y,
            payload.ray.origin().z),
        make_float3(payload.ray.direction().x, payload.ray.direction().y,
            payload.ray.direction().z),
        Ray::DefaultMinDistance, payload.gaussianTMax, 0.0f,
        visibilityMask,
        gaussian ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        payload0, payload1, payload2);

    if (!optixHitObjectIsHit())
    {
        optixInvoke(payload0, payload1, payload2);
        return;
    }

    const uint32_t sbtRecord = optixHitObjectGetSbtRecordIndex();
    const bool reorder = params.frame.serEnabled && payload.state->depth > 0;
    if (sbtRecord == PathGaussianSbtRecord)
    {
        if (reorder)
            optixReorder(CoherenceHintGaussian, CoherenceHintBits);
    }
    else if (sbtRecord == PathAnalyticLightSbtRecord)
    {
        // The custom-light closest-hit program performs the terminal light
        // contribution. It must not be interpreted as a mesh hit here.
    }
    else
    {
        RayHit hit{};
        hit.t = optixHitObjectGetRayTmax();
        hit.u = __uint_as_float(optixHitObjectGetAttribute_0());
        hit.v = __uint_as_float(optixHitObjectGetAttribute_1());
        hit.instanceIndex = optixHitObjectGetInstanceIndex();
        hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
        if (!hit.isValid(Ray::DefaultMinDistance, payload.gaussianTMax))
        {
            payload.status = PathTraceStatus::Terminate;
            return;
        }
        if (reorder)
            optixReorder(coherenceHint(hit), CoherenceHintBits);
    }
    optixInvoke(payload0, payload1, payload2);
}

NR_GPU inline void trace(Ray ray, PathState& state, const uint32_t pixel)
{
    // Every field read by tracePrimary is assigned before each traversal;
    // avoid clearing the payload object once per bounce.
    PathTracePayload payload;
    payload.state = &state;
    payload.pixel = pixel;
    for (uint32_t segment = 0; segment < params.depth; ++segment)
    {
        payload.ray = ray;
        payload.gaussianSampleIndex = gaussianEnabled()
            ? hashCombine32(pixel, segment) : pixel;
        payload.gaussianTMax = Ray::DefaultMaxDistance;
        payload.status = PathTraceStatus::Terminate;
        tracePrimary(payload);
        if (payload.status != PathTraceStatus::Continue)
            return;
        if (!payload.ray.isValid())
        {
            payload.status = PathTraceStatus::Terminate;
            return;
        }
        ray = payload.ray;
    }
}

NR_GPU inline void writeSensorSample(
    const uint32_t pixel,
    const uint32_t x,
    const uint32_t y,
    const PathState& state)
{
    SensorSampleContext ctx{};
    ctx.accumulation = params.accumulation;
    ctx.psfBuckets = params.psfGatherBuckets;
    ctx.width = params.frame.width;
    ctx.height = params.frame.height;
    ctx.totalAccumulated = params.frame.totalAccumulated;
    ctx.alpha = state.alpha;
    ctx.cieX = params.scene.cieX;
    ctx.cieY = params.scene.cieY;
    ctx.cieZ = params.scene.cieZ;
    const Sensor& sensor = params.scene.camera->Dispatch(
        [](const auto* camera) -> const Sensor& { return camera->getSensor(); });
    sensor.Dispatch([&](const auto* concreteSensor) {
        concreteSensor->addSample(pixel, state.radiance, state.wl, 1.0f, ctx,
            [&](const glm::vec4 out) {
                surf2Dwrite(make_float4(out.x, out.y, out.z, out.w),
                    params.output.color, x * sizeof(float4), y);
            });
    });
}

}

extern "C" __global__ void __raygen__pathTrace()
{
    const uint3 launchIndex = optixGetLaunchIndex();
    const uint32_t x = launchIndex.x;
    const uint32_t y = launchIndex.y;
    const uint32_t pixel = y * params.frame.width + x;
    const bool aovQuery = params.frame.aovQuery != 0u;
    const OwenSobolSampler sampler({
        params.frame.totalAccumulated, hashCombine32(x, y)});
    SampledWavelengths wavelengths = SampledWavelengths::sampleVisible(
        sampler.sample1D(SampleDimension::Wavelength));
    const glm::vec2 jitter = params.frame.frameIndex == 0u
        ? glm::vec2(0.5f) : sampler.sample2D(PixelSampleDimensions);
    const glm::vec2 lensSample = sampler.sample2D(LensSampleDimensions);
    const float nx = (static_cast<float>(x) + jitter.x)
        / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(y) + jitter.y)
        / static_cast<float>(params.frame.height) * 2.0f;
    const nr::rstd::optional<CameraSample> cameraSample = params.scene.camera->Dispatch(
        [&](const auto* camera) {
            const float filmY =
                camera->getSensor().origin() == SensorOrigin::LowerLeft
                ? -ny : ny;
            return camera->generateRay(
                nx, filmY, lensSample, pixel, wavelengths);
        });
    // Keep the deterministic AOV query independent from beauty sampling. In
    // particular, its fixed wavelength must never be accumulated into color.
    if (aovQuery)
    {
        SampledWavelengths aovWavelengths =
            SampledWavelengths::sampleVisible(0.5f);
        const float aovNx = (static_cast<float>(x) + 0.5f)
            / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
        const float aovNy = 1.0f - (static_cast<float>(y) + 0.5f)
            / static_cast<float>(params.frame.height) * 2.0f;
        const nr::rstd::optional<CameraSample> aovCameraSample =
            params.scene.camera->Dispatch([&](const auto* camera) {
                const float filmY =
                    camera->getSensor().origin() == SensorOrigin::LowerLeft
                    ? -aovNy : aovNy;
                return camera->generateRay(aovNx, filmY, glm::vec2(0.5f),
                    pixel, aovWavelengths, true);
            });
        nr::aov::writeAovs(
            x, y, pixel, aovCameraSample, aovWavelengths);
    }

    PathState state{};
    state.wl = wavelengths;
    const bool cameraRayValid = cameraSample
        && cameraSample->ray.isValid()
        && nr::isFinite(cameraSample->weight);
    if (cameraRayValid)
    {
        state.throughput = SampledSpectrum(cameraSample->weight);
        state.etaScale = 1.0f;
        nr::path_trace::trace(cameraSample->ray, state, pixel);
    }
    else
        state.alpha = params.scene.camera->Dispatch(
            [](const auto* camera) { return camera->invalidRayIsOpaque() ? 1.0f : 0.0f; });
    state.radiance *= exp2f(fmaxf(-100.0f,
        fminf(100.0f, params.scene.renderSettings.cameraExposure)));
    nr::path_trace::writeSensorSample(pixel, x, y, state);
}
