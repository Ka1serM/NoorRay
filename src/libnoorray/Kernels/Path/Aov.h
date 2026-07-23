#pragma once

#include <cuda_fp16.h>
#include <optix_device.h>

#include "Camera/Camera.h"
#include "Raytracing/Gpu/Geometry.h"
#include "Raytracing/Path/PathIntegrator.h"
#include "Shading/GaussianAlbedo.h"

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
// in one launch: a beauty-equivalent transparent surface query for data passes
// and an all-opaque Gaussian query for Cryptomatte IDs.
extern "C" __global__ void __raygen__aov()
{
    const uint3 launchIndex = optixGetLaunchIndex();
    const uint32_t x = launchIndex.x;
    const uint32_t y = launchIndex.y;
    const uint32_t pixel = y * params.frame.width + x;
    SampledWavelengths wavelengths = SampledWavelengths::sampleVisible(0.5f);
    const float nx = (static_cast<float>(x) + 0.5f)
        / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(y) + 0.5f)
        / static_cast<float>(params.frame.height) * 2.0f;
    const nr::rstd::optional<CameraRay> cameraRay = params.scene.camera->Dispatch(
        [&](const auto* camera) {
            return camera->generateRay(nx, ny, glm::vec2(0.5f), pixel,
                wavelengths, true);
    });
    const PathIntegrator integrator(params);
    uint32_t gaussianSampleIndex = hashCombine32(pixel, 0u);
    // Keep the opaque-query sentinel unambiguous for the transparent query.
    if (gaussianSampleIndex == OpaqueAovGaussianSample)
        gaussianSampleIndex = 0u;
    const RayHit hit = cameraRay
        ? integrator.intersect(cameraRay->ray, 0.001f, 1000.0f,
            gaussianSampleIndex)
        : RayHit{};
    const RayHit cryptomatteHit = cameraRay
        ? integrator.intersect(cameraRay->ray, 0.001f, 1000.0f,
            OpaqueAovGaussianSample)
        : RayHit{};
    const bool valid = cameraRay.has_value() && hit.instanceIndex != InvalidIndex;
    const bool gaussianHit = valid && hit.primitiveIndex == InvalidIndex;
    const bool cryptomatteValid = cameraRay.has_value()
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
            albedo = gaussianAlbedoRgb(params.scene, cameraRay->ray, hit.instanceIndex,
                SphericalHarmonicsOrder::Degree0);
            position = cameraRay->ray.at(hit.t);
        }
        else
        {
            const Surface surface = Surface::fromHit(params.scene, hit);
            albedo = surface.material->albedoAt(params.scene.textures, surface.uv);
            normal = surface.material->shadingNormalAt(params.scene.textures,
                surface.uv, surface.tangent, surface.normal);
            if (glm::dot(normal, cameraRay->ray.direction) > 0.0f)
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
    surf2Dwrite(packedAlbedo, params.alternateAovOutput.albedo,
        x * sizeof(uchar4), y);
    surf2Dwrite(packedNormal, params.alternateAovOutput.normal,
        x * sizeof(ushort4), y);
    surf2Dwrite(packedPosition, params.alternateAovOutput.position,
        x * sizeof(ushort4), y);
    surf2Dwrite(objectId, params.alternateAovOutput.cryptomatte,
        x * sizeof(uint32_t), y);
}
