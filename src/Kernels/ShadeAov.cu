#include <cuda_fp16.h>

#include "Raytracing/Geometry.h"

NR_GPU inline ushort4 packHalf4(const glm::vec3 value, const float w)
{
    return make_ushort4(
        __half_as_ushort(__float2half(value.x)),
        __half_as_ushort(__float2half(value.y)),
        __half_as_ushort(__float2half(value.z)),
        __half_as_ushort(__float2half(w)));
}

// Shades the deterministic AOV primary-ray hit and writes ID/normal/position/albedo
// directly into the output surfaces. Plain RGB, no spectral roundtrip — this is a
// denoiser-guide/debug pass, not physically sampled radiance. Opacity is ignored:
// the nearest geometric hit always wins.
NR_GPU_KERNEL void shadeAovKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    if (pixel >= total)
        return;

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const HitWorkItem hit = params.queues.aovHitQueue[pixel];

    if (hit.primitiveIndex == InvalidIndex)
    {
        surf2Dwrite(make_uchar4(0, 0, 0, 255), params.output.albedo, x * sizeof(uchar4), y);
        surf2Dwrite(packHalf4(glm::vec3(0.0f), 0.0f), params.output.normal, x * sizeof(ushort4), y);
        surf2Dwrite(packHalf4(glm::vec3(0.0f), 0.0f), params.output.position, x * sizeof(ushort4), y);
        surf2Dwrite(InvalidIndex, params.output.cryptomatte, x * sizeof(uint32_t), y);
        return;
    }

    const SurfaceData surface = loadSurface(
        params.scene, hit.instanceIndex, hit.primitiveIndex, hit.attribute0, hit.attribute1);
    if (surface.material == nullptr)
        return;

    const Material& material = *surface.material;
    const glm::vec3 shadingNormal = applyNormalMap(
        material, params.scene.textures, surface.uv, surface.tangent, surface.normal);

    glm::vec3 rgbAlbedo = material.albedo;
    if (material.albedoIndex >= 0)
        rgbAlbedo *= glm::vec3(params.scene.textures[material.albedoIndex].sample(surface.uv));
    rgbAlbedo = glm::clamp(rgbAlbedo, 0.0f, 1.0f);

    surf2Dwrite(make_uchar4(static_cast<unsigned char>(rgbAlbedo.x * 255.0f + 0.5f),
                            static_cast<unsigned char>(rgbAlbedo.y * 255.0f + 0.5f),
                            static_cast<unsigned char>(rgbAlbedo.z * 255.0f + 0.5f), 255),
                params.output.albedo, x * sizeof(uchar4), y);
    surf2Dwrite(packHalf4(shadingNormal, 0.0f), params.output.normal, x * sizeof(ushort4), y);
    surf2Dwrite(packHalf4(surface.position, 1.0f), params.output.position, x * sizeof(ushort4), y);
    surf2Dwrite(surface.objectIndex, params.output.cryptomatte, x * sizeof(uint32_t), y);
}
