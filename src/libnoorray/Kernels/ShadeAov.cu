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

NR_GPU inline void clearAovPixel(
    const OutputSurfaces& output, const uint32_t x, const uint32_t y)
{
    surf2Dwrite(packHalf4(glm::vec3(0.0f), 0.0f), output.position, x * sizeof(ushort4), y);
    surf2Dwrite(InvalidIndex, output.cryptomatte, x * sizeof(uint32_t), y);
}

NR_GPU inline void writeAovPixel(
    const OutputSurfaces& output,
    const uint32_t x,
    const uint32_t y,
    const glm::vec3 position,
    const uint32_t instanceIndex)
{
    surf2Dwrite(packHalf4(position, 1.0f), output.position, x * sizeof(ushort4), y);
    surf2Dwrite(instanceIndex, output.cryptomatte, x * sizeof(uint32_t), y);
}

// Writes the deterministic mesh instance ID and position used by object picking.
// Opacity is ignored: the nearest geometric hit always wins.
NR_GPU_KERNEL void shadeAovKernel(const KernelParams params)
{
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t total = params.frame.width * params.frame.height;
    if (pixel >= total)
        return;

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const HitWorkItem hit = params.queues.aovHitQueue[pixel];

    if (hit.instanceIndex == InvalidIndex || hit.instanceIndex >= params.scene.meshInstanceCount)
    {
        clearAovPixel(params.output, x, y);
        clearAovPixel(params.alternateAovOutput, x, y);
        return;
    }

    const glm::vec3 position = loadSurfacePosition(
        params.scene, hit.instanceIndex, hit.primitiveIndex, hit.attribute0, hit.attribute1);
    writeAovPixel(params.output, x, y, position, hit.instanceIndex);
    writeAovPixel(params.alternateAovOutput, x, y, position, hit.instanceIndex);
}
