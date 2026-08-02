#include <optix_device.h>

#include "Rendering/Camera/Camera.h"
#include "Backend/OptiX/ABI/SceneData.h"

extern "C"
{
extern __constant__ KernelParams params;
}

// This hit group is used only by the diagnostic SBT. Every front-facing proxy
// entry increments the ray-local count and traversal continues through the
// complete Gaussian TLAS.
extern "C" __global__ void __anyhit__gaussianProxyOverdraw()
{
    optixSetPayload_0(optixGetPayload_0() + 1u);
    optixIgnoreIntersection();
}

static __forceinline__ __device__ float3 magma(const float value)
{
    constexpr float3 colors[] = {
        {0.001462f, 0.000466f, 0.013866f},
        {0.316654f, 0.071690f, 0.485380f},
        {0.716387f, 0.214982f, 0.475290f},
        {0.986700f, 0.535582f, 0.382210f},
        {0.987053f, 0.991438f, 0.749504f},
    };
    const float scaled = fminf(fmaxf(value, 0.0f), 1.0f) * 4.0f;
    const int segment = min(static_cast<int>(scaled), 3);
    const float t = scaled - static_cast<float>(segment);
    const float3 a = colors[segment];
    const float3 b = colors[segment + 1];
    return make_float3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t);
}

extern "C" __global__ void __raygen__gaussianProxyOverdraw()
{
    const uint32_t pixel = optixGetLaunchIndex().x;
    const uint32_t total = params.frame.width * params.frame.height;
    if (pixel >= total)
        return;

    const uint32_t x = pixel % params.frame.width;
    const uint32_t y = pixel / params.frame.width;
    const float nx = (static_cast<float>(x) + 0.5f)
        / static_cast<float>(params.frame.width) * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(y) + 0.5f)
        / static_cast<float>(params.frame.height) * 2.0f;
    SampledWavelengths wavelengths = SampledWavelengths::sampleVisible(0.5f);
    const nr::rstd::optional<CameraSample> cameraSample = params.scene.camera->Dispatch(
        [&](const auto* camera) {
            const float filmY =
                camera->getSensor().origin() == SensorOrigin::LowerLeft
                ? -ny : ny;
            return camera->generateRay(nx, filmY, glm::vec2(0.5f), pixel,
                wavelengths, true);
        });
    if (!cameraSample)
    {
        surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 1.0f),
            params.output.color, x * sizeof(float4), y);
        return;
    }

    uint32_t count = 0;
    const Ray& ray = cameraSample->ray;
    optixTrace(
        params.scene.tlasHandle,
        make_float3(ray.origin().x, ray.origin().y, ray.origin().z),
        make_float3(ray.direction().x, ray.direction().y, ray.direction().z),
        Ray::DefaultMinDistance, Ray::DefaultMaxDistance, 0.0f,
        GaussianVisibility,
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        count);

    const float range = static_cast<float>(max(params.scene.renderSettings.gaussianProxyOverdrawMax, 1));
    const float normalized = log2f(static_cast<float>(count) + 1.0f) / log2f(range + 1.0f);
    const float3 color = magma(normalized);
    surf2Dwrite(make_float4(color.x, color.y, color.z, 1.0f),
        params.output.color, x * sizeof(float4), y);
}
