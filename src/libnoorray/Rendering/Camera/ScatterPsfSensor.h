#pragma once

#include "Rendering/Camera/Sensor.h"
#include "Backend/CUDA/rstd/UniquePtr.h"

#include <cuda_runtime_api.h>

#include "libross/imaging/interpolatedpsfgrid/InterpolatedPsfGrid.h"

#include <memory>
#include <string>

class ScatterPsfSensor : public Sensor::Type<ScatterPsfSensor, RectangularSensor> {
public:
    nr::rstd::unique_ptr<ross::InterpolatedPsfGrid> psfGrid;

    ScatterPsfSensor();
    explicit ScatterPsfSensor(const Sensor& other);
    std::string psfGridPath;
    std::string psfLoadStatus;
    std::unique_ptr<pfd::open_file> psfGridDialog;

    ~ScatterPsfSensor();
    bool renderUi(Sensor& owner);
    void freePsfGrid();
    void loadPsfGrid();
    uint32_t psfBinCount() const;

    NR_CPU_GPU glm::vec4 resolvePixel(uint32_t pixel, uint32_t width, const glm::vec4* accumulation) const
    {
        if (accumulation == nullptr)
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

        const glm::vec4 p = accumulation[pixel];
        glm::vec3 rgb(0.0f);
        if (p.w != 0.0f) {
            rgb.x = p.x / p.w;
            rgb.y = p.y / p.w;
            rgb.z = p.z / p.w;
        }
        (void)width;
        return glm::vec4(rgb, 1.0f);
    }

    template <typename WritePixel>
    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float sampleWeight, const SensorSampleContext& ctx,
        const WritePixel&) const
    {
        if (!psfGrid || ctx.accumulation == nullptr)
            return;
        glm::vec3 rgb = sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ);
        const uint32_t x = pixel % ctx.width;
        const uint32_t y = pixel / ctx.width;
        auto addSplat = [=](const ross::Vector2i& dest, float psfWeight) {
            if (dest.x < 0 || dest.y < 0 ||
                dest.x >= static_cast<int>(ctx.width) || dest.y >= static_cast<int>(ctx.height))
                return;
            glm::vec4& out = ctx.accumulation[static_cast<uint32_t>(dest.y) * ctx.width
                + static_cast<uint32_t>(dest.x)];
            sensorAtomicAdd(&out.x, rgb.x * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.y, rgb.y * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.z, rgb.z * sampleWeight * psfWeight);
            sensorAtomicAdd(&out.w, sampleWeight * psfWeight);
        };
        psfGrid->splatPsfForPixel(ross::Vector2i(static_cast<int>(x), static_cast<int>(y)),
            wl[0] * 0.001f, addSplat);
    }
};

// The scatter resolve belongs to the sensor because it is the sensor's
// accumulation representation that defines the resolve operation. Keeping the
// launch here also means generic post-processing does not need PSF knowledge.
inline NR_GPU_KERNEL void resolveScatterPsfKernel(
    const ScatterPsfSensor* sensor, const glm::vec4* accumulation,
    cudaSurfaceObject_t output, uint32_t width, uint32_t height)
{
#if defined(__CUDACC__)
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = width * height;
    if (pixel >= pixelCount)
        return;
    const glm::vec4 value = sensor->resolvePixel(pixel, width, accumulation);
    const uint32_t x = pixel % width;
    const uint32_t y = pixel / width;
    surf2Dwrite(make_float4(value.x, value.y, value.z, value.w),
        output, x * sizeof(float4), y);
#else
    (void)sensor; (void)accumulation; (void)output; (void)width; (void)height;
#endif
}
