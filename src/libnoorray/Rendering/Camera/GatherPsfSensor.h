#pragma once

#include "Rendering/Camera/Sensor.h"
#include "Backend/CUDA/rstd/UniquePtr.h"

#include "libross/imaging/interpolatedpsfgrid/InterpolatedPsfGrid.h"

#include <memory>
#include <string>
#include <cuda_runtime_api.h>
#include "Backend/CUDA/Unique/AsyncDeviceBuffer.h"

class GatherPsfSensor : public Sensor::Type<GatherPsfSensor, RectangularSensor> {
public:
    nr::rstd::unique_ptr<ross::InterpolatedPsfGrid> psfGrid;

    GatherPsfSensor();
    explicit GatherPsfSensor(const Sensor& other);
    std::string psfGridPath;
    std::string psfLoadStatus;
    std::unique_ptr<pfd::open_file> psfGridDialog;
    nr::cuda::UniqueAsyncDeviceBuffer psfGatherBuckets;
    size_t psfGatherBucketCapacity{};

    ~GatherPsfSensor();
    bool renderUi(Sensor& owner);
    void freePsfGrid();
    void loadPsfGrid();
    uint32_t psfBinCount() const;
    void freeScratch(cudaStream_t stream) noexcept;
    void prepareFrame(uint32_t width, uint32_t height, bool resetAccumulation,
        cudaStream_t stream, PsfGatherBucketSample*& buckets, uint32_t& binCount);

    NR_CPU_GPU glm::vec4 resolvePixel(uint32_t pixel, uint32_t width, uint32_t height,
        uint32_t psfBinCount, const PsfGatherBucketSample* psfBuckets) const
    {
        if (!psfGrid || psfBuckets == nullptr || psfBinCount == 0)
            return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

        const size_t pixelCount = static_cast<size_t>(width) * height;
        const int destX = static_cast<int>(pixel % width);
        const int destY = static_cast<int>(pixel / width);

        double rgbSum[3] = {0.0, 0.0, 0.0};
        double weightSum = 0.0;

        for (uint32_t bin = 0; bin < psfBinCount; ++bin) {
            const auto& psfMeta = psfGrid->metadata.psfs[bin];
            const int halfKernelX = int(psfMeta.discretePsfSampleCount.x) / 2;
            const int halfKernelY = int(psfMeta.discretePsfSampleCount.y) / 2;

            for (int dy = -halfKernelY; dy <= halfKernelY; ++dy) {
                for (int dx = -halfKernelX; dx <= halfKernelX; ++dx) {
                    const int srcX = destX + dx;
                    const int srcY = destY + dy;
                    if (srcX < 0 || srcY < 0 ||
                        srcX >= static_cast<int>(width) || srcY >= static_cast<int>(height))
                        continue;

                    const float psfWeight = psfGrid->sampleScatterWeightForPsf(
                        bin, ross::Vector2i(srcX, srcY), ross::Vector2i(destX, destY));
                    if (psfWeight == 0.0f)
                        continue;

                    const PsfGatherBucketSample& bucket =
                        psfBuckets[static_cast<size_t>(bin) * pixelCount
                            + static_cast<uint32_t>(srcY) * width
                            + static_cast<uint32_t>(srcX)];
                    rgbSum[0] += static_cast<double>(psfWeight) * bucket.rgbSum[0];
                    rgbSum[1] += static_cast<double>(psfWeight) * bucket.rgbSum[1];
                    rgbSum[2] += static_cast<double>(psfWeight) * bucket.rgbSum[2];
                    weightSum += static_cast<double>(psfWeight) * bucket.count;
                }
            }
        }

        glm::vec3 rgb(0.0f);
        if (weightSum != 0.0) {
            rgb.x = static_cast<float>(rgbSum[0] / weightSum);
            rgb.y = static_cast<float>(rgbSum[1] / weightSum);
            rgb.z = static_cast<float>(rgbSum[2] / weightSum);
        }
        return glm::vec4(rgb, 1.0f);
    }

    template <typename WritePixel>
    NR_CPU_GPU void addSample(uint32_t pixel, const SampledSpectrum& L,
        const SampledWavelengths& wl, float sampleWeight, const SensorSampleContext& ctx,
        const WritePixel& writePixel) const
    {
        RectangularSensor::addSample(pixel, L, wl, sampleWeight, ctx, writePixel);

        if (!psfGrid || ctx.psfBuckets == nullptr)
            return;
        glm::vec3 rgb = sensorRGBFromSpectrum(L, wl, ctx.cieX, ctx.cieY, ctx.cieZ);
        const size_t bin = psfGrid->nearestPsfIndex(wl[0] * 0.001f);
        const size_t pixelCount = static_cast<size_t>(ctx.width) * ctx.height;
        PsfGatherBucketSample& bucket = ctx.psfBuckets[bin * pixelCount + pixel];
        sensorAtomicAdd(&bucket.rgbSum[0], static_cast<double>(rgb.x));
        sensorAtomicAdd(&bucket.rgbSum[1], static_cast<double>(rgb.y));
        sensorAtomicAdd(&bucket.rgbSum[2], static_cast<double>(rgb.z));
        sensorAtomicAdd(&bucket.count, 1.0);
    }
};

inline NR_GPU_KERNEL void resolveGatherPsfKernel(
    const GatherPsfSensor* sensor, glm::vec4* accumulation,
    cudaSurfaceObject_t output, uint32_t width, uint32_t height,
    uint32_t psfBinCount, const PsfGatherBucketSample* psfBuckets)
{
#if defined(__CUDACC__)
    const uint32_t pixel = NR_GPU_LAUNCH_IDX;
    const uint32_t pixelCount = width * height;
    if (pixel >= pixelCount)
        return;
    const glm::vec4 value = sensor->resolvePixel(
        pixel, width, height, psfBinCount, psfBuckets);
    accumulation[pixel] = value;
    const uint32_t x = pixel % width;
    const uint32_t y = pixel / width;
    surf2Dwrite(make_float4(value.x, value.y, value.z, value.w),
        output, x * sizeof(float4), y);
#else
    (void)sensor; (void)accumulation; (void)output; (void)width;
    (void)height; (void)psfBinCount; (void)psfBuckets;
#endif
}
