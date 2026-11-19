#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/Restir/RestirPathTracer.h"
#include "NoorRay/Render/Targets.h"
#include "NoorRay/Render/VariantKey.h"
#include "NoorRay/Render/Wavefront/PathPool.h"
#include "NoorRay/Render/Wavefront/Queues.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"

namespace NoorRay::Render {

enum class GaussianShading : std::uint32_t { Emissive, Scattering };

struct WavefrontSettings {
    std::uint32_t maxDepth = 16;
    std::uint32_t rouletteStart = 3;
    float indirectClamp = 0.0f;
    std::uint32_t pathBudget = 0;
    bool accumulate = true;
    bool resamplesPaths = false;
    //! The wavefront path tracer renders while the ReSTIR kernels compile; false waits for them instead.
    bool compilesInBackground = true;
    bool rasterShading = false;
    //! The raster splat pass draws the Gaussians, so raster shading leaves them out of its rays.
    bool splatsGaussians = false;
    GaussianShading gaussianShading = GaussianShading::Emissive;
    std::uint32_t alphaShadowLayers = 4;
    std::uint32_t poolCapacity = 1u << 20;
    //! Workgroups of the fixed grid that every wavefront stage dispatches, enough to fill the GPU; capped by the pool.
    std::uint32_t gridGroups = 2048;
    bool operator==(const WavefrontSettings&) const = default;
};

//! Spectral wavefront path tracing with inline ray queries. One record call traces one frame of the path budget into the film; every count stays on the GPU.
class WavefrontPathTracer {
public:
    //! The pipelines may be shared with other path tracers on the device, so each kernel compiles once.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height,
                                                        std::shared_ptr<PipelineCache> pipelines);
    [[nodiscard]] std::expected<void, Core::Error> resize(std::uint32_t width, std::uint32_t height);
    void restartAccumulation() { accumulatedFrames = 0; }

    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Shaders::CameraRecord& camera,
                                                          const WavefrontSettings& settings, const Shaders::RasterRecord& raster);

    const Targets& getTargets() const { return targets; }
    std::uint32_t getAccumulatedFrames() const { return accumulatedFrames; }

private:
    struct Capacity {
        std::uint32_t paths = 0;
        std::uint32_t samples = 0;
        std::uint32_t bins = 0;
        bool operator==(const Capacity&) const = default;
    };

    [[nodiscard]] std::expected<void, Core::Error> reserve(const Capacity& needed);
    [[nodiscard]] std::expected<bool, Core::Error> usesRestir(const Gpu::GpuScene& scene, const WavefrontSettings& settings);
    [[nodiscard]] std::expected<void, Core::Error> recordWavefront(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings,
                                                                   const Shaders::RasterRecord& raster);
    [[nodiscard]] std::expected<void, Core::Error> recordRestir(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings);
    Shaders::WavefrontFrame makeFrame(const Gpu::GpuScene& scene, const WavefrontSettings& settings, const Shaders::RasterRecord& raster, bool regenerates,
                                      std::uint32_t gridThreads) const;

    NoorRHI::Device* device = nullptr;
    std::shared_ptr<PipelineCache> pipelines;
    PathPool pool;
    Queues queues;
    Targets targets;
    RestirPathTracer restir;
    NoorRHI::Buffer<Shaders::CameraRecord> cameraBuffer;
    NoorRHI::Buffer<Shaders::WavefrontFrame> frameBuffer;
    Capacity capacity;
    std::uint32_t frameIndex = 0;
    std::uint32_t accumulatedFrames = 0;
    bool restirActive = false;
};

}
