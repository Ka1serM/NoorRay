#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Gpu/GpuScene.h"
#include "NoorRay/Render/GpuProfiler.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Render/RadianceCache.h"
#include "NoorRay/Render/Restir/RestirDirectLighting.h"
#include "NoorRay/Render/Restir/RestirPathTracer.h"
#include "NoorRay/Render/Targets.h"
#include "NoorRay/Render/VariantKey.h"
#include "NoorRay/Render/Wavefront/PathPool.h"
#include "NoorRay/Render/Wavefront/Queues.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"

namespace NoorRay::Render {

//! Off leaves the Gaussians out of the traced rays, so the scene keeps ReSTIR and builds no Gaussian TLAS; the raster splats still draw them.
enum class GaussianShading : std::uint32_t { Off, Emissive, Scattering };
//! The reuse of the frames before: ReSTIR DI resamples the direct light of the primary vertices of the wavefront paths, ReSTIR PT whole paths.
enum class Reuse : std::uint8_t { Off, Di, Pt };

struct WavefrontSettings {
    std::uint32_t maxDepth = 16;
    //! The volume scattering events after which a path ends, as Cycles' volume bounces: 0 renders single scattering.
    std::uint32_t maxVolumeBounces = 15;
    std::uint32_t rouletteStart = 3;
    float indirectClamp = 0.0f;
    //! The clamp of light that reaches the camera after no bounce, as Cycles' Direct Light clamp; 0 is off.
    float directClamp = 0.0f;
    //! Cycles' Diffuse, Glossy and Transmission bounces: a path whose count of the kind passes its limit ends at its next vertex but for the emission there.
    std::uint32_t maxDiffuseBounces = 1024;
    std::uint32_t maxGlossyBounces = 1024;
    std::uint32_t maxTransmissionBounces = 1024;
    //! Cycles' Filter Glossy: the blur that raises the roughness of glossy surfaces after scatters of small pdf; 0 is off.
    float filterGlossy = 0.0f;
    //! The width in pixels of the Blackman-Harris pixel filter, as Cycles' Pixel Filter; 0 is a box of one pixel.
    float filterWidth = 1.5f;
    //! The factor of the output, as Cycles' film exposure.
    float exposure = 1.0f;
    std::uint32_t pathBudget = 0;
    //! Picks the lights of a receiver by the SG light tree; false picks them by their power alone, as Cycles without its light tree.
    bool lightTree = true;
    //! The vertices below this depth pick their lights by the light tree, the deeper ones by power alone, which costs much less; all by default.
    std::uint32_t lightTreeDepth = ~0u;
    //! Learns the visibility of the first light choices per cell of a hash grid (Wavefront/VisibilityCache.slang).
    bool visibilityCache = true;
    //! Cycles' Seed: another seed gives other random numbers in every frame.
    std::uint32_t seed = 0;
    bool accumulate = true;
    Reuse reuse = Reuse::Off;
    //! ReSTIR resamples the reservoirs of the previous frame too.
    bool reusesHistory = true;
    bool radianceCache = false;
    //! The wavefront path tracer renders while the ReSTIR kernels compile; false waits for them instead.
    bool compilesInBackground = true;
    bool rasterShading = false;
    //! The raster splat pass draws the Gaussians, so raster shading leaves them out of its rays.
    bool splatsGaussians = false;
    //! The film keeps diffuse and specular light apart for the denoiser.
    bool splitsLighting = false;
    //! The specular demodulation albedo is the EnvBRDFApprox2 guide of DLSS-RR.
    bool approximatesSpecular = false;
    GaussianShading gaussianShading = GaussianShading::Emissive;
    std::uint32_t alphaShadowLayers = 4;
    //! Cycles' Transparent max bounces: the alpha-tested surfaces that a path may pass, each of which costs a wavefront iteration.
    std::uint32_t maxTransparentBounces = 8;
    //! About 256 bytes per slot; 2M slots hold one path per pixel of 1080p, so a frame needs one round; 0 takes the capacity that suits the memory of the
    //! device.
    std::uint32_t poolCapacity = 0;
    //! Workgroups of the fixed grid that every wavefront stage dispatches, enough to fill the GPU; capped by the pool.
    std::uint32_t gridGroups = 2048;
    bool tracesGaussians() const { return gaussianShading != GaussianShading::Off && !splatsGaussians; }
    bool operator==(const WavefrontSettings&) const = default;
};

//! Passes around the film resolve: before sees the sums of this frame, after sees the accumulated mean in the film output.
struct FilmHooks {
    using Pass = std::function<std::expected<void, Core::Error>(NoorRHI::CommandList&, NoorRHI::Ptr<Shaders::WavefrontFrame>)>;
    Pass before;
    Pass after;
    //! The sub-pixel position in [0, 1) of every primary ray of the frame; negative is a random position per sample.
    NoorRHI::float2 jitter{-1.0f, -1.0f};
};

//! Spectral wavefront path tracing with inline ray queries. One record call traces one frame of the path budget into the film; every count stays on the GPU.
class WavefrontPathTracer {
public:
    //! The pipelines may be shared with other path tracers on the device, so each kernel compiles once. The film opens at the first resize.
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::shared_ptr<PipelineCache> pipelines);
    [[nodiscard]] std::expected<void, Core::Error> resize(std::uint32_t width, std::uint32_t height);
    void restartAccumulation() { accumulatedFrames = 0; }
    //! Times every kernel into the profiler; null stops the timing.
    void setProfiler(GpuProfiler* timing) { profiler = timing; }

    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Shaders::CameraRecord& camera,
                                                          const WavefrontSettings& settings, const Shaders::RasterRecord& raster, const FilmHooks& hooks = {});

    const Targets& getTargets() const { return targets; }
    void setSharedOutput(const NoorRHI::StorageTextureHandle image) { targets.setShared(image); }
    std::uint32_t getAccumulatedFrames() const { return accumulatedFrames; }
    [[nodiscard]] std::expected<void, Core::Error> openFilmSums() { return targets.openFilmSums(*device); }
    //! The record of the last traced frame, which the film passes after it read.
    NoorRHI::Ptr<Shaders::WavefrontFrame> getFrame() const { return frameBuffer.getPtr(); }
    //! The reuse whose kernels still compile in the background; until they are ready the frames render without reuse.
    Reuse getCompilingReuse() const { return compilingReuse; }

private:
    struct Capacity {
        std::uint32_t paths = 0;
        std::uint32_t samples = 0;
        std::uint32_t bins = 0;
        bool operator==(const Capacity&) const = default;
    };

    std::uint32_t getPoolCapacity(const WavefrontSettings& settings) const;
    [[nodiscard]] std::expected<void, Core::Error> reserve(const Capacity& needed);
    //! The reuse of this frame: the wanted one once its kernels are compiled, and none for raster shading or a scene with traced Gaussians.
    [[nodiscard]] std::expected<Reuse, Core::Error> selectReuse(const Gpu::GpuScene& scene, const WavefrontSettings& settings);
    //! directs hands the direct light of the primary vertices to ReSTIR DI.
    [[nodiscard]] std::expected<void, Core::Error> recordWavefront(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings,
                                                                   const Shaders::RasterRecord& raster, const FilmHooks& hooks, bool directs);
    //! Opens the SHaRC cache for this frame; an empty record when it is off.
    [[nodiscard]] std::expected<Shaders::RadianceCacheRecord, Core::Error> beginRadianceCache(const Gpu::GpuScene& scene, const WavefrontSettings& settings);
    [[nodiscard]] std::expected<void, Core::Error> bakeEmitters(NoorRHI::CommandList& list, const Gpu::GpuScene& scene);
    [[nodiscard]] std::expected<void, Core::Error> recordRestir(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings,
                                                                const FilmHooks& hooks);
    [[nodiscard]] std::expected<void, Core::Error> resolveFilm(NoorRHI::CommandList& list, const VariantKey& key, std::uint32_t gridThreads,
                                                               const FilmHooks& hooks);
    //! With restirs the budget is one camera path per pixel, whose ReSTIR shading writes the film.
    Shaders::WavefrontFrame makeFrame(const Gpu::GpuScene& scene, const WavefrontSettings& settings, const Shaders::RasterRecord& raster, bool regenerates,
                                      std::uint32_t gridThreads, NoorRHI::float2 jitter, bool restirs) const;

    NoorRHI::Device* device = nullptr;
    std::shared_ptr<PipelineCache> pipelines;
    PathPool pool;
    Queues queues;
    Targets targets;
    RestirPathTracer restir;
    RestirDirectLighting directLighting;
    RadianceCache radianceCache;
    NoorRHI::Buffer<Shaders::CameraRecord> cameraBuffer;
    NoorRHI::Buffer<Shaders::WavefrontFrame> frameBuffer;
    //! The visibility cache of the light tree learns across frames; its decay follows changes to the scene, and a new light tree empties it.
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> visibilityKeys;
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> visibilityCounts;
    NoorRHI::Buffer<std::uint32_t> visibilityData;
    std::uint64_t visibilityTrees = 0;
    bool visibilityCleared = false;
    Capacity capacity;
    std::uint32_t frameIndex = 0;
    std::uint32_t accumulatedFrames = 0;
    NoorRHI::float3 cameraPosition;
    GpuProfiler* profiler = nullptr;
    Reuse activeReuse = Reuse::Off;
    Reuse compilingReuse = Reuse::Off;
};

}
