#include "NoorRay/Render/Wavefront/WavefrontPathTracer.h"

#include <algorithm>
#include <span>
#include <utility>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

namespace {

Gpu::SceneContent shadedContent(const Gpu::SceneContent& content, const WavefrontSettings& settings) {
    if (!settings.rasterShading)
        return content;
    Gpu::SceneContent result = content;
    result.hasGaussians = content.hasGaussians && !settings.splatsGaussians;
    result.hasVolumes = false;
    result.hasDispersion = false;
    return result;
}

std::uint32_t divideUp(const std::uint64_t value, const std::uint64_t divisor) {
    return static_cast<std::uint32_t>((value + divisor - 1) / divisor);
}

//! Records the dispatches of one frame. The first failure is kept, and every later step is skipped.
class FrameRecorder {
public:
    FrameRecorder(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines, const VariantKey key,
                  const NoorRHI::Ptr<Shaders::WavefrontFrame> frame, const std::uint32_t gridThreads)
        : device(device), list(list), pipelines(pipelines), key(key), frame(frame), gridThreads(gridThreads) {}

    void advance(const std::uint32_t parity, const std::uint32_t phase) { direct(Shaders::advance, 1, parity, phase); }

    void direct(const KernelVariants& variants, const std::uint32_t threads, const std::uint32_t parity = 0, const std::uint32_t bin = 0) {
        if (const NoorRHI::ComputePipeline* pipeline = find(variants))
            list.dispatch(*pipeline, NoorRHI::DispatchSize{threads, 1, 1}, Shaders::WavefrontArguments{frame, parity, bin});
    }

    //! A stage over GPU-counted work: the fixed grid, whose threads stride over the count the kernel reads.
    void grid(const KernelVariants& variants, const std::uint32_t parity, const std::uint32_t bin = 0) { direct(variants, gridThreads, parity, bin); }

    std::expected<void, Core::Error> takeResult() { return std::move(result); }

private:
    const NoorRHI::ComputePipeline* find(const KernelVariants& variants) {
        if (!result)
            return nullptr;
        std::expected<const NoorRHI::ComputePipeline*, Core::Error> pipeline = pipelines.get(device, key.select(variants));
        if (!pipeline) {
            result = std::unexpected(std::move(pipeline.error()));
            return nullptr;
        }
        return *pipeline;
    }

    NoorRHI::Device& device;
    NoorRHI::CommandList& list;
    PipelineCache& pipelines;
    VariantKey key;
    NoorRHI::Ptr<Shaders::WavefrontFrame> frame;
    std::uint32_t gridThreads;
    std::expected<void, Core::Error> result;
};

}

std::expected<void, Core::Error> WavefrontPathTracer::open(NoorRHI::Device& openedDevice, const std::uint32_t width, const std::uint32_t height,
                                                           std::shared_ptr<PipelineCache> sharedPipelines) {
    device = &openedDevice;
    pipelines = std::move(sharedPipelines);
    if (auto opened = Core::check(cameraBuffer.open(*device, 1)); !opened)
        return opened;
    if (auto opened = Core::check(frameBuffer.open(*device, 1)); !opened)
        return opened;
    return resize(width, height);
}

std::expected<void, Core::Error> WavefrontPathTracer::resize(const std::uint32_t width, const std::uint32_t height) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the wavefront path tracer is not open");
    if (targets.getWidth() == width && targets.getHeight() == height)
        return {};
    accumulatedFrames = 0;
    return targets.open(*device, width, height);
}

std::expected<void, Core::Error> WavefrontPathTracer::reserve(const Capacity& needed) {
    if (needed.paths != capacity.paths)
        if (auto opened = pool.open(*device, needed.paths); !opened)
            return opened;
    if (needed != capacity)
        if (auto opened = queues.open(*device, needed.paths, needed.samples, needed.bins); !opened)
            return opened;
    capacity = needed;
    return {};
}

Shaders::WavefrontFrame WavefrontPathTracer::makeFrame(const Gpu::GpuScene& scene, const WavefrontSettings& settings, const Shaders::RasterRecord& raster,
                                                       const bool regenerates, const std::uint32_t gridThreads) const {
    const std::uint32_t budget = settings.pathBudget != 0 ? settings.pathBudget : targets.getPixelCount();
    const auto seed = static_cast<std::uint32_t>(Core::combineHash(Core::hashSeed, frameIndex));
    const bool accumulate = settings.accumulate && accumulatedFrames > 0;
    Shaders::QueueRecord queueRecord = queues.getRecord();
    queueRecord.gridThreads = gridThreads;
    return {
        .scene = scene.getRecord(),
        .camera = cameraBuffer.getPtr(),
        .pool = pool.getRecord(),
        .queues = queueRecord,
        .film = targets.getRecord(accumulate),
        .settings = {std::max(settings.maxDepth, 1u), settings.rouletteStart, seed, frameIndex},
        .limits = {settings.indirectClamp, static_cast<float>(budget), 0.0f, 0.0f},
        .options = {0, 0, settings.alphaShadowLayers, regenerates ? 1u : 0u},
        .raster = raster,
    };
}

std::expected<void, Core::Error> WavefrontPathTracer::record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Shaders::CameraRecord& camera,
                                                             const WavefrontSettings& settings, const Shaders::RasterRecord& raster) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the wavefront path tracer is not open");
    list.upload(cameraBuffer, std::span<const Shaders::CameraRecord>(&camera, 1));
    std::expected<bool, Core::Error> restirs = usesRestir(scene, settings);
    if (!restirs)
        return std::unexpected(std::move(restirs.error()));
    if (auto recorded = *restirs ? recordRestir(list, scene, settings) : recordWavefront(list, scene, settings, raster); !recorded)
        return recorded;
    ++frameIndex;
    ++accumulatedFrames;
    if (!settings.accumulate)
        accumulatedFrames = 0;
    return {};
}

std::expected<bool, Core::Error> WavefrontPathTracer::usesRestir(const Gpu::GpuScene& scene, const WavefrontSettings& settings) {
    //! ReSTIR PT shades only surfaces: Gaussians and volumes would never be hit, so such scenes keep the wavefront path tracer.
    const VariantKey key(scene.getContent(), false);
    const bool hasMedia = key.has(LinkConstant::HasGaussians) || key.has(LinkConstant::HasVolumes);
    const bool wanted = settings.resamplesPaths && !settings.rasterShading && !hasMedia;
    if (!wanted) {
        if (std::exchange(restirActive, false)) {
            accumulatedFrames = 0;
            if (hasMedia)
                Core::Log::info("the scene has Gaussians or volumes, which ReSTIR PT does not shade; the wavefront path tracer renders it");
        }
        return false;
    }
    if (restirActive || !settings.compilesInBackground) {
        restirActive = true;
        return true;
    }
    std::expected<bool, Core::Error> ready = RestirPathTracer::isReady(*device, *pipelines, key);
    if (!ready || !*ready)
        return ready;
    Core::Log::info("ReSTIR PT is compiled; the path tracer switches to it, restarts the accumulation and releases the wavefront path pool");
    restirActive = true;
    pool = {};
    queues = {};
    capacity = {};
    accumulatedFrames = 0;
    return true;
}

std::expected<void, Core::Error> WavefrontPathTracer::recordRestir(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings) {
    const std::uint32_t pixels = targets.getPixelCount();
    const std::uint32_t gridThreads = std::min(std::max(settings.gridGroups, 1u), divideUp(pixels, Shaders::wavefrontGroupSize)) * Shaders::wavefrontGroupSize;
    const Shaders::WavefrontFrame frame = makeFrame(scene, settings, {}, false, gridThreads);
    list.upload(frameBuffer, std::span<const Shaders::WavefrontFrame>(&frame, 1));
    const VariantKey key(scene.getContent(), false);
    list.pushLabel("Restir");
    const RestirInputs inputs{
        .wavefront = frameBuffer.getPtr(),
        .width = targets.getWidth(),
        .height = targets.getHeight(),
        .maxDepth = std::max(settings.maxDepth, 1u),
        .seed = frame.settings.z,
        .gridThreads = gridThreads,
    };
    if (auto recorded = restir.record(*device, list, *pipelines, key, inputs); !recorded)
        return recorded;
    FrameRecorder recorder(*device, list, *pipelines, key, frameBuffer.getPtr(), gridThreads);
    recorder.grid(Shaders::accumulate, 0);
    list.popLabel();
    return recorder.takeResult();
}

std::expected<void, Core::Error> WavefrontPathTracer::recordWavefront(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings,
                                                                      const Shaders::RasterRecord& raster) {
    const std::uint32_t pixels = targets.getPixelCount();
    const std::uint32_t budget = settings.pathBudget != 0 ? settings.pathBudget : pixels;
    const std::uint32_t samples = budget + pixels;
    const Capacity needed{std::clamp(settings.poolCapacity, 1u, samples), samples, Shaders::shadeBinCount};
    if (auto reserved = reserve(needed); !reserved)
        return reserved;

    const bool rasterShading = settings.rasterShading;
    //! Raster shading has no bounce, so the Gaussians it traces are emissive.
    const VariantKey key(shadedContent(scene.getContent(), settings), !rasterShading && settings.gaussianShading == GaussianShading::Scattering);
    const std::uint32_t rounds = divideUp(samples, capacity.paths);
    const std::uint32_t maxDepth = rasterShading ? 1u : std::max(settings.maxDepth, 1u);
    const bool tracesAlpha = key.has(LinkConstant::HasAlphaTest) && !rasterShading;
    const std::uint32_t iterations = maxDepth * rounds + (tracesAlpha ? 2u : 0u);
    const std::uint32_t shadowLayers = tracesAlpha ? std::max(settings.alphaShadowLayers, 1u) : 1u;
    const KernelVariants& shadeVariants = rasterShading ? Shaders::shadeRaster : Shaders::shadeSurface;
    const std::uint32_t gridThreads =
        std::min(std::max(settings.gridGroups, 1u), divideUp(capacity.paths, Shaders::wavefrontGroupSize)) * Shaders::wavefrontGroupSize;
    const Shaders::WavefrontFrame frame = makeFrame(scene, settings, raster, rounds > 1, gridThreads);
    list.upload(frameBuffer, std::span<const Shaders::WavefrontFrame>(&frame, 1));

    FrameRecorder recorder(*device, list, *pipelines, key, frameBuffer.getPtr(), gridThreads);
    list.pushLabel("Wavefront");
    recorder.advance(0, Shaders::advanceReset);
    recorder.grid(Shaders::buildSamples, 0);
    recorder.advance(0, Shaders::advanceGenerate);
    recorder.grid(Shaders::generate, 0);

    const bool shadesMedia = key.has(LinkConstant::HasGaussians) || key.has(LinkConstant::HasVolumes);
    std::uint32_t parity = 0;
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration, parity ^= 1u) {
        recorder.grid(Shaders::extend, parity);
        recorder.advance(parity, Shaders::advanceBins);
        recorder.grid(Shaders::binPaths, parity, 0);
        recorder.grid(shadeVariants, parity, Shaders::surfaceBin);
        if (shadesMedia)
            recorder.grid(Shaders::shadeMedium, parity, Shaders::mediumBin);
        recorder.advance(parity, Shaders::advanceShadow);
        for (std::uint32_t layer = 0; layer + 1 < shadowLayers; ++layer) {
            recorder.grid(Shaders::shadow, parity, 0);
            recorder.advance(parity, Shaders::advanceBins | Shaders::advanceSourceShadow);
            recorder.grid(Shaders::binPaths, parity, Shaders::advanceSourceShadow);
            recorder.grid(Shaders::resolveShadowAlpha, parity, Shaders::surfaceBin);
            recorder.advance(parity, Shaders::advanceShadow);
        }
        if (!rasterShading)
            recorder.grid(Shaders::shadow, parity, 1);
        recorder.advance(parity, Shaders::advanceRegenerate);
        if (rounds > 1)
            recorder.grid(Shaders::regenerate, parity);
        recorder.advance(parity, Shaders::advanceSwap);
    }
    recorder.grid(Shaders::finish, parity);
    recorder.grid(Shaders::accumulate, 0);
    list.popLabel();
    return recorder.takeResult();
}

}
