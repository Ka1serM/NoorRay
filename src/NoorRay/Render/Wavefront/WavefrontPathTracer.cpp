#include "NoorRay/Render/Wavefront/WavefrontPathTracer.h"

#include <algorithm>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"
#include "NoorRayShadersVariants.h"

namespace NoorRay::Render {

namespace {

//! A kind of the light path counts up to 63, so a limit of 63 or more never ends a path.
std::uint32_t kindBounces(const std::uint32_t limit) {
    return std::min(limit, 63u);
}

Gpu::SceneContent shadedContent(const Gpu::SceneContent& content, const WavefrontSettings& settings) {
    Gpu::SceneContent result = content;
    result.hasGaussians = content.hasGaussians && settings.tracesGaussians();
    if (!settings.rasterShading)
        return result;
    result.hasVolumes = false;
    result.hasDispersion = false;
    result.hasSubsurface = false;
    return result;
}

//! As Cycles, the last vertex traces one more ray: only it finds unsampled light and light that shadow rays cannot reach, as through glass.
bool tracesFinalRay(const WavefrontSettings& settings, const bool restirs) {
    return !settings.rasterShading && !restirs;
}

//! The cells of the visibility cache, a power of two.
constexpr std::uint32_t visibilityCells = 1u << 16;

std::uint32_t divideUp(const std::uint64_t value, const std::uint64_t divisor) {
    return static_cast<std::uint32_t>((value + divisor - 1) / divisor);
}

//! The kernels that shade the hits of a wavefront iteration.
struct ShadeOptions {
    const KernelVariants* surfaces = nullptr;
    bool media = false;
    bool shadows = true;
    std::uint32_t shadowLayers = 1;
    bool regenerates = false;
};

//! Records the dispatches of one frame. The first failure is kept, and every later step is skipped.
class FrameRecorder {
public:
    FrameRecorder(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines, const VariantKey key, const Queues& queues,
                  const NoorRHI::Ptr<Shaders::WavefrontFrame> frame, const std::uint32_t gridThreads, GpuProfiler* profiler)
        : device(device), list(list), pipelines(pipelines), key(key), queues(queues), frame(frame), gridThreads(gridThreads), profiler(profiler) {}

    void advance(const std::uint32_t parity, const std::uint32_t phase) { direct(Shaders::advance, 1, parity, phase); }

    void direct(const KernelVariants& variants, const std::uint32_t threads, const std::uint32_t parity = 0, const std::uint32_t bin = 0) {
        run(variants, threads, Shaders::WavefrontArguments{frame, parity, bin});
    }

    template <NoorRHI::GpuArguments Arguments>
    void run(const KernelVariants& variants, const std::uint32_t threads, const Arguments& arguments) {
        const NoorRHI::ComputePipeline* pipeline = find(variants);
        if (!pipeline)
            return;
        const GpuScope scope(profiler, list, key.select(variants).entryPoint);
        list.dispatch(*pipeline, NoorRHI::DispatchSize{threads, 1, 1}, arguments);
    }

    //! A stage over GPU-counted work that the last advance phase sized: no group when the count is zero, at most the fixed grid.
    void stage(const KernelVariants& variants, const std::uint32_t slot, const std::uint32_t parity, const std::uint32_t bin = 0) {
        stage(variants, slot, Shaders::WavefrontArguments{frame, parity, bin});
    }

    template <NoorRHI::GpuArguments Arguments>
        requires std::is_class_v<Arguments>
    void stage(const KernelVariants& variants, const std::uint32_t slot, const Arguments& arguments) {
        const NoorRHI::ComputePipeline* pipeline = find(variants);
        if (!pipeline)
            return;
        const GpuScope scope(profiler, list, key.select(variants).entryPoint);
        list.dispatchIndirect(*pipeline, queues.getGridDispatch(slot), arguments);
    }

    //! A stage over GPU-counted work: the fixed grid, whose threads stride over the count the kernel reads.
    void grid(const KernelVariants& variants, const std::uint32_t parity, const std::uint32_t bin = 0) { direct(variants, gridThreads, parity, bin); }

    //! A number would pick this overload over the parity one, so the arguments must be a record.
    template <NoorRHI::GpuArguments Arguments>
        requires std::is_class_v<Arguments>
    void grid(const KernelVariants& variants, const Arguments& arguments) {
        run(variants, gridThreads, arguments);
    }

    //! The material kernel of every surface bin over the shade queue of source; the kernels write disjoint slots, so they run together.
    void materials(const Queues& queues, const std::uint32_t parity, const std::uint32_t source) {
        if (!result)
            return;
        const GpuScope scope(profiler, list, "materials");
        list.beginConcurrent();
        for (std::uint32_t bin = Shaders::fixedMaterialBin; bin < queues.getBinCount() && result; ++bin) {
            std::expected<const NoorRHI::ComputePipeline*, Core::Error> pipeline = pipelines.getMaterial(device, bin - Shaders::fixedMaterialBin);
            if (!pipeline) {
                result = std::unexpected(std::move(pipeline.error()));
                break;
            }
            list.dispatchIndirect(**pipeline, queues.getBinDispatch(bin), Shaders::WavefrontArguments{frame, parity, bin | source});
        }
        list.endConcurrent();
    }

    //! A wavefront iteration after Extend: bins the hits, shades them, traces the shadow rays, refills the free slots and swaps the queues.
    void shade(const std::uint32_t parity, const ShadeOptions& options) {
        advance(parity, Shaders::advanceBins);
        stage(Shaders::binPaths, Shaders::gridBinSource, parity, 0);
        if (options.media)
            stage(Shaders::shadeMedium, Shaders::gridMedium, parity, Shaders::mediumBin);
        materials(queues, parity, 0);
        stage(*options.surfaces, Shaders::gridSurface, parity);
        advance(parity, Shaders::advanceShadow);
        for (std::uint32_t layer = 0; layer + 1 < options.shadowLayers; ++layer) {
            grid(Shaders::shadow, parity, 0);
            advance(parity, Shaders::advanceBins | Shaders::advanceSourceShadow);
            grid(Shaders::binPaths, parity, Shaders::advanceSourceShadow);
            materials(queues, parity, Shaders::materialSourceShadow);
            grid(Shaders::resolveShadowAlpha, parity);
            advance(parity, Shaders::advanceShadow);
        }
        if (options.shadows)
            stage(Shaders::shadow, Shaders::gridShadow, parity, 1);
        if (!options.regenerates) {
            advance(parity, Shaders::advanceRegenerate | Shaders::advanceThenSwap);
            return;
        }
        advance(parity, Shaders::advanceRegenerate);
        stage(Shaders::regenerate, Shaders::gridFree, parity);
        advance(parity, Shaders::advanceSwap);
    }

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
    const Queues& queues;
    NoorRHI::Ptr<Shaders::WavefrontFrame> frame;
    std::uint32_t gridThreads;
    GpuProfiler* profiler;
    std::expected<void, Core::Error> result;
};

//! Folds the visibility counts of the last frame into the cells before a path reads them; the first frame after the cache was off or the light trees changed
//! clears it.
void resolveVisibility(FrameRecorder& recorder, const WavefrontSettings& settings, const std::uint64_t lightTrees, std::uint64_t& learnedTrees, bool& cleared) {
    if (!settings.visibilityCache) {
        cleared = false;
        return;
    }
    if (std::exchange(learnedTrees, lightTrees) != lightTrees)
        cleared = false;
    recorder.direct(Shaders::visibilityResolve, visibilityCells, 0, std::exchange(cleared, true) ? 0u : 1u);
}

}

std::expected<void, Core::Error> WavefrontPathTracer::open(NoorRHI::Device& openedDevice, std::shared_ptr<PipelineCache> sharedPipelines) {
    device = &openedDevice;
    pipelines = std::move(sharedPipelines);
    if (auto opened = Core::check(cameraBuffer.open(*device, 1)); !opened)
        return opened;
    if (auto opened = Core::check(visibilityKeys.open(*device, visibilityCells)); !opened)
        return opened;
    if (auto opened = Core::check(visibilityCounts.open(*device, 2 * Shaders::visibilityChoices * visibilityCells)); !opened)
        return opened;
    if (auto opened = Core::check(visibilityData.open(*device, Shaders::visibilityCellWords * visibilityCells)); !opened)
        return opened;
    visibilityCleared = false;
    return Core::check(frameBuffer.open(*device, 1));
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
                                                       const bool regenerates, const std::uint32_t gridThreads, const NoorRHI::float2 jitter,
                                                       const bool restirs) const {
    const std::uint32_t budget = settings.pathBudget != 0 && !restirs ? settings.pathBudget : targets.getPixelCount();
    const std::uint64_t base = Core::combineHash(Core::hashSeed, settings.seed);
    const auto seed = static_cast<std::uint32_t>(Core::combineHash(base, frameIndex));
    //! One scramble for all samples of an accumulation keeps the Sobol points of a pixel stratified across frames.
    const auto patternSeed = static_cast<std::uint32_t>(Core::combineHash(base, frameIndex - accumulatedFrames));
    const bool accumulate = settings.accumulate && accumulatedFrames > 0;
    Shaders::QueueRecord queueRecord = queues.getRecord();
    queueRecord.gridThreads = gridThreads;
    return {
        .scene = scene.getRecord(),
        .camera = cameraBuffer.getPtr(),
        .pool = pool.getRecord(),
        .queues = queueRecord,
        .film = targets.getRecord(accumulate, settings.splitsLighting, settings.approximatesSpecular, settings.exposure),
        .settings = {std::clamp(settings.maxDepth, 1u, 255u) | std::min(settings.maxVolumeBounces, 15u) << 8, settings.rouletteStart, seed, patternSeed},
        .limits = {settings.indirectClamp, static_cast<float>(budget), jitter.x, jitter.y},
        .options = {0, restirs ? 1u : 0u, settings.alphaShadowLayers, regenerates ? 1u : 0u},
        .raster = raster,
        .filter = {std::max(settings.filterWidth, 0.0f), std::max(settings.directClamp, 0.0f),
                   settings.filterGlossy > 0.0f ? 1.0f / settings.filterGlossy : 0.0f, 0.0f},
        .bounces = {kindBounces(settings.maxDiffuseBounces), kindBounces(settings.maxGlossyBounces), kindBounces(settings.maxTransmissionBounces),
                    tracesFinalRay(settings, restirs) ? 1u : 0u},
        .visibility = settings.visibilityCache
                          ? Shaders::VisibilityCacheRecord{visibilityKeys.getPtr(), visibilityCounts.getPtr(), visibilityData.getPtr(), visibilityCells - 1, 0}
                          : Shaders::VisibilityCacheRecord{},
        .lighting = {settings.lightTreeDepth, 0, 0, 0},
    };
}

std::expected<Shaders::RadianceCacheRecord, Core::Error> WavefrontPathTracer::beginRadianceCache(const Gpu::GpuScene& scene,
                                                                                                 const WavefrontSettings& settings) {
    if (!settings.radianceCache || settings.rasterShading)
        return Shaders::RadianceCacheRecord{};
    if (auto reserved = radianceCache.reserve(*device, capacity.paths, scene.getLightLayout()); !reserved)
        return std::unexpected(std::move(reserved.error()));
    return radianceCache.makeRecord(cameraPosition);
}

std::expected<void, Core::Error> WavefrontPathTracer::record(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const Shaders::CameraRecord& camera,
                                                             const WavefrontSettings& settings, const Shaders::RasterRecord& raster, const FilmHooks& hooks) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the wavefront path tracer is not open");
    if (settings.splitsLighting)
        if (auto opened = targets.openLightingSplit(*device); !opened)
            return opened;
    if (auto selected = scene.setLightTree(settings.lightTree); !selected)
        return selected;
    list.upload(cameraBuffer, std::span<const Shaders::CameraRecord>(&camera, 1));
    cameraPosition = {camera.cameraToWorld[0].w, camera.cameraToWorld[1].w, camera.cameraToWorld[2].w};
    std::expected<Reuse, Core::Error> reuse = selectReuse(scene, settings);
    if (!reuse)
        return std::unexpected(std::move(reuse.error()));
    if (*reuse != Reuse::Di)
        directLighting.resetHistory();
    if (auto recorded =
            *reuse == Reuse::Pt ? recordRestir(list, scene, settings, hooks) : recordWavefront(list, scene, settings, raster, hooks, *reuse == Reuse::Di);
        !recorded)
        return recorded;
    ++frameIndex;
    ++accumulatedFrames;
    if (!settings.accumulate)
        accumulatedFrames = 0;
    return {};
}

std::expected<Reuse, Core::Error> WavefrontPathTracer::selectReuse(const Gpu::GpuScene& scene, const WavefrontSettings& settings) {
    //! ReSTIR never hits Gaussians, so scenes with traced Gaussians keep the wavefront path tracer.
    const VariantKey key(shadedContent(scene.getContent(), settings), false);
    const bool hasGaussians = key.has(LinkConstant::HasGaussians);
    const Reuse wanted = settings.rasterShading || hasGaussians ? Reuse::Off : settings.reuse;
    const std::string_view name = wanted == Reuse::Pt ? "ReSTIR PT" : "ReSTIR DI";
    compilingReuse = Reuse::Off;
    if (wanted == Reuse::Off) {
        if (std::exchange(activeReuse, Reuse::Off) != Reuse::Off) {
            accumulatedFrames = 0;
            if (hasGaussians)
                Core::Log::info("the scene has Gaussians, which ReSTIR does not shade; the wavefront path tracer renders it");
        }
        return Reuse::Off;
    }
    if (activeReuse != wanted && settings.compilesInBackground) {
        std::expected<bool, Core::Error> ready =
            wanted == Reuse::Pt ? RestirPathTracer::isReady(*device, *pipelines, key) : RestirDirectLighting::isReady(*device, *pipelines, key);
        if (!ready)
            return std::unexpected(std::move(ready.error()));
        if (!*ready) {
            compilingReuse = wanted;
            return Reuse::Off;
        }
        Core::Log::info("{} is compiled; the path tracer switches to it and restarts the accumulation", name);
    }
    if (std::exchange(activeReuse, wanted) != wanted)
        accumulatedFrames = 0;
    return wanted;
}

std::expected<void, Core::Error> WavefrontPathTracer::resolveFilm(NoorRHI::CommandList& list, const VariantKey& key, const std::uint32_t gridThreads,
                                                                  const FilmHooks& hooks) {
    if (hooks.before)
        if (auto recorded = hooks.before(list, frameBuffer.getPtr()); !recorded)
            return recorded;
    FrameRecorder recorder(*device, list, *pipelines, key, queues, frameBuffer.getPtr(), gridThreads, profiler);
    recorder.grid(Shaders::accumulate, 0);
    if (auto recorded = recorder.takeResult(); !recorded)
        return recorded;
    if (hooks.after)
        return hooks.after(list, frameBuffer.getPtr());
    return {};
}

//! The material kernel of each graph with varying emission bakes the emission of its triangle lights when the camera or the scene changes.
std::expected<void, Core::Error> WavefrontPathTracer::bakeEmitters(NoorRHI::CommandList& list, const Gpu::GpuScene& scene) {
    const std::span<const std::pair<std::uint32_t, std::uint32_t>> programs = scene.getEmitterPrograms();
    if (programs.empty() || (accumulatedFrames != 0 && scene.getLastChange() == Scene::Change::None))
        return {};
    const GpuScope scope(profiler, list, "emitters");
    list.beginConcurrent();
    for (const auto& [program, count] : programs) {
        std::expected<const NoorRHI::ComputePipeline*, Core::Error> pipeline = pipelines->getMaterial(*device, program + 1);
        if (!pipeline) {
            list.endConcurrent();
            return std::unexpected(std::move(pipeline.error()));
        }
        const NoorRHI::DispatchSize size{std::min(count, Shaders::emitterRowThreads), divideUp(count, Shaders::emitterRowThreads), 1};
        list.dispatch(**pipeline, size,
                      Shaders::WavefrontArguments{frameBuffer.getPtr(), 0, (Shaders::firstGraphBin + program) | Shaders::materialSourceEmitters});
    }
    list.endConcurrent();
    return {};
}

//! A path slot takes about 310 bytes.
std::uint32_t WavefrontPathTracer::getPoolCapacity(const WavefrontSettings& settings) const {
    if (settings.poolCapacity != 0)
        return settings.poolCapacity;
    const std::uint64_t gigabytes = device->getInfo().memoryBytes >> 30;
    if (gigabytes >= 7 || device->getInfo().memoryBytes == 0)
        return 1u << 21;
    return gigabytes >= 3 ? 1u << 20 : 1u << 19;
}

std::expected<void, Core::Error> WavefrontPathTracer::recordRestir(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings,
                                                                   const FilmHooks& hooks) {
    const std::uint32_t pixels = targets.getPixelCount();
    const auto bins = static_cast<std::uint32_t>(Shaders::firstGraphBin + scene.getMaterials().getGraphCodes().size());
    const Capacity needed{std::clamp(getPoolCapacity(settings), 1u, pixels), 2 * pixels, bins};
    if (auto reserved = reserve(needed); !reserved)
        return reserved;
    const VariantKey key(shadedContent(scene.getContent(), settings), false);
    const std::uint32_t rounds = divideUp(pixels, capacity.paths);
    const std::uint32_t depth = RestirPathTracer::getPathDepth(settings.maxDepth);
    //! An alpha-tested hit that lets the ray through costs a wavefront step without a bounce.
    const std::uint32_t alphaSteps = key.has(LinkConstant::HasAlphaTest) ? std::max(settings.maxTransparentBounces, 1u) : 0u;
    //! ReSTIR PT attenuates its paths in closed form; the wavefront path tracer continues the paths that scatter in a volume.
    const bool forks = key.has(LinkConstant::HasVolumes) && settings.maxVolumeBounces > 0;
    const std::uint32_t gridThreads =
        std::min(std::max(settings.gridGroups, 1u), divideUp(capacity.paths, Shaders::wavefrontGroupSize)) * Shaders::wavefrontGroupSize;
    Shaders::WavefrontFrame frame = makeFrame(scene, settings, {}, rounds > 1, gridThreads, hooks.jitter, true);
    std::expected<Shaders::RadianceCacheRecord, Core::Error> cache = beginRadianceCache(scene, settings);
    if (!cache)
        return std::unexpected(std::move(cache.error()));
    frame.cache = *cache;
    const bool caches = settings.radianceCache;
    list.upload(frameBuffer, std::span<const Shaders::WavefrontFrame>(&frame, 1));
    if (auto baked = bakeEmitters(list, scene); !baked)
        return baked;
    const RestirInputs inputs{
        .wavefront = frameBuffer.getPtr(),
        .width = targets.getWidth(),
        .height = targets.getHeight(),
        .maxDepth = depth,
        .poolCapacity = capacity.paths,
        .lightLayout = scene.getLightLayout(),
        .temporal = settings.reusesHistory,
        .forks = forks,
        .cache = frame.cache,
    };
    std::expected<NoorRHI::Ptr<Shaders::RestirFrame>, Core::Error> restirFrame = restir.begin(*device, list, inputs);
    if (!restirFrame)
        return std::unexpected(std::move(restirFrame.error()));
    const auto arguments = [&](const std::uint32_t parity, const std::uint32_t mode) {
        return Shaders::RestirArguments{.frame = *restirFrame, .parity = parity, .stage = 0, .mode = mode, .start = 0, .count = 0, .forward = 0};
    };

    FrameRecorder recorder(*device, list, *pipelines, key, queues, frameBuffer.getPtr(), gridThreads, profiler);
    if (forks)
        resolveVisibility(recorder, settings, scene.getLightTrees(), visibilityTrees, visibilityCleared);
    if (caches && radianceCache.clearsCache())
        recorder.grid(Shaders::radianceCacheResolve, 0, Shaders::radianceCacheModeClear);
    list.pushLabel("Restir initial paths");
    recorder.advance(0, Shaders::advanceReset);
    recorder.grid(Shaders::buildSamples, 0);
    recorder.advance(0, Shaders::advanceGenerate);
    recorder.stage(Shaders::generate, Shaders::gridActive, 0);
    std::uint32_t parity = 0;
    const std::uint32_t iterations = depth * rounds + alphaSteps;
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration, parity ^= 1u) {
        recorder.stage(Shaders::restirExtend, Shaders::gridActive, arguments(parity, Shaders::restirPathInitial));
        recorder.advance(parity, Shaders::advanceBins);
        recorder.stage(Shaders::binPaths, Shaders::gridBinSource, parity, 0);
        recorder.materials(queues, parity, 0);
        recorder.stage(Shaders::restirBounce, Shaders::gridBinned, arguments(parity, Shaders::restirPathInitial));
        if (rounds == 1) {
            recorder.advance(parity, Shaders::advanceShadow | Shaders::advanceThenSwap);
            continue;
        }
        recorder.advance(parity, Shaders::advanceShadow);
        recorder.grid(Shaders::regenerate, parity);
        recorder.advance(parity, Shaders::advanceSwap);
    }
    recorder.stage(Shaders::restirBounce, Shaders::gridActive, arguments(parity, Shaders::restirPathFinish));
    list.popLabel();

    if (forks) {
        list.pushLabel("Restir volume forks");
        const ShadeOptions shading{
            .surfaces = &Shaders::shadeSurface,
            .media = true,
            .shadows = true,
            .shadowLayers = alphaSteps != 0 ? std::max(settings.alphaShadowLayers, 1u) : 1u,
            .regenerates = false,
        };
        for (std::uint32_t start = 0; start < pixels; start += capacity.paths) {
            recorder.advance(parity, Shaders::advanceReset);
            Shaders::RestirArguments sweep = arguments(parity, 0);
            sweep.start = start;
            sweep.count = std::min(capacity.paths, pixels - start);
            recorder.grid(Shaders::restirFork, sweep);
            recorder.shade(parity, shading);
            parity ^= 1u;
            for (std::uint32_t iteration = 0; iteration < settings.maxDepth + alphaSteps; ++iteration, parity ^= 1u) {
                recorder.stage(Shaders::extend, Shaders::gridActive, parity);
                recorder.shade(parity, shading);
            }
            recorder.grid(Shaders::finish, parity);
        }
        list.popLabel();
    }

    list.pushLabel("Restir resampling");
    recorder.advance(parity, Shaders::advanceReset);
    const std::uint32_t replaySteps = restir.getReplayDepth(settings.maxDepth) - 1 + alphaSteps;
    for (const std::uint32_t mode : {Shaders::restirModeTemporal, Shaders::restirModeSpatial}) {
        if (mode == Shaders::restirModeTemporal && !settings.reusesHistory)
            continue;
        const std::uint32_t forward = restir.getForwardShifts(mode);
        const std::uint32_t sweep = std::max(capacity.paths / forward, 1u);
        for (std::uint32_t start = 0; start < pixels; start += sweep) {
            Shaders::RestirArguments resample = arguments(parity, mode);
            resample.start = start;
            resample.count = std::min(sweep, pixels - start);
            resample.forward = forward;
            for (std::uint32_t stage = 0; stage < Shaders::restirStages; ++stage) {
                resample.stage = stage;
                resample.parity = parity;
                recorder.grid(Shaders::restirResample, resample);
                if (stage + 1 == Shaders::restirStages)
                    break;
                recorder.advance(parity, Shaders::advanceSwap);
                parity ^= 1u;
                for (std::uint32_t step = 0; step < replaySteps; ++step, parity ^= 1u) {
                    recorder.stage(Shaders::restirExtend, Shaders::gridActive, arguments(parity, Shaders::restirPathReplay));
                    recorder.advance(parity, Shaders::advanceBins);
                    recorder.stage(Shaders::binPaths, Shaders::gridBinSource, parity, 0);
                    recorder.materials(queues, parity, 0);
                    recorder.stage(Shaders::restirReplay, Shaders::gridBinned, arguments(parity, Shaders::restirPathReplay));
                    recorder.advance(parity, Shaders::advanceShadow | Shaders::advanceThenSwap);
                }
            }
        }
    }
    list.popLabel();

    list.pushLabel("Restir shading");
    recorder.grid(Shaders::restirShade, arguments(parity, 0));
    list.popLabel();
    if (caches) {
        list.pushLabel("SHaRC resolve");
        recorder.grid(Shaders::radianceCacheResolve, parity, Shaders::radianceCacheModeResolve);
        list.popLabel();
    }
    if (auto recorded = recorder.takeResult(); !recorded)
        return recorded;
    restir.end();
    if (caches)
        radianceCache.end(cameraPosition);
    return resolveFilm(list, key, gridThreads, hooks);
}

std::expected<void, Core::Error> WavefrontPathTracer::recordWavefront(NoorRHI::CommandList& list, Gpu::GpuScene& scene, const WavefrontSettings& settings,
                                                                      const Shaders::RasterRecord& raster, const FilmHooks& hooks, const bool directs) {
    const std::uint32_t pixels = targets.getPixelCount();
    const std::uint32_t budget = settings.pathBudget != 0 ? settings.pathBudget : pixels;
    const std::uint32_t samples = budget + pixels;
    const auto bins = static_cast<std::uint32_t>(Shaders::firstGraphBin + scene.getMaterials().getGraphCodes().size());
    const Capacity needed{std::clamp(getPoolCapacity(settings), 1u, samples), samples, bins};
    if (auto reserved = reserve(needed); !reserved)
        return reserved;

    const bool rasterShading = settings.rasterShading;
    //! Raster shading has no bounce, so the Gaussians it traces are emissive.
    const VariantKey key(shadedContent(scene.getContent(), settings), !rasterShading && settings.gaussianShading == GaussianShading::Scattering);
    //! A frame has about the budget in samples, so the rounds follow it; regeneration stays on while the samples can pass the pool.
    const std::uint32_t rounds = divideUp(budget, capacity.paths);
    const bool regenerates = samples > capacity.paths;
    const std::uint32_t maxDepth = rasterShading ? 1u : std::max(settings.maxDepth, 1u) + (tracesFinalRay(settings, false) ? 1u : 0u);
    const bool tracesAlpha = key.has(LinkConstant::HasAlphaTest) && !rasterShading;
    const std::uint32_t iterations = (maxDepth + (tracesAlpha ? std::max(settings.maxTransparentBounces, 1u) : 0u)) * rounds;
    const std::uint32_t shadowLayers = tracesAlpha ? std::max(settings.alphaShadowLayers, 1u) : 1u;
    const KernelVariants& shadeVariants = rasterShading ? Shaders::shadeRaster : directs ? Shaders::shadeSurfaceDirect : Shaders::shadeSurface;
    const std::uint32_t gridThreads =
        std::min(std::max(settings.gridGroups, 1u), divideUp(capacity.paths, Shaders::wavefrontGroupSize)) * Shaders::wavefrontGroupSize;
    Shaders::WavefrontFrame frame = makeFrame(scene, settings, raster, regenerates, gridThreads, hooks.jitter, false);
    std::expected<Shaders::RadianceCacheRecord, Core::Error> cache = beginRadianceCache(scene, settings);
    if (!cache)
        return std::unexpected(std::move(cache.error()));
    frame.cache = *cache;
    const bool caches = settings.radianceCache && !rasterShading;
    NoorRHI::Ptr<Shaders::RestirFrame> directFrame;
    if (directs) {
        const RestirDirectInputs inputs{
            .wavefront = frameBuffer.getPtr(),
            .width = targets.getWidth(),
            .height = targets.getHeight(),
            .lightLayout = scene.getLightLayout(),
            .temporal = settings.reusesHistory,
        };
        std::expected<NoorRHI::Ptr<Shaders::RestirFrame>, Core::Error> begun = directLighting.begin(*device, list, inputs);
        if (!begun)
            return std::unexpected(std::move(begun.error()));
        directFrame = *begun;
        frame.direct = directLighting.getRecord();
    }
    list.upload(frameBuffer, std::span<const Shaders::WavefrontFrame>(&frame, 1));
    if (auto baked = bakeEmitters(list, scene); !baked)
        return baked;

    FrameRecorder recorder(*device, list, *pipelines, key, queues, frameBuffer.getPtr(), gridThreads, profiler);
    list.pushLabel("Wavefront");
    resolveVisibility(recorder, settings, scene.getLightTrees(), visibilityTrees, visibilityCleared);
    if (caches && radianceCache.clearsCache())
        recorder.grid(Shaders::radianceCacheResolve, 0, Shaders::radianceCacheModeClear);
    recorder.advance(0, Shaders::advanceReset);
    recorder.grid(Shaders::buildSamples, 0);
    recorder.advance(0, Shaders::advanceGenerate);
    recorder.stage(Shaders::generate, Shaders::gridActive, 0);

    const ShadeOptions shading{
        .surfaces = &shadeVariants,
        .media = key.has(LinkConstant::HasGaussians) || key.has(LinkConstant::HasVolumes),
        .shadows = !rasterShading,
        .shadowLayers = shadowLayers,
        .regenerates = regenerates,
    };
    std::uint32_t parity = 0;
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration, parity ^= 1u) {
        recorder.stage(Shaders::extend, Shaders::gridActive, parity);
        recorder.shade(parity, shading);
    }
    recorder.grid(Shaders::finish, parity);
    list.popLabel();
    if (directs) {
        list.pushLabel("Restir DI");
        for (const std::uint32_t mode : {Shaders::restirDirectSample, Shaders::restirDirectShade})
            recorder.grid(Shaders::restirDirect,
                          Shaders::RestirArguments{.frame = directFrame, .parity = 0, .stage = 0, .mode = mode, .start = 0, .count = 0, .forward = 0});
        list.popLabel();
    }
    if (caches) {
        list.pushLabel("SHaRC resolve");
        recorder.grid(Shaders::radianceCacheResolve, parity, Shaders::radianceCacheModeResolve);
        list.popLabel();
    }
    if (auto recorded = recorder.takeResult(); !recorded)
        return recorded;
    if (directs)
        directLighting.end();
    if (caches)
        radianceCache.end(cameraPosition);
    return resolveFilm(list, key, gridThreads, hooks);
}
}
