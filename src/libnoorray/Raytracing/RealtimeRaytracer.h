#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <vector>

#include "Raytracer.h"
#include "Realtime/Accumulator.h"
#include "Realtime/Denoiser.h"
#include "Realtime/FrameContext.h"
#include "Realtime/RenderTargets.h"
#include "Realtime/Restir.h"
#include "Realtime/Upscaler.h"

namespace nr::graphics { struct RealtimeArgs; }

// Interactive biased RGB renderer. It shares the scene/resource contract with
// the spectral renderer but owns a separate, cheaper shader pipeline: meshes
// only, each section hit through its material's hit group, one path per pixel.
//
// A frame is a fixed sequence of stages around the primary passes, each of
// which passes the frame through unchanged in its Off mode:
//
//   Restir::presample -> [G-buffer pass] -> lighting pass
//   -> Restir::resample -> Denoiser -> composite and output AOVs -> Upscaler
//   -> Accumulator
//
// The lighting pass traces, resamples and denoises lighting at the lighting
// resolution, drawing ReSTIR's initial samples on the surfaces it finds. At full lighting resolution it also writes the G-buffer; below
// it the G-buffer pass traces primary visibility at the render resolution and
// the composite upsamples the lighting onto it.
//
// The primary passes write RenderTargets; the stages read them. Each stage
// owns its resources and settings translation; this class owns the targets,
// the frame's view and the one decision of when temporal history restarts.
class RealtimeRaytracer final : public Raytracer
{
public:
    // onMaterialShadersCompiled is called from a background thread when
    // material shaders finish compiling or linking, so the host calls
    // linkCompiledMaterialShaders().
    RealtimeRaytracer(noorrhi::Device& device, uint32_t width, uint32_t height,
        bool exportColorMemory = false, std::function<void()> onMaterialShadersCompiled = {});
    ~RealtimeRaytracer() override;

    RaytracerType type() const noexcept override
    {
        return RaytracerType::Realtime;
    }

    // Rays are traced at the resolution the upscaler mode derives; the
    // upscaler brings the result back to the logical size. Hosts and the
    // viewport composite need the trace resolution to reason about how sharp
    // the AOVs behind the beauty are.
    uint32_t traceWidth() const override;
    uint32_t traceHeight() const override;
    RaytracerResources& resources() override { return common; }
    const RaytracerResources& resources() const override { return common; }
    std::uint32_t readCryptomatteAtOutput(uint32_t x, uint32_t y) override;
    noorrhi::float4 readPositionAtOutput(uint32_t x, uint32_t y) override;
    noorrhi::TextureHandle viewportDepthTexture() const { return targets ? noorrhi::TextureHandle{targets->handles().depth} : noorrhi::TextureHandle{}; }
    glm::vec2 viewportDepthJitter() const { return {previousJitter[0], previousJitter[1]}; }
    void setSelectionAovRequired(bool required) { selectionAovRequired = required; }

    bool prepareFrameResources() override;
    void render(uint32_t frameIndex = 0, uint32_t sampleIndex = 0) override;
    void restartTemporalHistory() override;
    bool linkCompiledMaterialShaders() override;
    MaterialShaderStage materialShaderStage() const override;
    void waitForMaterialShaders() override;

protected:
    void renderImpl();
    void onImageAllocationChanged();
    void onLightsUploaded();
    void onRenderSettingsApplied(const RenderSettings& settings);
    void onMaterialShadersChanged(std::span<const MaterialHitShaders> shaders);
    void onHitRecordsChanged(std::span<const HitRecord> records);

private:
    RaytracerResources common;
    // Everything the stages' images and buffers are allocated for: the
    // largest rectangles the base class's image allocation admits, and the
    // settings that decide which exist. Frames render into smaller
    // rectangles of them freely; a change of layout reallocates and restarts
    // history.
    struct ResourceLayout
    {
        RenderTargetLayout targets;
        Extent output;
        DenoiserMode denoiser{};

        bool operator==(const ResourceLayout&) const = default;
    };

    // This frame's rectangles.
    Extent outputExtent() const;
    Extent renderExtent() const;
    Extent lightingExtent() const;
    Extent lightingExtent(Extent render) const;
    ResourceLayout requiredLayout() const;
    // Reallocates the stages' resources when the required layout changed.
    // Waits for the device when it does.
    void ensureResources();
    // Starts compiling the material shaders no batch holds yet, as far as
    // threads are free.
    void compilePendingMaterialShaders();
    struct MaterialLibraryBatch;
    // Appends a finished batch's library and its shaders' hit groups.
    void addCompiledBatch(MaterialLibraryBatch& batch);
    // Links the pass library with every material library on a background
    // thread into the pipeline all ray-tracing passes launch from.
    void startPipelineLink();
    // Makes a finished link the trace pipeline; returns whether it did.
    bool adoptLinkedPipeline();
    // Fills hitGroups from hitRecords, shading records of materials whose
    // library is not linked yet, or failed to compile, with the default
    // material.
    void assignHitGroups();
    // Rebinds the trace pipeline to hitGroups, first linking the pass
    // library alone when there is none yet.
    void linkTracePipeline();
    // Builds this frame's context and fills args->view, then records the
    // frame as the previous one.
    FrameContext beginFrame();

    noorrhi::Shader lightingRaygen;
    noorrhi::Shader layeredLightingRaygen;
    noorrhi::Sampler materialSampler;
    noorrhi::Shader gBufferRaygen;
    noorrhi::Shader layeredGBufferRaygen;
    LightingResolution lightingResolution{RenderSettings{}.lightingResolution};
    BufferVisualization bufferVisualization{RenderSettings{}.bufferVisualization};
    DenoiserMode denoiserMode{RenderSettings{}.denoiserMode};
    bool selectionAovRequired{};
    Restir restir;
    Denoiser denoiser;
    Upscaler upscaler;
    noorrhi::Buffer<nr::graphics::BeautyAccumulation> beautyAccumulation;
    // Alternates every render(); see BeautyAccumulation.
    std::uint32_t accumulationSlot{};
    Accumulator accumulator;
    // Allocated by ensureResources(), with the other stages' resources.
    std::optional<RenderTargets> targets;
    std::optional<ResourceLayout> allocatedLayout;
    // Every pass's ray generation with the miss stages and the default
    // material's hit group. It never changes, so the driver compiles it once
    // and caches it on disk.
    noorrhi::RayTracingLibrary passLibrary;
    // One library per compiled batch of material hit groups, in
    // material-shader order. A new batch compiles only its own materials.
    std::vector<noorrhi::RayTracingLibrary> materialLibraries;
    std::vector<MaterialHitShaders> materialShaders;
    // Each material shader's first hit group in a pipeline that links
    // materialLibraries, or NoMaterialGroups when its batch did not compile.
    std::vector<uint32_t> materialGroupBases;
    uint32_t nextMaterialGroup{};
    // The shaders tracePipeline links, those with a compiled batch, and those
    // plus the compiling batches.
    std::size_t linkedMaterialShaderCount{};
    std::size_t compiledMaterialShaderCount{};
    std::size_t requestedMaterialShaderCount{};
    std::vector<HitRecord> hitRecords;
    // The hit group of each of the scene's hit records.
    std::vector<std::uint32_t> hitGroups;
    bool transparentMaterials{};
    noorrhi::RayTracingPipeline tracePipeline;
    // The material libraries the latest link covers.
    std::size_t requestedLibraryCount{};
    noorrhi::ComputePipeline compositePipeline;
    noorrhi::ComputePipeline layeredCompositePipeline;
    noorrhi::ComputePipeline outputAovsPipeline;
    // Reconstructs one pixel's surface position into pickPosition.
    noorrhi::ComputePipeline pickPipeline;
    noorrhi::Buffer<noorrhi::float4> pickPosition;
    // The latest frame's arguments, which picking reads the image on screen
    // with.
    std::unique_ptr<nr::graphics::RealtimeArgs> args;
    // `args` for the passes that light the translucent layer surface set.
    std::unique_ptr<nr::graphics::RealtimeArgs> layerArgs;

    bool hasHistory{};
    Extent previousLighting;
    std::array<float, 16> previousWorldToView{};
    std::array<float, 16> previousViewToClip{};
    std::array<float, 2> previousJitter{};
    std::array<float, 3> previousCameraPosition{};
    std::array<float, 3> previousPreviousCameraPosition{};
    std::chrono::steady_clock::time_point previousFrameStart{};
    std::function<void()> onMaterialShadersCompiled;
    // Drivers take seconds to compile hit shaders and to link them, so both
    // run off the render thread. `done` is set as the work finishes, before
    // the callback runs, so a host woken by it never finds it pending. It is
    // declared before the future, whose destruction waits for the work.
    struct MaterialLibraryBatch
    {
        std::size_t shaderCount{};
        std::atomic<bool> done{};
        // Empty when the driver rejected the batch.
        std::future<noorrhi::RayTracingLibrary> library;
    };
    struct PipelineLink
    {
        std::size_t shaderCount{};
        std::atomic<bool> done{};
        // Empty when the driver rejected the link.
        std::future<noorrhi::RayTracingPipeline> pipeline;
    };
    // The batch compiling, at most one, which joins materialLibraries once
    // done. A deque keeps it in place while its compile refers to it.
    std::deque<MaterialLibraryBatch> compilingMaterialLibraries;
    std::optional<PipelineLink> pipelineLink;
};
