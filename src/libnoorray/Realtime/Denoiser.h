#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <noorrhi/noorrhi.hpp>

#include "Realtime/FrameContext.h"
#include "Realtime/RenderTargets.h"
#include "Shared/RealtimeArgs.h"

namespace nrd
{
struct CommonSettings;
struct Instance;
}

// What the denoiser runs, and the lighting rectangles up to `extent` its
// images hold.
struct DenoiserLayout
{
    Extent extent;
    DenoiserMode mode{DenoiserMode::Off};
    bool sphericalHarmonics{};
    // The translucent layer set has its own denoiser.
    bool layers{};

    bool operator==(const DenoiserLayout&) const = default;
};

// NVIDIA's RELAX diffuse + specular denoiser (external/NRD), in its
// radiance or its spherical-harmonics (SH) form. It reads the diffuse,
// specular, normal-roughness, view Z and motion targets (plus the SH1 targets
// in SH form) and writes its own outputs, which the composite reads. Off, the
// composite reads the diffuse and specular targets directly.
//
// NRD describes compute pipelines built from its own SPIR-V, which binds
// resources through descriptor sets of its own rather than NoorRHI's
// bindless set. This class is the host side NRD expects: it creates those pipelines,
// its texture pools, samplers and constant buffer with Vulkan directly, and
// records NRD's dispatches into NoorRHI's command stream through
// noorrhi::interop::record.
class Denoiser
{
public:
    explicit Denoiser(noorrhi::Device& device);
    ~Denoiser();

    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;

    // Creates NRD with only the denoisers the layout runs, with their pools
    // and outputs; off, it holds no GPU memory at all. The GPU must be idle:
    // the previous configuration's resources are destroyed.
    void configure(const DenoiserLayout& layout);
    DenoiserMode mode() const { return layout_.mode; }
    // Whether the SH form runs. Always false while the denoiser is off.
    bool sphericalHarmonics() const
    {
        return layout_.sphericalHarmonics && layout_.mode != DenoiserMode::Off;
    }

    // What the composite reads and how the image pass packs its signals.
    nr::graphics::DenoiserArgs args(const RenderTargets& targets) const;
    // Denoises the diffuse and specular lighting targets into the outputs.
    void record(const FrameContext& frame, const RenderTargets& targets);

private:
    struct Texture;
    struct Resources;

    // The denoisers of the configured layout: the main one, then the layer's.
    std::vector<std::uint32_t> identifiers() const;
    void createPipelines();
    void createPools();
    void destroyPools();
    // Releases everything configure() created.
    void destroyInstance();
    void transitionPoolsToGeneral(std::uintptr_t commandBuffer);
    std::uint64_t descriptorSet(std::uint16_t pipelineIndex,
        const std::vector<std::uint64_t>& views);
    void dispatch(const nrd::CommonSettings& settings, const RenderTargets& targets);
    // Sizes the histories to a fixed time at the current frame rate.
    void updateHistoryLength(float frameTimeMilliseconds);

    noorrhi::Device& device_;
    DenoiserLayout layout_;
    // Exponentially smoothed, so the history length does not follow every
    // frame-time spike.
    float smoothedFrameTimeMilliseconds_{};
    // RELAX's history length as last set.
    std::uint32_t historyFrames_{};
    nrd::Instance* instance_{};
    std::unique_ptr<Resources> resources_;
    std::vector<Texture> pool_;
    // Descriptor sets are written once and reused for the same pipeline and
    // views, so sets referenced by in-flight command buffers are never updated.
    std::map<std::vector<std::uint64_t>, std::uint64_t> descriptorSets_;
    std::vector<std::uint64_t> descriptorPools_;
    noorrhi::Image<std::byte> diffuseOutput_;
    noorrhi::Image<std::byte> specularOutput_;
    noorrhi::Image<std::byte> diffuseSh1Output_;
    noorrhi::Image<std::byte> specularSh1Output_;
    noorrhi::Image<std::byte> layerDiffuseOutput_;
    noorrhi::Image<std::byte> layerSpecularOutput_;
    std::uint32_t frameIndex_{};
    std::uint64_t constantOffset_{};
    std::uint64_t previousConstantOffset_{};
    bool poolsNeedTransition_{};
};
