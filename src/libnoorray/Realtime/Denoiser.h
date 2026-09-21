#pragma once

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

// NVIDIA's REBLUR or RELAX diffuse + specular denoiser (external/NRD). It
// reads the diffuse, specular, normal-roughness, view Z and motion targets and
// writes its own outputs, which the composite reads. Off, the composite reads
// the diffuse and specular targets directly.
//
// NRD describes compute pipelines built from its own SPIR-V, which binds
// resources through classic descriptor sets rather than NoorRHI's descriptor
// heap. This class is the host side NRD expects: it creates those pipelines,
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

    void setMode(DenoiserMode mode) { mode_ = mode; }
    DenoiserMode mode() const { return mode_; }

    // Allocates NRD's texture pools and the outputs at the render resolution.
    // The GPU must be idle: resources of the previous size are destroyed.
    void resize(Extent render);
    // What the composite reads and how the image pass packs its signals.
    nr::graphics::DenoiserArgs args(const RenderTargets& targets) const;
    // Denoises the diffuse and specular targets into the outputs.
    void record(const FrameContext& frame, const RenderTargets& targets);

private:
    struct Texture;
    struct Resources;

    void createPipelines();
    void destroyPools();
    void transitionPoolsToGeneral(std::uintptr_t commandBuffer);
    std::uint64_t descriptorSet(std::uint16_t pipelineIndex,
        const std::vector<std::uint64_t>& views);
    void dispatch(const nrd::CommonSettings& settings, const RenderTargets& targets);

    noorrhi::Device& device_;
    DenoiserMode mode_{RenderSettings{}.denoiserMode};
    nrd::Instance* instance_{};
    std::unique_ptr<Resources> resources_;
    std::vector<Texture> pool_;
    // Descriptor sets are written once and reused for the same pipeline and
    // views, so sets referenced by in-flight command buffers are never updated.
    std::map<std::vector<std::uint64_t>, std::uint64_t> descriptorSets_;
    std::vector<std::uint64_t> descriptorPools_;
    noorrhi::Image<std::byte> diffuseOutput_;
    noorrhi::Image<std::byte> specularOutput_;
    Extent extent_;
    std::uint32_t frameIndex_{};
    std::uint64_t constantOffset_{};
    std::uint64_t previousConstantOffset_{};
    bool poolsNeedTransition_{};
};
