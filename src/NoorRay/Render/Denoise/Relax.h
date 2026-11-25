#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "NoorRHI/Commands/CommandList.h"
#include "NoorRHI/Commands/RenderGraph.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRHI/Resources/Sampler.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/PipelineCache.h"
#include "NoorRay/Shaders/Common/CameraRecords.h"

namespace nrd {
struct Instance;
}

namespace NoorRay::Render {

//! The camera of one frame as NRD needs it: column-major matrices of a view space that looks along +z, so view depth is positive.
struct DenoiseView {
    std::array<float, 16> worldToView{};
    std::array<float, 16> viewToClip{};
    //! The sub-pixel offset of the primary rays from the pixel centre, in pixels.
    NoorRHI::float2 jitter{};

    static DenoiseView fromCamera(const Shaders::CameraRecord& camera, NoorRHI::float2 jitter);
};

//! A frame of few samples per pixel that RELAX filters with its history, or an accumulated film that it filters once, without history.
enum class RelaxInput : std::uint8_t { Frame, Film };

//! NVIDIA NRD RELAX on NoorRHI: NRD plans the passes of the unmodified RELAX shaders and this class runs them, with RELAX diffuse for the fog light.
class Relax {
public:
    Relax();
    ~Relax();
    Relax(Relax&&) noexcept;
    Relax& operator=(Relax&&) noexcept;

    [[nodiscard]] std::expected<void, Core::Error> resize(NoorRHI::Device& device, std::uint32_t width, std::uint32_t height, bool fog);
    //! restart discards the history, for a new accumulation or a cut; a film always restarts.
    [[nodiscard]] std::expected<void, Core::Error> record(NoorRHI::Device& device, NoorRHI::CommandList& list, PipelineCache& pipelines,
                                                          const DenoiseView& view, bool restart, RelaxInput input);

    //! The inputs that the prepare kernel writes.
    const NoorRHI::Image& getDiffuse() const { return inputs[0]; }
    const NoorRHI::Image& getSpecular() const { return inputs[1]; }
    const NoorRHI::Image& getNormalRoughness() const { return inputs[2]; }
    const NoorRHI::Image& getViewZ() const { return inputs[3]; }
    const NoorRHI::Image& getMotion() const { return inputs[4]; }
    const NoorRHI::Image& getFog() const { return inputs[5]; }
    const NoorRHI::Image& getDiffuseSh() const { return inputs[6]; }
    const NoorRHI::Image& getSpecularSh() const { return inputs[7]; }
    //! The guides of the fog denoiser, at the light-weighted distance of the fog instead of the surface.
    const NoorRHI::Image& getFogNormalRoughness() const { return inputs[8]; }
    const NoorRHI::Image& getFogViewZ() const { return inputs[9]; }
    const NoorRHI::Image& getFogMotion() const { return inputs[10]; }
    const NoorRHI::Image& getDenoisedDiffuse() const { return outputs[0]; }
    const NoorRHI::Image& getDenoisedSpecular() const { return outputs[1]; }
    const NoorRHI::Image& getDenoisedFog() const { return outputs[2]; }
    const NoorRHI::Image& getDenoisedDiffuseSh() const { return outputs[3]; }
    const NoorRHI::Image& getDenoisedSpecularSh() const { return outputs[4]; }
    //! The mean of the composited denoised frames of an accumulation and their count in w; composite writes it.
    const NoorRHI::Image& getDenoisedMean() const { return mean; }
    bool hasFog() const { return denoiserCount == denoisers.size(); }
    std::uint32_t getWidth() const { return width; }
    std::uint32_t getHeight() const { return height; }

private:
    struct InstanceDeleter {
        void operator()(nrd::Instance* instance) const;
    };

    //! NRD shares one scratch pool between the denoisers of an instance, so each denoiser has its own instance and the render graph overlaps their passes.
    struct Denoiser {
        std::unique_ptr<nrd::Instance, InstanceDeleter> instance;
        //! The NoorRay shader of each NRD pipeline index.
        std::vector<const NoorRHI::Shader*> shaders;
        std::vector<NoorRHI::Image> permanentPool;
        std::vector<NoorRHI::Image> transientPool;
    };

    [[nodiscard]] std::expected<void, Core::Error> openInstances(NoorRHI::Device& device);
    [[nodiscard]] std::expected<void, Core::Error> addDispatches(NoorRHI::RenderGraph& graph, NoorRHI::Device& device, PipelineCache& pipelines,
                                                                 std::size_t denoiser);
    const NoorRHI::Image* findImage(std::size_t denoiser, std::uint32_t type, std::uint16_t indexInPool) const;

    std::array<Denoiser, 2> denoisers;
    //! 1 without fog: the fog denoiser and its images take as much memory and time as the rest of RELAX.
    std::size_t denoiserCount = 0;
    std::array<NoorRHI::Sampler, 2> samplers;
    std::array<NoorRHI::Image, 11> inputs;
    std::array<NoorRHI::Image, 5> outputs;
    NoorRHI::Image mean;
    DenoiseView previous;
    std::uint32_t frameIndex = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool resized = true;
};

}
