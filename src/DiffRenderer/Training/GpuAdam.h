#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Training {

struct AdamSettings {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    //! 3DGS trains with 1e-15.
    float epsilon = 1e-15f;
};

//! One group of parameters with its gradients and moments, each count floats long.
struct AdamGroup {
    NoorRHI::Ptr<float> parameters;
    NoorRHI::Ptr<float> gradients;
    NoorRHI::Ptr<float> first;
    NoorRHI::Ptr<float> second;
    std::uint32_t count = 0;
    float learningRate = 0.0f;
};

//! Adam with the bias correction of PyTorch on GPU buffers that the caller owns.
class GpuAdam {
public:
    //! The marker of recordGather for an entry that starts at zero, as the moments of a new primitive.
    static constexpr std::uint32_t noSource = 0xffffffffu;

    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const AdamSettings& settings = {});
    //! Updates the group; the step counts from 1 and sets the bias correction.
    void record(NoorRHI::CommandList& list, const AdamGroup& group, std::uint32_t step) const;
    //! Updates the group with a step size that already holds the bias correction of the first moment, as a method with its own schedule computes it.
    void recordCorrected(NoorRHI::CommandList& list, const AdamGroup& group, float stepSize, float secondCorrection) const;
    //! destination[i] = source[indices[i]] for entries of stride floats; the same indices on parameters and moments keep, duplicate or drop primitives.
    void recordGather(NoorRHI::CommandList& list, NoorRHI::Ptr<float> source, NoorRHI::Ptr<float> destination, NoorRHI::Ptr<NoorRHI::uint> indices,
                      std::uint32_t count, std::uint32_t stride) const;
    const AdamSettings& getSettings() const { return settings; }

private:
    AdamSettings settings;
    NoorRHI::ComputePipeline adam;
    NoorRHI::ComputePipeline gather;
};

}
