#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Training {

//! The exclusive prefix sum of uint32 values on the GPU, by reduce-then-scan over blocks of 2048 values.
class PrefixScan {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::size_t maxCount);
    //! Input and output may be the same buffer; with deviceCount, the GPU reads the count there and count is its upper bound.
    [[nodiscard]] std::expected<void, NoorRHI::Error> record(NoorRHI::CommandList& list, NoorRHI::Ptr<NoorRHI::uint> input, NoorRHI::Ptr<NoorRHI::uint> output,
                                                             std::uint32_t count, NoorRHI::Ptr<NoorRHI::uint> deviceCount = {}) const;

private:
    void record(NoorRHI::CommandList& list, NoorRHI::Ptr<NoorRHI::uint> input, NoorRHI::Ptr<NoorRHI::uint> output, std::uint32_t count,
                NoorRHI::Ptr<NoorRHI::uint> deviceCount, std::size_t level) const;

    NoorRHI::ComputePipeline reduceBlocks;
    NoorRHI::ComputePipeline scanBlocks;
    //! The block sums of each level of the recursion.
    std::vector<NoorRHI::Buffer<NoorRHI::uint>> levels;
    std::size_t maxCount = 0;
};

}
