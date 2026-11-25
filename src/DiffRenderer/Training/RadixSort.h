#pragma once

#include <cstdint>
#include <expected>

#include "DiffRenderer/Training/PrefixScan.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Training {

//! A stable least-significant-digit radix sort of key-value pairs on the GPU, eight bits per pass.
class RadixSort {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, std::size_t maxCount);
    //! Sorts the pairs in place by the low keyBits bits of the keys. With deviceCount, the GPU reads the count there, and count is its upper bound.
    [[nodiscard]] std::expected<void, NoorRHI::Error> record(NoorRHI::CommandList& list, NoorRHI::Ptr<std::uint64_t> keys, NoorRHI::Ptr<NoorRHI::uint> values,
                                                             std::uint32_t count, std::uint32_t keyBits, NoorRHI::Ptr<NoorRHI::uint> deviceCount = {}) const;
    [[nodiscard]] std::expected<void, NoorRHI::Error> record(NoorRHI::CommandList& list, NoorRHI::Ptr<NoorRHI::uint> keys, NoorRHI::Ptr<NoorRHI::uint> values,
                                                             std::uint32_t count, std::uint32_t keyBits, NoorRHI::Ptr<NoorRHI::uint> deviceCount = {}) const;

private:
    [[nodiscard]] std::expected<void, NoorRHI::Error> record(NoorRHI::CommandList& list, NoorRHI::Ptr<NoorRHI::uint> keys, NoorRHI::Ptr<NoorRHI::uint> values,
                                                             std::uint32_t count, std::uint32_t keyBits, NoorRHI::Ptr<NoorRHI::uint> deviceCount,
                                                             bool wideKeys) const;

    NoorRHI::ComputePipeline countDigits;
    NoorRHI::ComputePipeline scatterDigits;
    NoorRHI::ComputePipeline copyPairs;
    PrefixScan scan;
    NoorRHI::Buffer<NoorRHI::uint> sortedKeys;
    NoorRHI::Buffer<NoorRHI::uint> sortedValues;
    NoorRHI::Buffer<NoorRHI::uint> histograms;
    std::size_t maxCount = 0;
};

}
