#include "DiffRenderer/Training/RadixSort.h"

#include <algorithm>
#include <array>
#include <utility>

#include "Sorting.h"
#include "Sorting.slang.h"

namespace DiffRenderer::Training {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t groupSize = 256;
constexpr std::uint32_t blockSize = groupSize * 16;
constexpr std::uint32_t digitBits = 8;
constexpr std::uint32_t digitCount = 1u << digitBits;
constexpr std::uint32_t maxThreads = 65535u * groupSize;

std::size_t getBlocks(const std::size_t count) {
    return (count + blockSize - 1) / blockSize;
}

}

std::expected<void, Error> RadixSort::open(Device& device, const std::size_t count) {
    if (count == 0 || getBlocks(count) > 65535)
        return fail(ErrorCode::InvalidArgument, "the radix sort supports between 1 and 65535 x 4096 pairs");
    maxCount = count;
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 3>{
             {{&countDigits, &Sorting::countDigits}, {&scatterDigits, &Sorting::scatterDigits}, {&copyPairs, &Sorting::copyPairs}}})
        if (std::expected<void, Error> opened = pipeline->open(device, *shader); !opened)
            return opened;
    const std::size_t histogramCount = getBlocks(count) * digitCount;
    for (std::expected<void, Error> step :
         {sortedKeys.open(device, count * 2), sortedValues.open(device, count), histograms.open(device, histogramCount), scan.open(device, histogramCount)})
        if (!step)
            return step;
    return {};
}

std::expected<void, Error> RadixSort::record(CommandList& list, const Ptr<std::uint64_t> keys, const Ptr<uint> values, const std::uint32_t count,
                                             const std::uint32_t keyBits, const Ptr<uint> deviceCount) const {
    return record(list, keys.as<uint>(), values, count, keyBits, deviceCount, true);
}

std::expected<void, Error> RadixSort::record(CommandList& list, const Ptr<uint> keys, const Ptr<uint> values, const std::uint32_t count,
                                             const std::uint32_t keyBits, const Ptr<uint> deviceCount) const {
    if (keyBits > 32)
        return fail(ErrorCode::InvalidArgument, "32-bit keys have at most 32 key bits");
    return record(list, keys, values, count, keyBits, deviceCount, false);
}

std::expected<void, Error> RadixSort::record(CommandList& list, const Ptr<uint> keys, const Ptr<uint> values, const std::uint32_t count,
                                             const std::uint32_t keyBits, const Ptr<uint> deviceCount, const bool wideKeys) const {
    if (count > maxCount)
        return fail(ErrorCode::InvalidArgument, "the radix sort has more pairs than it opened for");
    if (keyBits == 0 || keyBits > 64)
        return fail(ErrorCode::InvalidArgument, "the radix sort needs between 1 and 64 key bits");
    if (count == 0)
        return {};
    const auto blocks = static_cast<std::uint32_t>(getBlocks(count));
    const std::uint32_t passes = (keyBits + digitBits - 1) / digitBits;
    std::array<Ptr<uint>, 2> pairKeys{keys, sortedKeys.getPtr()};
    std::array<Ptr<uint>, 2> pairValues{values, sortedValues.getPtr()};
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
        const std::uint32_t bits = std::min(digitBits, keyBits - pass * digitBits);
        const Sorting::SortArguments arguments{pairKeys[0], pairValues[0], pairKeys[1],      pairValues[1],    histograms.getPtr(), deviceCount,
                                               count,       blocks,        pass * digitBits, (1u << bits) - 1, wideKeys ? 1u : 0u,  0};
        list.dispatch(countDigits, {blocks * groupSize}, arguments);
        if (std::expected<void, Error> scanned = scan.record(list, histograms.getPtr(), histograms.getPtr(), blocks * digitCount); !scanned)
            return scanned;
        list.dispatch(scatterDigits, {blocks * groupSize}, arguments);
        std::swap(pairKeys[0], pairKeys[1]);
        std::swap(pairValues[0], pairValues[1]);
    }
    if (passes % 2 == 0)
        return {};
    const std::uint32_t threads = std::min(count, maxThreads);
    list.dispatch(
        copyPairs, {threads},
        Sorting::SortArguments{keys, values, sortedKeys.getPtr(), sortedValues.getPtr(), {}, deviceCount, count, blocks, 0, 0, wideKeys ? 1u : 0u, threads});
    return {};
}

}
