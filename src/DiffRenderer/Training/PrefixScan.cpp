#include "DiffRenderer/Training/PrefixScan.h"

#include "Sorting.h"
#include "Sorting.slang.h"

namespace DiffRenderer::Training {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t groupSize = 256;
constexpr std::uint32_t blockSize = 2048;

std::size_t getBlocks(const std::size_t count) {
    return (count + blockSize - 1) / blockSize;
}

}

std::expected<void, Error> PrefixScan::open(Device& device, const std::size_t count) {
    if (count == 0 || count > std::size_t{65535} * blockSize)
        return fail(ErrorCode::InvalidArgument, "the prefix scan supports between 1 and 65535 x 2048 values");
    maxCount = count;
    levels.clear();
    if (std::expected<void, Error> opened = reduceBlocks.open(device, Sorting::reduceBlocks); !opened)
        return opened;
    if (std::expected<void, Error> opened = scanBlocks.open(device, Sorting::scanBlocks); !opened)
        return opened;
    for (std::size_t blocks = getBlocks(count); blocks > 1; blocks = getBlocks(blocks))
        if (std::expected<void, Error> opened = levels.emplace_back().open(device, blocks); !opened)
            return opened;
    return {};
}

std::expected<void, Error> PrefixScan::record(CommandList& list, const Ptr<uint> input, const Ptr<uint> output, const std::uint32_t count,
                                              const Ptr<uint> deviceCount) const {
    if (count > maxCount)
        return fail(ErrorCode::InvalidArgument, "the prefix scan has more values than it opened for");
    record(list, input, output, count, deviceCount, 0);
    return {};
}

void PrefixScan::record(CommandList& list, const Ptr<uint> input, const Ptr<uint> output, const std::uint32_t count, const Ptr<uint> deviceCount,
                        const std::size_t level) const {
    const auto blocks = static_cast<std::uint32_t>(getBlocks(count));
    if (blocks <= 1) {
        list.dispatch(scanBlocks, {groupSize}, Sorting::ScanArguments{input, output, {}, deviceCount, count});
        return;
    }
    const Ptr<uint> sums = levels[level].getPtr();
    list.dispatch(reduceBlocks, {blocks * groupSize}, Sorting::ScanArguments{input, output, sums, deviceCount, count});
    record(list, sums, sums, blocks, {}, level + 1);
    list.dispatch(scanBlocks, {blocks * groupSize}, Sorting::ScanArguments{input, output, sums, deviceCount, count});
}

}
