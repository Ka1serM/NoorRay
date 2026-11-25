#include <algorithm>
#include <format>
#include <numeric>
#include <print>
#include <random>
#include <ranges>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Training/PrefixScan.h"
#include "DiffRenderer/Training/RadixSort.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;

namespace {

constexpr std::array<std::uint32_t, 7> sizes{1, 1000, 2048, 4097, 100003, 1u << 20, 3000017};
constexpr std::uint32_t benchmarkCount = 10'000'000;
constexpr int benchmarkRepeats = 10;

template <class T>
std::expected<std::vector<T>, std::string> download(Device& device, const Buffer<T>& buffer, const std::size_t count) {
    Readback<T> readback;
    if (Result opened = check(readback.open(device, count)); !opened)
        return std::unexpected(opened.error());
    CommandList list(device);
    list.download(buffer, readback);
    if (Result submitted = submit(device, std::move(list)); !submitted)
        return std::unexpected(submitted.error());
    std::expected<std::span<const T>, Error> values = readback.read();
    if (!values)
        return std::unexpected(values.error().message);
    return std::vector<T>(values->begin(), values->end());
}

template <class Key>
Result checkSort(Device& device, const Training::RadixSort& sort, std::mt19937_64& random, const std::uint32_t count, const std::uint32_t keyBits,
                 const bool countOnGpu) {
    const std::uint64_t mask = keyBits == 64 ? ~0ull : (1ull << keyBits) - 1;
    std::vector<Key> keys(count);
    for (Key& key : keys)
        key = static_cast<Key>(random() & mask & (random() % 4 == 0 ? 0xff0000ffull : ~0ull));
    std::vector<uint> values(count);
    std::iota(values.begin(), values.end(), 0u);
    Buffer<Key> keyBuffer;
    Buffer<uint> valueBuffer;
    Buffer<uint> countBuffer;
    for (std::expected<void, Error> step : {keyBuffer.open(device, count), valueBuffer.open(device, count), countBuffer.open(device, 1), keyBuffer.upload(keys),
                                            valueBuffer.upload(values), countBuffer.upload(std::array{count})})
        if (Result result = check(step); !result)
            return result;
    CommandList list(device);
    const Ptr<uint> deviceCount = countOnGpu ? countBuffer.getPtr() : Ptr<uint>{};
    const std::uint32_t bound = countOnGpu ? count + 77 : count;
    if (Result recorded = check(sort.record(list, keyBuffer.getPtr(), valueBuffer.getPtr(), bound, keyBits, deviceCount)); !recorded)
        return recorded;
    if (Result submitted = submit(device, std::move(list)); !submitted)
        return submitted;
    std::expected<std::vector<Key>, std::string> sortedKeys = download(device, keyBuffer, count);
    std::expected<std::vector<uint>, std::string> sortedValues = download(device, valueBuffer, count);
    if (!sortedKeys || !sortedValues)
        return std::unexpected(!sortedKeys ? sortedKeys.error() : sortedValues.error());
    std::vector<uint> expected = values;
    std::ranges::stable_sort(expected, {}, [&](const uint index) { return keys[index]; });
    for (std::uint32_t index = 0; index < count; ++index)
        if ((*sortedValues)[index] != expected[index] || (*sortedKeys)[index] != keys[expected[index]])
            return std::unexpected(std::format("{}-bit keys, {} bits, {} pairs{}: wrong pair at {}", sizeof(Key) * 8, keyBits, count,
                                               countOnGpu ? ", count on the GPU" : "", index));
    return {};
}

}

Result testScan(Device& device) {
    Training::PrefixScan scan;
    if (Result opened = check(scan.open(device, benchmarkCount)); !opened)
        return opened;
    std::mt19937 random(3);
    for (const std::uint32_t count : std::array<std::uint32_t, 8>{1, 7, 2048, 2049, 4096 * 2048 + 5, 1000003, 777, benchmarkCount}) {
        std::vector<uint> values(count);
        for (uint& value : values)
            value = random() % 100;
        Buffer<uint> input;
        Buffer<uint> output;
        for (std::expected<void, Error> step : {input.open(device, count), output.open(device, count), input.upload(values)})
            if (Result result = check(step); !result)
                return result;
        CommandList list(device);
        if (Result recorded = check(scan.record(list, input.getPtr(), output.getPtr(), count)); !recorded)
            return recorded;
        if (Result submitted = submit(device, std::move(list)); !submitted)
            return submitted;
        std::expected<std::vector<uint>, std::string> scanned = download(device, output, count);
        if (!scanned)
            return std::unexpected(scanned.error());
        std::vector<uint> expected(count);
        std::exclusive_scan(values.begin(), values.end(), expected.begin(), 0u);
        if (*scanned != expected)
            return std::unexpected(std::format("the scan of {} values differs from std::exclusive_scan", count));
        std::println("scan of {} values: correct", count);
    }
    return {};
}

Result testSort(Device& device) {
    Training::RadixSort sort;
    if (Result opened = check(sort.open(device, benchmarkCount)); !opened)
        return opened;
    std::mt19937_64 random(5);
    for (const std::uint32_t count : sizes) {
        for (const std::uint32_t bits : {44u, 64u, 61u})
            if (Result result = checkSort<std::uint64_t>(device, sort, random, count, bits, false); !result)
                return result;
        for (const std::uint32_t bits : {32u, 20u, 9u})
            if (Result result = checkSort<std::uint32_t>(device, sort, random, count, bits, false); !result)
                return result;
        if (Result result = checkSort<std::uint64_t>(device, sort, random, count, 44, true); !result)
            return result;
        std::println("sort of {} pairs: correct for 64-bit keys (44, 64, 61 bits, and 44 bits with the count on the GPU) and 32-bit keys (32, 20, 9 bits)",
                     count);
    }
    std::vector<std::uint64_t> keys(benchmarkCount);
    std::vector<uint> values(benchmarkCount);
    Buffer<std::uint64_t> keyBuffer;
    Buffer<uint> valueBuffer;
    TimestampQuery timer;
    for (std::expected<void, Error> step : {keyBuffer.open(device, benchmarkCount), valueBuffer.open(device, benchmarkCount), timer.open(device)})
        if (Result result = check(step); !result)
            return result;
    for (const std::uint32_t bits : {32u, 44u, 64u}) {
        std::vector<double> times;
        for (int repeat = 0; repeat < benchmarkRepeats; ++repeat) {
            for (std::size_t index = 0; index < keys.size(); ++index) {
                keys[index] = random() & (bits == 64 ? ~0ull : (1ull << bits) - 1);
                values[index] = static_cast<uint>(index);
            }
            CommandList list(device);
            list.upload(keyBuffer, std::span<const std::uint64_t>(keys));
            list.upload(valueBuffer, std::span<const uint>(values));
            list.beginTimestamp(timer);
            if (Result recorded = check(sort.record(list, keyBuffer.getPtr(), valueBuffer.getPtr(), benchmarkCount, bits)); !recorded)
                return recorded;
            list.endTimestamp(timer);
            if (Result submitted = submit(device, std::move(list)); !submitted)
                return submitted;
            times.push_back(timer.getMilliseconds());
        }
        std::ranges::sort(times);
        const double median = times[times.size() / 2];
        std::println("sort of {} pairs with {}-bit keys (64-bit storage): median {:.2f} ms, {:.0f} M pairs/s", benchmarkCount, bits, median,
                     benchmarkCount / median / 1e3);
    }
    return {};
}

}
