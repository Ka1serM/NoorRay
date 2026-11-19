#pragma once

#include <cstdint>
#include <expected>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"

namespace NoorRay::Render {

//! The GPU-resident queues and counters of the wavefront stages. The CPU never reads them back.
class Queues {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(NoorRHI::Device& device, std::uint32_t pathCapacity, std::uint32_t sampleCapacity,
                                                        std::uint32_t binCount);

    const Shaders::QueueRecord& getRecord() const { return record; }
    std::uint32_t getBinCount() const { return record.binCount; }
    const NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>>& getCounters() const { return counters; }

private:
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> counters;
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> binCounts;
    NoorRHI::Buffer<NoorRHI::Atomic<std::uint32_t>> binCursors;
    NoorRHI::Buffer<std::uint32_t> active0;
    NoorRHI::Buffer<std::uint32_t> active1;
    NoorRHI::Buffer<std::uint32_t> shadow0;
    NoorRHI::Buffer<std::uint32_t> shadow1;
    NoorRHI::Buffer<std::uint32_t> shade;
    NoorRHI::Buffer<std::uint32_t> free;
    NoorRHI::Buffer<NoorRHI::uint2> samples;
    Shaders::QueueRecord record{};
};

}
