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
    //! The thread groups of the material kernel of bin, which advanceBins writes.
    NoorRHI::Ptr<NoorRHI::DispatchSize> getBinDispatch(const std::uint32_t bin) const { return binDispatch.getPtr() + bin; }
    //! The thread groups of a grid stage slot, which the advance phases write.
    NoorRHI::Ptr<NoorRHI::DispatchSize> getGridDispatch(const std::uint32_t slot) const { return gridDispatch.getPtr() + slot; }

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
    NoorRHI::Buffer<NoorRHI::DispatchSize> binDispatch;
    NoorRHI::Buffer<NoorRHI::DispatchSize> gridDispatch;
    Shaders::QueueRecord record{};
};

}
