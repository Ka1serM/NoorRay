#include "NoorRay/Render/Restir/NeighborOffsets.h"

#include "Rtxdi/RtxdiUtils.h"

namespace NoorRay::Render {

std::vector<NoorRHI::float2> makeNeighborOffsets() {
    std::vector<std::uint8_t> packed(neighborOffsetCount * 2);
    rtxdi::FillNeighborOffsetBuffer(packed.data(), neighborOffsetCount);
    std::vector<NoorRHI::float2> offsets(neighborOffsetCount);
    //! RTXDI fills the offsets for an R8G8_SNORM buffer.
    for (std::uint32_t index = 0; index < neighborOffsetCount; ++index)
        offsets[index] = {static_cast<float>(static_cast<std::int8_t>(packed[index * 2])) / 127.0f,
                          static_cast<float>(static_cast<std::int8_t>(packed[index * 2 + 1])) / 127.0f};
    return offsets;
}

}
