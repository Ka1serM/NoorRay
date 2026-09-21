#include "Realtime/Accumulator.h"

#include "Realtime/ShaderLoading.h"

namespace
{
alignas(uint32_t) constexpr unsigned char accumulateSpv[] = {
    #embed "RealtimeRaytracer/Accumulate.spv"
};
// Matches Accumulate.slang.
constexpr uint32_t GroupSize = 8u;
}

Accumulator::Accumulator(noorrhi::Device& device)
    : pipeline_(device.compute(loadShader(device, accumulateSpv)))
{
}

void Accumulator::record(const nr::graphics::RealtimeArgs& args,
    const nr::graphics::RealtimeRoot root) const
{
    pipeline_.launch({divideRoundingUp(args.view.outputWidth, GroupSize),
        divideRoundingUp(args.view.outputHeight, GroupSize), 1}, root);
}
