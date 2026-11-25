#include "Optimizing.h"

static const uint noSource = 0xffffffff;

[shader("compute")]
[numthreads(256, 1, 1)]
void adam(uint3 id : SV_DispatchThreadID) {
    const Optimizing::AdamArguments arguments = rootArguments<Optimizing::AdamArguments>();
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        const float gradient = arguments.gradients[index];
        const float first = arguments.beta1 * arguments.first[index] + (1.0 - arguments.beta1) * gradient;
        const float second = arguments.beta2 * arguments.second[index] + (1.0 - arguments.beta2) * gradient * gradient;
        arguments.first[index] = first;
        arguments.second[index] = second;
        arguments.parameters[index] = arguments.parameters[index] - arguments.stepSize * first / (sqrt(second) * arguments.secondCorrection + arguments.epsilon);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gather(uint3 id : SV_DispatchThreadID) {
    const Optimizing::GatherArguments arguments = rootArguments<Optimizing::GatherArguments>();
    for (uint index = id.x; index < arguments.count * arguments.stride; index += arguments.threads) {
        const uint entry = index / arguments.stride;
        const uint source = arguments.indices[entry];
        arguments.destination[index] = source == noSource ? 0.0 : arguments.source[source * arguments.stride + index - entry * arguments.stride];
    }
}
