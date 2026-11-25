#include "GaussianInstances.h"

import Scene.GaussianKernel;

static const uint recordWords = 16;

float3 load3(Ptr<float> values, uint index) { return float3(values[3 * index], values[3 * index + 1], values[3 * index + 2]); }

float4 loadRotation(Ptr<float> values, uint index) {
    return normalize(float4(values[4 * index + 1], values[4 * index + 2], values[4 * index + 3], values[4 * index]));
}

void storeFloat(Ptr<uint> words, uint index, float value) { words[index] = asuint(value); }

void writeInstance(Instances::Arguments arguments, uint gaussian) {
    const float opacity = 1.0 / (1.0 + exp(-arguments.opacityLogits[gaussian]));
    const float cutoff = sqrt(gaussianCutoffSquared(opacity));
    const bool visible = cutoff > 0.0;
    const float3 scale = exp(load3(arguments.logScales, gaussian)) * (visible ? cutoff : 1.0);
    const float4 rotation = loadRotation(arguments.quaternions, gaussian);
    const float3 mean = load3(arguments.means, gaussian);
    const float3 unitX = rotateByInverse(rotation, float3(1.0, 0.0, 0.0));
    const float3 unitY = rotateByInverse(rotation, float3(0.0, 1.0, 0.0));
    const float3 unitZ = rotateByInverse(rotation, float3(0.0, 0.0, 1.0));
    const Ptr<uint> words = arguments.records + gaussian * recordWords;
    const float3 rows[3] = { unitX, unitY, unitZ };
    for (uint row = 0; row < 3; ++row) {
        const float3 values = rows[row] * scale;
        storeFloat(words, 4 * row, values.x);
        storeFloat(words, 4 * row + 1, values.y);
        storeFloat(words, 4 * row + 2, values.z);
        storeFloat(words, 4 * row + 3, mean[row]);
    }
    words[12] = visible ? 0xff000000u : 0u;
    words[13] = 0;
    words[14] = arguments.proxy.low;
    words[15] = arguments.proxy.high;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void writeInstances(uint3 id : SV_DispatchThreadID) {
    const Instances::Arguments arguments = rootArguments<Instances::Arguments>();
    for (uint gaussian = id.x; gaussian < arguments.count; gaussian += arguments.threads)
        writeInstance(arguments, gaussian);
}
