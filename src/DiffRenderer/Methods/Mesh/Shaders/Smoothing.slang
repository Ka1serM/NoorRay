#include "Smoothing.h"

typealias SolveArguments = Smoothing::SolveArguments;

static const uint groupSize = 256;

groupshared float3 groupSums[groupSize];

float3 loadVector(Ptr<float> values, uint index) {
    return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]);
}

void storeVector(Ptr<float> values, uint index, float3 value) {
    values[index * 3] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

float3 sumGroup(uint thread, float3 value) {
    groupSums[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = groupSize / 2; stride > 0; stride /= 2) {
        if (thread < stride)
            groupSums[thread] += groupSums[thread + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float3 sum = groupSums[0];
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

void storePartial(Ptr<float> partials, uint group, uint thread, float3 value) {
    const float3 sum = sumGroup(thread, value);
    if (thread == 0)
        storeVector(partials, group, sum);
}

float3 sumPartials(SolveArguments arguments, Ptr<float> partials, uint thread) {
    float3 sum = float3(0.0);
    for (uint group = thread; group < arguments.groupCount; group += groupSize)
        sum += loadVector(partials, group);
    return sumGroup(thread, sum);
}

float3 divide(float3 numerator, float3 denominator) {
    return select(denominator != 0.0, numerator / denominator, float3(0.0));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void startSolve(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const SolveArguments arguments = rootArguments<SolveArguments>();
    float3 squared = float3(0.0);
    if (id.x < arguments.vertexCount) {
        const float3 right = loadVector(arguments.right, id.x);
        storeVector(arguments.solution, id.x, float3(0.0));
        storeVector(arguments.residual, id.x, right);
        storeVector(arguments.direction, id.x, right);
        squared = right * right;
    }
    storePartial(arguments.nextResiduals, group.x, thread, squared);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void multiply(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const SolveArguments arguments = rootArguments<SolveArguments>();
    float3 product = float3(0.0);
    if (id.x < arguments.vertexCount) {
        const float3 direction = loadVector(arguments.direction, id.x);
        const uint begin = arguments.offsets[id.x];
        const uint end = arguments.offsets[id.x + 1];
        float3 laplacian = float(end - begin) * direction;
        for (uint entry = begin; entry < end; ++entry)
            laplacian -= loadVector(arguments.direction, arguments.neighbors[entry]);
        const float3 result = direction + arguments.lambda * laplacian;
        storeVector(arguments.product, id.x, result);
        product = direction * result;
    }
    storePartial(arguments.products, group.x, thread, product);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void update(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const SolveArguments arguments = rootArguments<SolveArguments>();
    const float3 alpha = divide(sumPartials(arguments, arguments.residuals, thread), sumPartials(arguments, arguments.products, thread));
    float3 squared = float3(0.0);
    if (id.x < arguments.vertexCount) {
        storeVector(arguments.solution, id.x, loadVector(arguments.solution, id.x) + alpha * loadVector(arguments.direction, id.x));
        const float3 residual = loadVector(arguments.residual, id.x) - alpha * loadVector(arguments.product, id.x);
        storeVector(arguments.residual, id.x, residual);
        squared = residual * residual;
    }
    storePartial(arguments.nextResiduals, group.x, thread, squared);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void advance(uint3 id : SV_DispatchThreadID, uint thread : SV_GroupIndex) {
    const SolveArguments arguments = rootArguments<SolveArguments>();
    const float3 beta = divide(sumPartials(arguments, arguments.nextResiduals, thread), sumPartials(arguments, arguments.residuals, thread));
    if (id.x < arguments.vertexCount)
        storeVector(arguments.direction, id.x, loadVector(arguments.residual, id.x) + beta * loadVector(arguments.direction, id.x));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void uniformMoments(uint3 id : SV_DispatchThreadID) {
    const Smoothing::AdamArguments arguments = rootArguments<Smoothing::AdamArguments>();
    if (id.x >= arguments.count)
        return;
    const float gradient = arguments.gradients[id.x];
    const float first = arguments.beta1 * arguments.first[id.x] + (1.0 - arguments.beta1) * gradient;
    const float second = arguments.beta2 * arguments.second[id.x] + (1.0 - arguments.beta2) * gradient * gradient;
    arguments.first[id.x] = first;
    arguments.second[id.x] = second;
    atomicMax(arguments.largestSecond, asuint(second));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void uniformStep(uint3 id : SV_DispatchThreadID) {
    const Smoothing::AdamArguments arguments = rootArguments<Smoothing::AdamArguments>();
    if (id.x >= arguments.count)
        return;
    const float largest = asfloat(atomicLoad(arguments.largestSecond));
    arguments.step[id.x] = -arguments.stepSize * arguments.first[id.x] / (sqrt(largest) * arguments.secondCorrection + arguments.epsilon);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void resetLargest(uint3 id : SV_DispatchThreadID) {
    const Smoothing::AdamArguments arguments = rootArguments<Smoothing::AdamArguments>();
    if (id.x == 0)
        atomicStore(arguments.largestSecond, 0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void applyOffsets(uint3 id : SV_DispatchThreadID) {
    const Smoothing::ApplyArguments arguments = rootArguments<Smoothing::ApplyArguments>();
    if (id.x < arguments.count)
        arguments.positions[id.x] = arguments.positions[id.x] + arguments.offsets[id.x];
}
