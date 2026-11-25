#include "Radiograph.h"
#include "Volume.slang"

struct Segment : IDifferentiable {
    float3 source;
    float3 target;
    float length;
};

[Differentiable]
float3 rotateX(float3 v, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float3(v.x, c * v.y - s * v.z, s * v.y + c * v.z);
}

[Differentiable]
float3 rotateY(float3 v, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

[Differentiable]
float3 rotateZ(float3 v, float angle) {
    const float c = cos(angle);
    const float s = sin(angle);
    return float3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

[Differentiable]
float3 rotate(float3 v, float3 angles) {
    return rotateZ(rotateX(rotateY(v, angles.z), angles.y), angles.x);
}

[Differentiable]
Segment getSegment(float3 angles, float3 translation, no_diff float3 detector, no_diff uint volumeSize) {
    const float3 source = rotate(translation, angles);
    const float3 target = rotate(detector, angles) + source;
    Segment segment;
    segment.source = toVoxel(source, volumeSize);
    segment.target = toVoxel(target, volumeSize);
    segment.length = length(target - source);
    return segment;
}

[Differentiable]
float sampleLine(no_diff Grid grid, float3 source, float3 target, no_diff float alpha) {
    return sampleGrid(grid, source + alpha * (target - source));
}

float3 getDetector(Radiograph::Arguments arguments, uint2 pixel) {
    const float2 centered = float2(pixel) + 0.5 - 0.5 * float2(arguments.width, arguments.height);
    return float3(centered.x * arguments.pixelSpacing, -centered.y * arguments.pixelSpacing, arguments.sourceToDetector);
}

float getAlpha(Radiograph::Arguments arguments, uint index) {
    return arguments.alphaMin + (arguments.alphaMax - arguments.alphaMin) * (float(index) / float(arguments.pointCount - 1));
}

float getStep(Radiograph::Arguments arguments) {
    return (arguments.alphaMax - arguments.alphaMin) / float(arguments.pointCount - 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 id : SV_DispatchThreadID) {
    const Radiograph::Arguments arguments = rootArguments<Radiograph::Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Grid grid = {arguments.density, AtomicPtr<float>(), arguments.volumeSize};
    const Segment segment = getSegment(arguments.angles.xyz, arguments.translation.xyz, getDetector(arguments, id.xy), arguments.volumeSize);
    float sum = 0.0;
    for (uint index = 0; index < arguments.pointCount; ++index)
        sum += sampleLine(grid, segment.source, segment.target, getAlpha(arguments, index));
    arguments.image[id.y * arguments.width + id.x] = sum * segment.length * getStep(arguments);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void backward(uint3 id : SV_DispatchThreadID) {
    const Radiograph::Arguments arguments = rootArguments<Radiograph::Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Grid grid = {arguments.density, AtomicPtr<float>(), arguments.volumeSize};
    const float3 detector = getDetector(arguments, id.xy);
    const Segment segment = getSegment(arguments.angles.xyz, arguments.translation.xyz, detector, arguments.volumeSize);
    const float gradient = arguments.imageGradients[id.y * arguments.width + id.x] * getStep(arguments);
    const float sampleGradient = gradient * segment.length;
    Segment.Differential segmentGradient;
    segmentGradient.source = float3(0.0);
    segmentGradient.target = float3(0.0);
    float sum = 0.0;
    for (uint index = 0; index < arguments.pointCount; ++index) {
        const float alpha = getAlpha(arguments, index);
        var source = diffPair(segment.source);
        var target = diffPair(segment.target);
        sum += sampleLine(grid, segment.source, segment.target, alpha);
        bwd_diff(sampleLine)(grid, source, target, alpha, sampleGradient);
        segmentGradient.source += source.d;
        segmentGradient.target += target.d;
    }
    segmentGradient.length = gradient * sum;
    var angles = diffPair(arguments.angles.xyz);
    var translation = diffPair(arguments.translation.xyz);
    bwd_diff(getSegment)(angles, translation, detector, arguments.volumeSize, segmentGradient);
    const float3 angleGradient = WaveActiveSum(angles.d);
    const float3 translationGradient = WaveActiveSum(translation.d);
    if (!WaveIsFirstLane())
        return;
    atomicAdd(arguments.poseGradients, angleGradient.x);
    atomicAdd(arguments.poseGradients + 1, angleGradient.y);
    atomicAdd(arguments.poseGradients + 2, angleGradient.z);
    atomicAdd(arguments.poseGradients + 3, translationGradient.x);
    atomicAdd(arguments.poseGradients + 4, translationGradient.y);
    atomicAdd(arguments.poseGradients + 5, translationGradient.z);
}
