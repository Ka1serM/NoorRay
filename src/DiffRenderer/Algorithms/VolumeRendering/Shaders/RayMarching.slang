#include "RayMarching.h"
#include "Volume.slang"

struct TransferFunction {
    Ptr<float4> entries;
    AtomicPtr<float> gradients;
    uint size;
};

struct Scene {
    Grid density;
    Grid tumor;
    TransferFunction transferFunction;
    float stepSize;
    float extinctionScale;
};

Scene getScene(RayMarching::Arguments arguments) {
    Scene scene;
    scene.density = {arguments.density, AtomicPtr<float>(), arguments.volumeSize};
    scene.tumor = {arguments.tumor, AtomicPtr<float>(), arguments.volumeSize};
    scene.transferFunction = {arguments.transferFunction, arguments.transferFunctionGradients, arguments.transferFunctionSize};
    scene.stepSize = arguments.stepSize;
    scene.extinctionScale = arguments.extinctionScale;
    return scene;
}

[Differentiable]
[BackwardDerivative(loadEntryBackward)]
float4 loadEntry(no_diff TransferFunction transferFunction, no_diff uint index) {
    return transferFunction.entries[index];
}

static const int noEntry = 1 << 20;
static int cachedEntries[2];
static float4 cachedGradients[2];

void clearEntryCache() {
    cachedEntries[0] = noEntry;
    cachedEntries[1] = noEntry;
}

void flushEntry(TransferFunction transferFunction, uint slot) {
    if (cachedEntries[slot] == noEntry)
        return;
    const AtomicPtr<float> target = transferFunction.gradients + uint(cachedEntries[slot]) * 4;
    atomicAdd(target, cachedGradients[slot].x);
    atomicAdd(target + 1, cachedGradients[slot].y);
    atomicAdd(target + 2, cachedGradients[slot].z);
    atomicAdd(target + 3, cachedGradients[slot].w);
    cachedEntries[slot] = noEntry;
}

void flushEntryCache(TransferFunction transferFunction) {
    if (isNull(transferFunction.gradients))
        return;
    flushEntry(transferFunction, 0);
    flushEntry(transferFunction, 1);
}

void loadEntryBackward(TransferFunction transferFunction, uint index, float4 gradient) {
    if (isNull(transferFunction.gradients))
        return;
    const int entry = int(index);
    for (uint slot = 0; slot < 2; ++slot) {
        if (cachedEntries[slot] != entry)
            continue;
        cachedGradients[slot] += gradient;
        return;
    }
    const uint slot = abs(cachedEntries[0] - entry) > abs(cachedEntries[1] - entry) ? 0 : 1;
    flushEntry(transferFunction, slot);
    cachedEntries[slot] = entry;
    cachedGradients[slot] = gradient;
}

[Differentiable]
float4 classify(no_diff TransferFunction transferFunction, float density) {
    const float position = clamp(density, 0.0, 1.0) * float(transferFunction.size - 1);
    const float base = min(no_diff floor(detach(position)), float(transferFunction.size - 2));
    const float fraction = position - base;
    const float4 low = loadEntry(transferFunction, uint(base));
    const float4 high = loadEntry(transferFunction, uint(base) + 1);
    return low + (high - low) * fraction;
}

[Differentiable]
float3 getDirection(float3 eye, no_diff float2 film) {
    const float3 forward = normalize(-eye);
    const float3 right = normalize(cross(forward, float3(0.0, 1.0, 0.0)));
    const float3 up = cross(right, forward);
    return normalize(forward + right * film.x + up * film.y);
}

[Differentiable]
float getEntry(float3 origin, float3 direction) {
    const float3 inverse = 1.0 / direction;
    const float3 low = (-0.5 - origin) * inverse;
    const float3 high = (0.5 - origin) * inverse;
    const float3 near = min(low, high);
    return max(max(max(near.x, near.y), near.z), 0.0);
}

float getExit(float3 origin, float3 direction) {
    const float3 inverse = 1.0 / direction;
    const float3 low = (-0.5 - origin) * inverse;
    const float3 high = (0.5 - origin) * inverse;
    const float3 far = max(low, high);
    return min(min(far.x, far.y), far.z);
}

[Differentiable]
float3 getPosition(float3 eye, no_diff float2 film, no_diff float offset) {
    const float3 direction = getDirection(eye, film);
    return eye + direction * (getEntry(eye, direction) + offset);
}

[Differentiable]
float4 shadeSample(no_diff Scene scene, float3 eye, no_diff float2 film, no_diff float offset) {
    const float density = sampleGrid(scene.density, toVoxel(getPosition(eye, film, offset), scene.density.size));
    const float4 entry = classify(scene.transferFunction, density);
    const float alpha = 1.0 - exp(-entry.w * scene.extinctionScale * scene.stepSize);
    return float4(entry.xyz, alpha);
}

struct Ray {
    float3 eye;
    float2 film;
    uint sampleCount;
};

Ray getRay(RayMarching::Arguments arguments, uint3 pixel) {
    const float2 ndc = (float2(pixel.xy) + 0.5) / float2(arguments.width, arguments.height) * 2.0 - 1.0;
    const float aspect = float(arguments.width) / float(arguments.height);
    Ray ray;
    ray.eye = arguments.eyes[pixel.z].xyz;
    ray.film = float2(ndc.x * aspect, -ndc.y) * arguments.tanHalfFov;
    const float3 direction = getDirection(ray.eye, ray.film);
    const float entry = getEntry(ray.eye, direction);
    const float exit = getExit(ray.eye, direction);
    ray.sampleCount = 0;
    while (ray.sampleCount < 4096 && entry + (float(ray.sampleCount) + 0.5) * arguments.stepSize < exit)
        ++ray.sampleCount;
    return ray;
}

float getOffset(Scene scene, uint index) {
    return (float(index) + 0.5) * scene.stepSize;
}

float getTumor(Scene scene, Ray ray, float offset) {
    return sampleGrid(scene.tumor, toVoxel(getPosition(ray.eye, ray.film, offset), scene.tumor.size));
}

float4 composite(Scene scene, Ray ray, float3 background) {
    float transmittance = 1.0;
    float4 color = float4(0.0);
    for (uint index = 0; index < ray.sampleCount; ++index) {
        const float offset = getOffset(scene, index);
        const float4 sample = shadeSample(scene, ray.eye, ray.film, offset);
        color += transmittance * sample.w * float4(sample.xyz, getTumor(scene, ray, offset));
        transmittance *= 1.0 - sample.w;
    }
    return color + transmittance * float4(background, 0.0);
}

uint getPixel(RayMarching::Arguments arguments, uint3 id) {
    return (id.z * arguments.height + id.y) * arguments.width + id.x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 id : SV_DispatchThreadID) {
    const RayMarching::Arguments arguments = rootArguments<RayMarching::Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Scene scene = getScene(arguments);
    arguments.image[getPixel(arguments, id)] = composite(scene, getRay(arguments, id), arguments.background.xyz);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void backward(uint3 id : SV_DispatchThreadID) {
    const RayMarching::Arguments arguments = rootArguments<RayMarching::Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Scene scene = getScene(arguments);
    const Ray ray = getRay(arguments, id);
    const float4 color = composite(scene, ray, arguments.background.xyz);
    const float4 gradient = arguments.imageGradients[getPixel(arguments, id)];
    float transmittance = 1.0;
    float4 accumulated = float4(0.0);
    clearEntryCache();
    for (uint index = 0; index < ray.sampleCount; ++index) {
        const float offset = getOffset(scene, index);
        const float4 sample = shadeSample(scene, ray.eye, ray.film, offset);
        const float4 emission = float4(sample.xyz, getTumor(scene, ray, offset));
        const float weight = transmittance * sample.w;
        accumulated += weight * emission;
        const float alphaGradient = dot(gradient, transmittance * emission - (color - accumulated) / (1.0 - sample.w));
        var eye = diffPair(ray.eye);
        bwd_diff(shadeSample)(scene, eye, ray.film, offset, float4(weight * gradient.xyz, alphaGradient));
        transmittance *= 1.0 - sample.w;
    }
    flushEntryCache(scene.transferFunction);
}
