#include "GridTracing.h"
#include "Voxels.slang"

typealias Arguments = GridTracing::Arguments;

static const float stopTransmittance = 1e-4;
static const float infinity = 1e30;

struct Ray {
    float3 origin;
    float3 direction;
    float near;
    float far;
};

Ray getRay(Arguments arguments, uint2 pixel) {
    const float2 film = (float2(pixel) + 0.5 - arguments.intrinsics.zw) / arguments.intrinsics.xy;
    const float3 world = film.x * arguments.rows[0].xyz + film.y * arguments.rows[1].xyz + arguments.rows[2].xyz;
    Ray ray;
    ray.origin = arguments.origin.xyz;
    ray.direction = normalize(world);
    ray.near = 0.0;
    ray.far = infinity;
    if (arguments.contracted != 0)
        return ray;
    const float3 inverse = 1.0 / ray.direction;
    const float3 low = (arguments.placement.xyz - 1.0 / arguments.placement.w - ray.origin) * inverse;
    const float3 high = (arguments.placement.xyz + 1.0 / arguments.placement.w - ray.origin) * inverse;
    const float3 near = min(low, high);
    const float3 far = max(low, high);
    ray.near = max(max(max(near.x, near.y), near.z), 0.0);
    ray.far = min(min(far.x, far.y), far.z);
    return ray;
}

float3 getLocal(Arguments arguments, Ray ray, float t) {
    return (ray.origin + ray.direction * t - arguments.placement.xyz) * arguments.placement.w;
}

//! The contraction of mip-NeRF 360 maps the outside of the unit ball into the shell of radius 2; the grid spans radius 2.
float3 toGrid(Arguments arguments, float3 local) {
    if (arguments.contracted == 0)
        return local;
    const float radius = length(local);
    if (radius <= 1.0)
        return local * 0.5;
    return local * ((2.0 - 1.0 / radius) / radius * 0.5);
}

struct March {
    float t;
    float step;
    float3 position;
    float3 next;
    float delta;
    bool done;
};

float getWorldStep(Arguments arguments, float3 local) {
    if (arguments.contracted == 0)
        return arguments.stepSize / arguments.placement.w;
    return 2.0 * arguments.stepSize / arguments.placement.w * max(1.0, dot(local, local));
}

void updateMarch(Arguments arguments, Ray ray, inout March march) {
    const float3 local = getLocal(arguments, ray, march.t);
    march.step = getWorldStep(arguments, local);
    march.position = toGrid(arguments, local);
    march.next = toGrid(arguments, getLocal(arguments, ray, march.t + march.step));
    march.delta = length(march.next - march.position);
    march.done = march.t > ray.far || (arguments.contracted != 0 && length(local) * arguments.stepSize > 0.5);
}

March beginMarch(Arguments arguments, Ray ray) {
    March march;
    march.t = ray.near;
    if (arguments.contracted == 0)
        march.t += 0.5 * getWorldStep(arguments, float3(0.0));
    updateMarch(arguments, ray, march);
    return march;
}

void advanceMarch(Arguments arguments, Ray ray, inout March march) {
    march.t += march.step;
    updateMarch(arguments, ray, march);
}

struct Sample {
    Corners corners;
    float density;
    float alpha;
    float sh[shCount];
    float3 raw;
    float3 color;
};

bool shadeSample(Arguments arguments, March march, float basis[shBasisCount], out Sample sample) {
    const float3 voxel = toVoxel(arguments.grid.resolution, march.position);
    sample = {};
    if (!isOccupied(arguments.grid, int3(floor(voxel))))
        return false;
    sample.corners = getCorners(arguments.grid, voxel);
    if (sample.corners.empty)
        return false;
    sample.density = sample.corners.getDensity(arguments.grid);
    if (sample.density <= 0.0)
        return false;
    sample.alpha = 1.0 - exp(-sample.density * march.delta);
    sample.corners.getSh(arguments.grid, sample.sh);
    sample.raw = getRawColor(sample.sh, basis);
    sample.color = max(sample.raw, 0.0);
    return true;
}

uint getPixel(Arguments arguments, uint2 pixel) {
    return (pixel.y * arguments.width + pixel.x) * 3;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Ray ray = getRay(arguments, id.xy);
    float basis[shBasisCount];
    getShBasis(ray.direction, basis);
    float3 color = float3(0.0);
    float transmittance = 1.0;
    March march = beginMarch(arguments, ray);
    for (uint index = 0; index < arguments.maxSamples && !march.done; ++index, advanceMarch(arguments, ray, march)) {
        Sample sample;
        if (!shadeSample(arguments, march, basis, sample))
            continue;
        color += transmittance * sample.alpha * sample.color;
        transmittance *= 1.0 - sample.alpha;
        if (transmittance < stopTransmittance)
            break;
    }
    color += transmittance * arguments.background.xyz;
    const uint pixel = getPixel(arguments, id.xy);
    arguments.image[pixel] = color.x;
    arguments.image[pixel + 1] = color.y;
    arguments.image[pixel + 2] = color.z;
}

void scatter(Arguments arguments, Sample sample, float densityGradient, float3 colorGradient, float basis[shBasisCount], float weight) {
    for (uint corner = 0; corner < 8; ++corner) {
        const int link = sample.corners.links[corner];
        if (link < 0)
            continue;
        const float share = sample.corners.weights[corner];
        if (share <= 0.0)
            continue;
        atomicAdd(arguments.densityGradients + uint(link), share * densityGradient);
        const AtomicPtr<float> sh = arguments.shGradients + uint(link) * shCount;
        for (uint channel = 0; channel < 3; ++channel) {
            const float gradient = share * colorGradient[channel];
            if (gradient == 0.0)
                continue;
            for (uint index = 0; index < shBasisCount; ++index)
                atomicAdd(sh + channel * shBasisCount + index, gradient * basis[index]);
        }
        if (!isNull(arguments.maxWeights))
            atomicMax(arguments.maxWeights + uint(link), asuint(weight));
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void backward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const uint pixel = getPixel(arguments, id.xy);
    const float3 color = float3(arguments.image[pixel], arguments.image[pixel + 1], arguments.image[pixel + 2]);
    const float3 gradient = float3(arguments.imageGradients[pixel], arguments.imageGradients[pixel + 1], arguments.imageGradients[pixel + 2]);
    if (all(gradient == 0.0) && isNull(arguments.maxWeights))
        return;
    const Ray ray = getRay(arguments, id.xy);
    float basis[shBasisCount];
    getShBasis(ray.direction, basis);
    float3 accumulated = float3(0.0);
    float transmittance = 1.0;
    March march = beginMarch(arguments, ray);
    for (uint index = 0; index < arguments.maxSamples && !march.done; ++index, advanceMarch(arguments, ray, march)) {
        Sample sample;
        if (!shadeSample(arguments, march, basis, sample))
            continue;
        const float weight = transmittance * sample.alpha;
        accumulated += weight * sample.color;
        transmittance *= 1.0 - sample.alpha;
        const float densityGradient = march.delta * dot(gradient, transmittance * sample.color - (color - accumulated));
        const float3 colorGradient = select(sample.raw > 0.0, weight * gradient, float3(0.0));
        scatter(arguments, sample, densityGradient, colorGradient, basis, weight);
        if (transmittance < stopTransmittance)
            break;
    }
}
