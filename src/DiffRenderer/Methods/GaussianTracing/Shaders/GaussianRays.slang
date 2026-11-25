#include "GaussianRays.h"
import GaussianList;
import RayGaussian;
import Scene.GaussianKernel;

typealias Arguments = GaussianRays::Arguments;
typealias Parameters = GaussianRays::Parameters;

static const uint gradientStride = 11 + shCount * 3;
static const float transmittanceEnd = 1e-3;

float3 load3(Ptr<float> values, uint index) {
    return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]);
}

void store3(Ptr<float> values, uint index, float3 value) {
    values[index * 3] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

float4 loadRotation(Parameters parameters, uint index) {
    return float4(parameters.rotations[index * 4], parameters.rotations[index * 4 + 1], parameters.rotations[index * 4 + 2],
                  parameters.rotations[index * 4 + 3]);
}

RayGaussian loadGaussian(Parameters parameters, uint index) {
    RayGaussian gaussian;
    gaussian.mean = load3(parameters.means, index);
    gaussian.rotation = loadRotation(parameters, index);
    gaussian.logScale = load3(parameters.scales, index);
    gaussian.logitOpacity = parameters.opacities[index];
    gaussian.sh[0] = load3(parameters.featuresDc, index);
    [ForceUnroll]
    for (uint coefficient = 1; coefficient < shCount; ++coefficient)
        gaussian.sh[coefficient] = load3(parameters.featuresRest, index * (shCount - 1) + coefficient - 1);
    return gaussian;
}

struct ModelSource : IGaussianSource {
    Parameters parameters;

    GaussianPeak findPeak(uint gaussian, float3 origin, float3 direction) {
        return findRayPeak(load3(parameters.means, gaussian), loadRotation(parameters, gaussian), load3(parameters.scales, gaussian),
                           parameters.opacities[gaussian], origin, direction);
    }
};

struct PixelRay {
    float3 origin;
    float3 direction;
    uint index;
};

bool getPixelRay(Arguments arguments, uint2 id, out PixelRay ray) {
    const GaussianRays::View view = arguments.view;
    const uint2 pixel = uint2(id.x, view.firstRow + id.y);
    ray = (PixelRay)0;
    if (id.x >= view.width || id.y >= view.rows || pixel.y >= view.height)
        return false;
    const float2 camera = (float2(pixel) + 0.5 - view.intrinsics.zw) / view.intrinsics.xy;
    ray.origin = view.position.xyz;
    ray.direction = normalize(camera.x * view.worldToCamera[0].xyz + camera.y * view.worldToCamera[1].xyz + view.worldToCamera[2].xyz);
    ray.index = pixel.y * view.width + pixel.x;
    return true;
}

interface IContributionVisitor {
    [mutating]
    void visit(uint index, RayGaussian gaussian, RaySample sample, float transmittance);
}

//! Composites the Gaussians in the order of their peaks and returns the transmittance that is left for the background.
float composite<V : IContributionVisitor>(Arguments arguments, PixelRay ray, inout V visitor) {
    ModelSource source;
    source.parameters = arguments.parameters;
    float transmittance = 1.0;
    float afterT = 0.0;
    uint afterGaussian = 0xffffffff;
    for (;;) {
        const GaussianBatch batch = nextGaussians(arguments.tlas, source, ray.origin, ray.direction, gaussianRayEnd, afterT, afterGaussian);
        for (uint entry = 0; entry < batch.count; ++entry) {
            const uint index = batch.entries[entry].gaussian;
            const RayGaussian gaussian = loadGaussian(arguments.parameters, index);
            const RaySample sample = sampleRayGaussian(gaussian, ray.origin, ray.direction, arguments.shDegree);
            visitor.visit(index, gaussian, sample, transmittance);
            transmittance *= 1.0 - sample.alpha;
            if (transmittance < transmittanceEnd)
                return transmittance;
        }
        if (!batch.isFull)
            return transmittance;
        afterT = batch.entries[batch.count - 1].t;
        afterGaussian = batch.entries[batch.count - 1].gaussian;
    }
    return transmittance;
}

struct RadianceSum : IContributionVisitor {
    float3 radiance;

    [mutating]
    void visit(uint index, RayGaussian gaussian, RaySample sample, float transmittance) { radiance += sample.color * sample.alpha * transmittance; }
};

//! The adjoint of the composite: dC/dalpha_i = T_i c_i - (C - P_i) / (1 - alpha_i), where P_i is the radiance of the Gaussians up to i.
struct RadianceAdjoint : IContributionVisitor {
    AtomicPtr<float> gradients;
    float3 origin;
    float3 direction;
    float3 radiance;
    float3 radianceGradient;
    float3 prefix;
    uint shDegree;

    [mutating]
    void visit(uint index, RayGaussian gaussian, RaySample sample, float transmittance) {
        const float3 contribution = sample.color * sample.alpha * transmittance;
        prefix += contribution;
        RaySample.Differential sampleGradient;
        sampleGradient.color = sample.alpha * transmittance * radianceGradient;
        sampleGradient.alpha = transmittance * dot(sample.color, radianceGradient) - dot(radiance - prefix, radianceGradient) / (1.0 - sample.alpha);
        var pair = diffPair(gaussian);
        bwd_diff(sampleRayGaussian)(pair, origin, direction, shDegree, sampleGradient);
        const AtomicPtr<float> base = gradients + index * gradientStride;
        [ForceUnroll]
        for (uint axis = 0; axis < 3; ++axis) {
            atomicAdd(base + axis, pair.d.mean[axis]);
            atomicAdd(base + 7 + axis, pair.d.logScale[axis]);
        }
        [ForceUnroll]
        for (uint component = 0; component < 4; ++component)
            atomicAdd(base + 3 + component, pair.d.rotation[component]);
        atomicAdd(base + 10, pair.d.logitOpacity);
        const uint used = (shDegree + 1) * (shDegree + 1);
        for (uint coefficient = 0; coefficient < used; ++coefficient)
            [ForceUnroll]
            for (uint channel = 0; channel < 3; ++channel)
                atomicAdd(base + 11 + coefficient * 3 + channel, pair.d.sh[coefficient][channel]);
    }
};

[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradients(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    for (uint index = id.x; index < arguments.count * gradientStride; index += arguments.threads)
        atomicStore(arguments.rayGradients + index, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traceForward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    PixelRay ray;
    if (!getPixelRay(arguments, id.xy, ray))
        return;
    RadianceSum sum;
    sum.radiance = float3(0.0);
    const float transmittance = composite(arguments, ray, sum);
    const float3 radiance = sum.radiance + transmittance * arguments.view.background.xyz;
    store3(arguments.radiance, ray.index, radiance);
    store3(arguments.image, ray.index, linearToSrgb(radiance));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traceBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    PixelRay ray;
    if (!getPixelRay(arguments, id.xy, ray))
        return;
    var radiance = diffPair(load3(arguments.radiance, ray.index));
    bwd_diff(linearToSrgb)(radiance, load3(arguments.imageGradient, ray.index));
    RadianceAdjoint adjoint;
    adjoint.gradients = arguments.rayGradients;
    adjoint.origin = ray.origin;
    adjoint.direction = ray.direction;
    adjoint.radiance = radiance.p;
    adjoint.radianceGradient = radiance.d;
    adjoint.prefix = float3(0.0);
    adjoint.shDegree = arguments.shDegree;
    composite(arguments, ray, adjoint);
}

float3 loadGradient3(Arguments arguments, uint base) {
    return float3(atomicLoad(arguments.rayGradients + base), atomicLoad(arguments.rayGradients + base + 1), atomicLoad(arguments.rayGradients + base + 2));
}

//! The densification statistic of 3DGS is the gradient of the projected mean in NDC; a shift of one pixel moves the mean by depth / focal length.
float getNdcGradient(Arguments arguments, float3 mean, float3 meanGradient) {
    const GaussianRays::View view = arguments.view;
    const float depth = dot(view.worldToCamera[2].xyz, mean) + view.worldToCamera[2].w;
    const float2 cameraGradient = float2(dot(view.worldToCamera[0].xyz, meanGradient), dot(view.worldToCamera[1].xyz, meanGradient));
    return length(cameraGradient * depth / view.intrinsics.xy * 0.5 * float2(view.width, view.height));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gatherGradients(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const Parameters gradients = arguments.gradients;
    for (uint index = id.x; index < arguments.count; index += arguments.threads) {
        const uint base = index * gradientStride;
        const float3 meanGradient = loadGradient3(arguments, base);
        store3(gradients.means, index, meanGradient);
        [ForceUnroll]
        for (uint component = 0; component < 4; ++component)
            gradients.rotations[index * 4 + component] = atomicLoad(arguments.rayGradients + base + 3 + component);
        store3(gradients.scales, index, loadGradient3(arguments, base + 7));
        const float opacityGradient = atomicLoad(arguments.rayGradients + base + 10);
        gradients.opacities[index] = opacityGradient;
        store3(gradients.featuresDc, index, loadGradient3(arguments, base + 11));
        [ForceUnroll]
        for (uint coefficient = 1; coefficient < shCount; ++coefficient)
            store3(gradients.featuresRest, index * (shCount - 1) + coefficient - 1, loadGradient3(arguments, base + 11 + coefficient * 3));
        if (arguments.accumulate == 0 || opacityGradient == 0.0)
            continue;
        arguments.gradientSums[index] += getNdcGradient(arguments, load3(arguments.parameters.means, index), meanGradient);
        arguments.gradientCounts[index] += 1.0;
    }
}
