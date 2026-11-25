import NoorRHI;
import Common.Math;
import GaussianList;
import Scene.GaussianKernel;
import SceneRecords;
import Spectral.Blackbody;
import Spectral.Wavelengths;

#include "GaussianPaths.h"

typealias Arguments = GaussianPaths::Arguments;

static const float minTransmittance = 1e-4;
static const uint maxBatches = 8;
static const uint maxVertices = 3;
static const uint noGaussian = 0xffffffff;
static const float rayOffset = 1e-4;

struct Random {
    uint state;

    __init(uint a, uint b, uint c) {
        state = hash(a ^ hash(b ^ hash(c)));
    }

    static uint hash(uint value) {
        const uint state = value * 747796405u + 2891336453u;
        const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    [mutating]
    float next() {
        state = hash(state);
        return float(state >> 8) * (1.0 / 16777216.0);
    }
};

struct Shape : IDifferentiable {
    float3 mean;
    float4 quaternion;
    float3 scale;
    float opacity;
};

[Differentiable]
GaussianPeak findShapePeak(Shape shape, float3 origin, float3 direction) {
    const float4 q = shape.quaternion / length(shape.quaternion);
    return findGaussianPeak(shape.mean, float4(q.y, q.z, q.w, q.x), shape.scale, shape.opacity, origin, direction);
}

[Differentiable]
float3 decodeColor(float3 encoded) { return decodeGaussianColor(encoded - 0.5); }

struct RayGradient {
    float3 origin;
    float3 direction;
};

struct Cloud : IGaussianSource {
    Ptr<float> parameters;
    AtomicPtr<float> gradients;
    AccelerationStructureHandle proxies;
    uint count;

    float3 load3(uint offset, uint gaussian) {
        const uint first = offset * count + 3 * gaussian;
        return float3(parameters[first], parameters[first + 1], parameters[first + 2]);
    }

    Shape loadShape(uint gaussian) {
        const uint quaternion = GaussianPaths::quaternionOffset * count + 4 * gaussian;
        Shape shape;
        shape.mean = load3(0, gaussian);
        shape.quaternion = float4(parameters[quaternion], parameters[quaternion + 1], parameters[quaternion + 2], parameters[quaternion + 3]);
        shape.scale = load3(GaussianPaths::scaleOffset, gaussian);
        shape.opacity = parameters[GaussianPaths::opacityOffset * count + gaussian];
        return shape;
    }

    float3 loadEncodedColor(uint gaussian) { return load3(GaussianPaths::colorOffset, gaussian); }

    GaussianPeak findPeak(uint gaussian, float3 origin, float3 direction) { return findShapePeak(loadShape(gaussian), origin, direction); }

    GaussianBatch next(float3 origin, float3 direction, float tMax, GaussianListEntry after) {
        return nextGaussians(proxies, this, origin, direction, tMax, after.t, after.gaussian);
    }

    void add(uint index, float value) {
        if (value != 0.0)
            atomicAdd(gradients + index, value);
    }

    void add3(uint offset, uint gaussian, float3 value) {
        for (uint axis = 0; axis < 3; ++axis)
            add(offset * count + 3 * gaussian + axis, value[axis]);
    }

    //! Back-propagates the gradients of the peak to the shape of the Gaussian; returns those of the ray.
    RayGradient addPeak(uint gaussian, float3 origin, float3 direction, float t, float alpha) {
        var shape = diffPair(loadShape(gaussian));
        var start = diffPair(origin);
        var heading = diffPair(direction);
        GaussianPeak.Differential peak;
        peak.t = t;
        peak.alpha = alpha;
        bwd_diff(findShapePeak)(shape, start, heading, peak);
        add3(0, gaussian, shape.d.mean);
        for (uint component = 0; component < 4; ++component)
            add(GaussianPaths::quaternionOffset * count + 4 * gaussian + component, shape.d.quaternion[component]);
        add3(GaussianPaths::scaleOffset, gaussian, shape.d.scale);
        add(GaussianPaths::opacityOffset * count + gaussian, shape.d.opacity);
        RayGradient gradient;
        gradient.origin = start.d;
        gradient.direction = heading.d;
        return gradient;
    }

    void addColor(uint gaussian, float3 gradient) {
        var encoded = diffPair(loadEncodedColor(gaussian));
        bwd_diff(decodeColor)(encoded, gradient);
        add3(GaussianPaths::colorOffset, gaussian, encoded.d);
    }
};

GaussianListEntry getListStart() {
    GaussianListEntry start;
    start.t = 0.0;
    start.gaussian = noGaussian;
    return start;
}

struct Surface {
    bool valid;
    float t;
    float3 normal;
    float3 albedo;
};

struct Scene {
    Arguments arguments;
    Cloud cloud;
    Wavelengths wavelengths;
    float4 lightSpectrum;

    __init(Arguments values, float hero) {
        arguments = values;
        cloud.parameters = values.parameters;
        cloud.gradients = values.gradients;
        cloud.proxies = values.proxies;
        cloud.count = values.count;
        Ptr<SpectralEntry> entries;
        entries.low = values.spectralEntries.low;
        entries.high = values.spectralEntries.high;
        if (values.spectral != 0)
            wavelengths = sampleWavelengths(entries, hero, false);
        else
            wavelengths = rgbWavelengths();
        for (uint index = 0; index < 4; ++index)
            lightSpectrum[index] = values.light.w * normalizedPlanck(wavelengths.lambda[index], values.environment.w);
    }

    float4 getEnvironment() { return wavelengths.upsample(arguments.environment.xyz); }

    float3 loadPosition(uint index) { return float3(arguments.positions[3 * index], arguments.positions[3 * index + 1], arguments.positions[3 * index + 2]); }

    Surface traceMesh(float3 origin, float3 direction) {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = 0.0;
        ray.TMax = gaussianRayEnd;
        RayHit hit;
        Surface surface = {};
        surface.valid = traceRayInline(arguments.mesh, RAY_FLAG_NONE, 0xff, ray, hit);
        surface.t = gaussianRayEnd;
        if (!surface.valid)
            return surface;
        const uint first = 3 * hit.primitiveIndex;
        const float3 a = loadPosition(arguments.indices[first]);
        const float3 b = loadPosition(arguments.indices[first + 1]);
        const float3 c = loadPosition(arguments.indices[first + 2]);
        const float3 normal = normalize(cross(b - a, c - a));
        surface.t = hit.t;
        surface.normal = dot(normal, direction) > 0.0 ? -normal : normal;
        surface.albedo = float3(arguments.albedos[first], arguments.albedos[first + 1], arguments.albedos[first + 2]);
        return surface;
    }

    bool isLitThroughMesh(float3 origin) {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = arguments.light.xyz - origin;
        ray.TMin = 0.0;
        ray.TMax = 1.0 - rayOffset;
        RayHit hit;
        return !traceRayInline(arguments.mesh, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, ray, hit);
    }
};

float3 getWeight(Arguments arguments, uint pixel) {
    return float3(arguments.weights[3 * pixel], arguments.weights[3 * pixel + 1], arguments.weights[3 * pixel + 2]) / float(arguments.samples);
}

//! The adjoint of toFilm: the gradient of the loss with respect to the radiance of each wavelength.
float4 getSpectralWeight(Wavelengths wavelengths, float3 weight) {
    float4 result;
    for (uint index = 0; index < 4; ++index)
        result[index] = dot(wavelengths.film(index), weight) * wavelengths.weights[index] * 0.25;
    return result;
}

//! The adjoint of upsample.
float3 getColorWeight(Wavelengths wavelengths, float4 weight) {
    return wavelengths.basis(0) * weight.x + wavelengths.basis(1) * weight.y + wavelengths.basis(2) * weight.z + wavelengths.basis(3) * weight.w;
}

float3 getCameraDirection(Arguments arguments, float2 screen) {
    const float2 ndc = float2(2.0 * screen.x - float(arguments.width), float(arguments.height) - 2.0 * screen.y) / float(arguments.height);
    return normalize(arguments.cameraForward.xyz + arguments.cameraOrigin.w * (ndc.x * arguments.cameraRight.xyz + ndc.y * arguments.cameraUp.xyz));
}

float3 sampleCamera(Arguments arguments, uint2 pixel, inout Random random) {
    const float2 offset = float2(random.next(), random.next()) - 0.5;
    return getCameraDirection(arguments, float2(pixel) + 0.5 + arguments.filterWidth * offset);
}

[Differentiable]
float getLightGeometry(float3 position, no_diff float3 light, no_diff float3 normal, no_diff bool surface) {
    const float3 toLight = light - position;
    const float distanceSquared = dot(toLight, toLight);
    if (surface)
        return max(dot(normal, toLight), 0.0) * invPi / (distanceSquared * sqrt(distanceSquared));
    return invFourPi / distanceSquared;
}
