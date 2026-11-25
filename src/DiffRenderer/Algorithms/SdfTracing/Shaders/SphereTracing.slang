#include "SphereTracing.h"

typealias Arguments = SphereTracing::Arguments;

static const uint stencilSize = 64;
static const uint maximumSteps = 2048;
static const uint refinementSteps = 10;
static const uint bisectionSteps = 12;
static const float boundsMargin = 0.05;
static const float traceEpsilon = 1e-6;
static const float sideOffset = 1e-3;
static const float depthJump = 0.05;
static const float missed = 3.0e38;
static const float pi = 3.14159265358979;
static const float filterRadius = 2.0;
static const float filterAlpha = -2.0;

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
}

struct Grid {
    Ptr<float> values;
    AtomicPtr<float> gradients;
    int resolution;
}

Grid getGrid(Arguments arguments) {
    return Grid(arguments.parameters + SphereTracing::gridOffset, arguments.gradients + SphereTracing::gridOffset, int(arguments.resolution));
}

//! Matches Dr.Jit's Texture3f.eval_cubic with clamped addressing: texel centers at (i + 0.5) / resolution, layout [z][y][x].
int3 getBase(Grid grid, float3 position) {
    return int3(floor(position * float(grid.resolution) - 0.5));
}

uint getTexel(Grid grid, int3 base, uint index) {
    const int3 offset = int3(int(index % 4), int((index / 4) % 4), int(index / 16)) - 1;
    const int3 texel = clamp(base + offset, int3(0), int3(grid.resolution - 1));
    return uint((texel.z * grid.resolution + texel.y) * grid.resolution + texel.x);
}

[Differentiable]
[BackwardDerivative(loadTexelBackward)]
float loadTexel(no_diff Grid grid, no_diff int3 base, no_diff uint index) {
    return grid.values[getTexel(grid, base, index)];
}

void loadTexelBackward(Grid grid, int3 base, uint index, float gradient) {
    if (gradient != 0.0)
        atomicAdd(grid.gradients + getTexel(grid, base, index), gradient);
}

[Differentiable]
float[stencilSize] loadStencil(no_diff Grid grid, no_diff int3 base) {
    float values[stencilSize];
    [ForceUnroll]
    for (uint index = 0; index < stencilSize; ++index)
        values[index] = loadTexel(grid, base, index);
    return values;
}

[Differentiable]
float4 getSplineWeights(float a) {
    const float b = 1.0 - a;
    return float4(b * b * b, 3.0 * a * a * a - 6.0 * a * a + 4.0, -3.0 * a * a * a + 3.0 * a * a + 3.0 * a + 1.0, a * a * a) / 6.0;
}

[Differentiable]
float4 getSplineDerivatives(float a) {
    const float b = 1.0 - a;
    return float4(-0.5 * b * b, 1.5 * a * a - 2.0 * a, -1.5 * a * a + a + 0.5, 0.5 * a * a);
}

//! Returns the distance in x and the gradient in yzw for a position in grid space ([0, 1]^3).
[Differentiable]
float4 evaluateStencil(float values[stencilSize], float3 position, no_diff int3 base, no_diff float resolution) {
    const float3 fraction = position * resolution - 0.5 - float3(base);
    const float4 weightsX = getSplineWeights(fraction.x);
    const float4 weightsY = getSplineWeights(fraction.y);
    const float4 weightsZ = getSplineWeights(fraction.z);
    const float4 derivativesX = getSplineDerivatives(fraction.x);
    const float4 derivativesY = getSplineDerivatives(fraction.y);
    const float4 derivativesZ = getSplineDerivatives(fraction.z);
    float4 result = float4(0.0);
    [ForceUnroll]
    for (uint index = 0; index < stencilSize; ++index) {
        const uint x = index % 4;
        const uint y = (index / 4) % 4;
        const uint z = index / 16;
        const float value = values[index];
        result += value * float4(weightsX[x] * weightsY[y] * weightsZ[z], derivativesX[x] * weightsY[y] * weightsZ[z] * resolution,
                                 weightsX[x] * derivativesY[y] * weightsZ[z] * resolution, weightsX[x] * weightsY[y] * derivativesZ[z] * resolution);
    }
    return result;
}

float4 evaluate(Grid grid, float3 position) {
    const int3 base = getBase(grid, position);
    return evaluateStencil(loadStencil(grid, base), position, base, float(grid.resolution));
}

float3 loadTranslation(Arguments arguments) {
    const Ptr<float> values = arguments.parameters + SphereTracing::translationOffset;
    return float3(values[0], values[1], values[2]);
}

float3 loadAlbedo(Arguments arguments) {
    const Ptr<float> values = arguments.parameters + SphereTracing::albedoOffset;
    return float3(values[0], values[1], values[2]);
}

[Differentiable]
float3 getCameraDirection(no_diff Arguments arguments, float2 screen) {
    const float tangent = arguments.cameraOrigin.w;
    const float2 ndc = float2(2.0 * screen.x / float(arguments.width) - 1.0, 1.0 - 2.0 * screen.y / float(arguments.height));
    return normalize(arguments.cameraForward.xyz + tangent * (ndc.x * arguments.cameraRight.xyz + ndc.y * arguments.cameraUp.xyz));
}

//! The stopping rule and the 10 refinement steps of SDFBase.ray_intersect_non_diff in Vicini et al.'s code.
float trace(Grid grid, float3 translation, float3 origin, float3 direction) {
    const float3 lower = translation - boundsMargin;
    const float3 upper = translation + 1.0 + boundsMargin;
    const float3 inverse = 1.0 / direction;
    const float3 near = (lower - origin) * inverse;
    const float3 far = (upper - origin) * inverse;
    const float entry = max(max(min(near.x, far.x), min(near.y, far.y)), min(near.z, far.z));
    const float exit = min(min(max(near.x, far.x), max(near.y, far.y)), max(near.z, far.z));
    if (entry > exit || exit <= 0.0)
        return missed;
    const float epsilon = traceEpsilon * max(exit, 1.0);
    float distance = max(entry, 0.0) + 1e-5;
    float hit = missed;
    for (uint step = 0; step < maximumSteps && distance <= exit; ++step) {
        const float value = evaluate(grid, origin + distance * direction - translation).x;
        if (value < epsilon) {
            hit = distance;
            break;
        }
        distance += abs(value);
    }
    if (hit == missed)
        return hit;
    for (uint step = 0; step < refinementSteps; ++step) {
        const float value = evaluate(grid, origin + hit * direction - translation).x;
        hit += value * 10.0 / float(10 + step);
        if (value > 0.0 && value <= epsilon)
            break;
    }
    return hit;
}

[Differentiable]
float3 shade(float3 albedo, float3 normal, no_diff float4 light) {
    return albedo * (light.w + max(dot(normal, light.xyz), 0.0));
}

//! Implicit differentiation t' = -phi' / (grad phi . d) keeps the camera ray fixed, as SDFBase.compute_surface_interaction does.
[Differentiable]
float3 shadeHit(float3 albedo, float3 translation, no_diff Grid grid, no_diff float3 origin, no_diff float3 direction, no_diff float distance, no_diff float4 light) {
    const float3 hit = origin + distance * direction;
    const int3 base = getBase(grid, hit - detach(translation));
    const float values[stencilSize] = loadStencil(grid, base);
    const float resolution = float(grid.resolution);
    const float4 sample = evaluateStencil(values, hit - translation, base, resolution);
    const float slope = detach(dot(sample.yzw, direction));
    const float moved = distance - (sample.x - detach(sample.x)) / slope;
    const float3 normal = normalize(evaluateStencil(values, origin + moved * direction - translation, base, resolution).yzw);
    return shade(albedo, normal, light);
}

float3 getRadiance(Arguments arguments, Grid grid, float3 translation, float3 albedo, float2 screen) {
    const float3 origin = arguments.cameraOrigin.xyz;
    const float3 direction = getCameraDirection(arguments, screen);
    const float distance = trace(grid, translation, origin, direction);
    if (distance == missed)
        return float3(0.0);
    const float3 normal = normalize(evaluate(grid, origin + distance * direction - translation).yzw);
    return shade(albedo, normal, arguments.light);
}

bool isInside(Arguments arguments, int2 pixel) {
    return all(pixel >= 0) && pixel.x < int(arguments.width) && pixel.y < int(arguments.height);
}

float3 loadWeight(Arguments arguments, int2 pixel) {
    const uint index = 3 * (uint(pixel.y) * arguments.pitch + uint(pixel.x));
    return float3(arguments.weights[index], arguments.weights[index + 1], arguments.weights[index + 2]);
}

void addReduced(Arguments arguments, uint offset, float3 value) {
    const float3 sum = WaveActiveSum(value);
    if (!WaveIsFirstLane())
        return;
    for (uint channel = 0; channel < 3; ++channel)
        if (sum[channel] != 0.0)
            atomicAdd(arguments.gradients + offset + channel, sum[channel]);
}

//! Mitsuba's gaussian reconstruction filter (stddev 0.5, radius 2, zero at the radius), as its film without border sampling.
float evaluateFilter(float offset) {
    return max(0.0, exp(filterAlpha * offset * offset) - exp(filterAlpha * filterRadius * filterRadius));
}

//! Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
float getErf(float value) {
    const float t = 1.0 / (1.0 + 0.3275911 * abs(value));
    const float polynomial = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign(value) * (1.0 - polynomial * exp(-value * value));
}

float integrateFilter(float lower, float upper) {
    const float scale = sqrt(-filterAlpha);
    return 0.5 * sqrt(pi) / scale * (getErf(upper * scale) - getErf(lower * scale)) - exp(filterAlpha * filterRadius * filterRadius) * (upper - lower);
}

struct Footprint {
    float2 lower;
    float2 upper;
    float2 center;
}

Footprint getFootprint(Arguments arguments, int2 pixel) {
    const float2 center = float2(pixel) + 0.5;
    return Footprint(max(center - filterRadius, float2(0.0)), min(center + filterRadius, float2(float(arguments.width), float(arguments.height))), center);
}

float2 sampleFootprint(Footprint footprint, inout Random random, out float weight) {
    const float2 screen = footprint.lower + (footprint.upper - footprint.lower) * float2(random.next(), random.next());
    weight = evaluateFilter(screen.x - footprint.center.x) * evaluateFilter(screen.y - footprint.center.y);
    return screen;
}

//! The loss sum_p W_p I_p equals the integral of f(u) times this weight field, which is what the boundary term needs.
float3 getWeightField(Arguments arguments, float2 screen) {
    float3 sum = float3(0.0);
    const int2 first = int2(ceil(screen - filterRadius - 0.5));
    for (int y = first.y; y <= first.y + 4; ++y)
        for (int x = first.x; x <= first.x + 4; ++x) {
            const int2 pixel = int2(x, y);
            if (!isInside(arguments, pixel))
                continue;
            const Footprint footprint = getFootprint(arguments, pixel);
            const float filter = evaluateFilter(screen.x - footprint.center.x) * evaluateFilter(screen.y - footprint.center.y);
            if (filter == 0.0)
                continue;
            const float2 lower = footprint.lower - footprint.center;
            const float2 upper = footprint.upper - footprint.center;
            sum += loadWeight(arguments, pixel) * filter / (integrateFilter(lower.x, upper.x) * integrateFilter(lower.y, upper.y));
        }
    return sum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void renderMain(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Grid grid = getGrid(arguments);
    const float3 translation = loadTranslation(arguments);
    const float3 albedo = loadAlbedo(arguments);
    const uint pixel = id.y * arguments.width + id.x;
    const Footprint footprint = getFootprint(arguments, int2(id.xy));
    float3 sum = float3(0.0);
    float weights = 0.0;
    for (uint sample = 0; sample < arguments.samples; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        float weight;
        const float2 screen = sampleFootprint(footprint, random, weight);
        if (weight == 0.0)
            continue;
        sum += weight * getRadiance(arguments, grid, translation, albedo, screen);
        weights += weight;
    }
    const float3 color = weights > 0.0 ? sum / weights : float3(0.0);
    for (uint channel = 0; channel < 3; ++channel)
        arguments.image[3 * (id.y * arguments.pitch + id.x) + channel] = color[channel];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void interiorMain(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const bool inside = id.x < arguments.width && id.y < arguments.height;
    const Grid grid = getGrid(arguments);
    const float3 translation = loadTranslation(arguments);
    const float3 albedo = loadAlbedo(arguments);
    const uint pixel = id.y * arguments.width + id.x;
    const Footprint footprint = getFootprint(arguments, int2(id.xy));
    float weights = 0.0;
    for (uint sample = 0; inside && sample < arguments.samples; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        float weight;
        sampleFootprint(footprint, random, weight);
        weights += weight;
    }
    const float3 pixelWeight = inside && weights > 0.0 ? loadWeight(arguments, int2(id.xy)) / weights : float3(0.0);
    const float3 origin = arguments.cameraOrigin.xyz;
    float3 albedoGradient = float3(0.0);
    float3 translationGradient = float3(0.0);
    for (uint sample = 0; inside && sample < arguments.samples; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        float weight;
        const float2 screen = sampleFootprint(footprint, random, weight);
        if (weight == 0.0)
            continue;
        const float3 direction = getCameraDirection(arguments, screen);
        const float distance = trace(grid, translation, origin, direction);
        if (distance == missed)
            continue;
        var albedoPair = diffPair(albedo);
        var translationPair = diffPair(translation);
        bwd_diff(shadeHit)(albedoPair, translationPair, grid, origin, direction, distance, arguments.light, pixelWeight * weight);
        albedoGradient += albedoPair.d;
        translationGradient += translationPair.d;
    }
    addReduced(arguments, SphereTracing::albedoOffset, albedoGradient);
    addReduced(arguments, SphereTracing::translationOffset, translationGradient);
}

struct LineSample {
    float distance;
    float3 direction;
}

LineSample traceLine(Arguments arguments, Grid grid, float3 translation, float2 screen) {
    const float3 direction = getCameraDirection(arguments, screen);
    return LineSample(trace(grid, translation, arguments.cameraOrigin.xyz, direction), direction);
}

bool isNearSide(float distance, float leftDistance, float rightDistance) {
    if (leftDistance == missed || rightDistance == missed)
        return (distance == missed) == (leftDistance == missed);
    return abs(distance - leftDistance) < abs(distance - rightDistance);
}

bool isDiscontinuous(float leftDistance, float rightDistance) {
    if (leftDistance == missed || rightDistance == missed)
        return leftDistance != rightDistance;
    return abs(leftDistance - rightDistance) > depthJump;
}

float getWeightedRadiance(Arguments arguments, Grid grid, float3 translation, float3 albedo, float2 screen) {
    const float3 weight = getWeightField(arguments, screen);
    if (all(weight == 0.0))
        return 0.0;
    return dot(weight, getRadiance(arguments, grid, translation, albedo, screen));
}

[Differentiable]
float getSilhouetteDistance(float3 translation, no_diff Grid grid, no_diff float3 position) {
    const int3 base = getBase(grid, position - detach(translation));
    return evaluateStencil(loadStencil(grid, base), position - translation, base, float(grid.resolution)).x;
}

//! Each silhouette crossing of a scanline adds (f_left - f_right) ds*/dtheta n_s^2; the two line directions sum to n^2 = 1.
[shader("compute")]
[numthreads(64, 1, 1)]
void boundaryMain(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const uint size = arguments.width;
    const uint linesPerAxis = size * arguments.linesPerPixel;
    if (id.x >= size || id.y >= 2 * linesPerAxis)
        return;
    const uint axis = id.y / linesPerAxis;
    const uint line = id.y % linesPerAxis;
    Random random = Random(line, axis, arguments.seed);
    const float across = (float(line) + random.next()) / float(arguments.linesPerPixel);
    const Grid grid = getGrid(arguments);
    const float3 translation = loadTranslation(arguments);
    const float3 albedo = loadAlbedo(arguments);
    float2 left = axis == 0 ? float2(float(id.x), across) : float2(across, float(id.x));
    float2 right = left;
    right[axis] += 1.0;
    LineSample leftSample = traceLine(arguments, grid, translation, left);
    LineSample rightSample = traceLine(arguments, grid, translation, right);
    if (!isDiscontinuous(leftSample.distance, rightSample.distance))
        return;
    for (uint step = 0; step < bisectionSteps; ++step) {
        float2 middle = left;
        middle[axis] = 0.5 * (left[axis] + right[axis]);
        const LineSample middleSample = traceLine(arguments, grid, translation, middle);
        if (isNearSide(middleSample.distance, leftSample.distance, rightSample.distance)) {
            left = middle;
            leftSample = middleSample;
        } else {
            right = middle;
            rightSample = middleSample;
        }
    }
    if (!isDiscontinuous(leftSample.distance, rightSample.distance))
        return;
    const bool leftNear = leftSample.distance < rightSample.distance;
    const LineSample nearSample = leftNear ? leftSample : rightSample;
    const float2 nearScreen = leftNear ? left : right;
    const float3 position = arguments.cameraOrigin.xyz + nearSample.distance * nearSample.direction;
    const float3 gradient = evaluate(grid, position - translation).yzw;
    float2 screenGradient;
    for (uint component = 0; component < 2; ++component) {
        float2 unit = float2(0.0);
        unit[component] = 1.0;
        screenGradient[component] = nearSample.distance * dot(gradient, fwd_diff(getCameraDirection)(arguments, diffPair(nearScreen, unit)).d);
    }
    float2 crossing = left;
    crossing[axis] = 0.5 * (left[axis] + right[axis]);
    float2 offset = float2(0.0);
    offset[axis] = sideOffset;
    const float difference = getWeightedRadiance(arguments, grid, translation, albedo, crossing - offset) -
                             getWeightedRadiance(arguments, grid, translation, albedo, crossing + offset);
    const float scale = -difference * screenGradient[axis] / dot(screenGradient, screenGradient) / float(arguments.linesPerPixel);
    if (scale == 0.0 || !isfinite(scale))
        return;
    var translationPair = diffPair(translation);
    bwd_diff(getSilhouetteDistance)(translationPair, grid, position, scale);
    for (uint channel = 0; channel < 3; ++channel)
        atomicAdd(arguments.gradients + SphereTracing::translationOffset + channel, translationPair.d[channel]);
}
