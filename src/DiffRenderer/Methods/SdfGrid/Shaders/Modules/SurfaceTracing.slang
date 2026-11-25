#include "SurfaceGrid.slang"

static const uint maximumSteps = 4096;
static const uint hitRefinements = 8;
static const uint relaxedRefinements = 20;
static const float nearPlane = 0.05;
static const float minimumCosine = 0.05;
//! The directional derivative bound of a relaxed silhouette point, sdf_deriv_eps in the code of Wang et al. 2024.
static const float maximumGrazingCosine = 0.1;
static const float backgroundTolerance = 1.5 / 255.0;

struct Ray {
    float3 origin;
    float3 direction;
    float near;
    float far;

    float3 at(float distance) {
        return origin + distance * direction;
    }
}

Ray getRay(Arguments arguments, float2 film) {
    const float2 camera = (film - arguments.intrinsics.zw) / arguments.intrinsics.xy;
    Ray ray;
    ray.origin = arguments.cameraPosition.xyz;
    ray.direction = normalize(arguments.cameraForward.xyz + camera.x * arguments.cameraRight.xyz + camera.y * arguments.cameraDown.xyz);
    const float3 lower = arguments.grid.origin.xyz;
    const float3 upper = lower + float3(arguments.grid.size.xyz - 1) * arguments.grid.origin.w;
    const float3 inverse = 1.0 / ray.direction;
    const float3 near = (lower - ray.origin) * inverse;
    const float3 far = (upper - ray.origin) * inverse;
    ray.near = max(max(max(min(near.x, far.x), min(near.y, far.y)), min(near.z, far.z)), nearPlane);
    ray.far = min(min(max(near.x, far.x), max(near.y, far.y)), max(near.z, far.z));
    return ray;
}

float2 getFilm(Arguments arguments, uint2 pixel, uint sample) {
    Random random = Random(pixel.y * arguments.width + pixel.x, arguments.seed ^ (sample * 0x85ebca6bu));
    const float2 stratum = float2(float(sample % arguments.samplesPerSide), float(sample / arguments.samplesPerSide));
    return float2(pixel) + (stratum + float2(random.next(), random.next())) / float(arguments.samplesPerSide);
}

//! The first zero crossing of the SDF and, before it, the lowest local minimum of the SDF along the ray: the candidate for the relaxed silhouette point.
struct Trace {
    bool hit;
    //! False when the ray starts inside, so that the hit cannot move along the ray.
    bool crossing;
    float distance;
    bool bracketed;
    float lowest;
    float lowerBound;
    float upperBound;
}

float refineHit(Arguments arguments, Ray ray, float outside, float inside) {
    for (uint refinement = 0; refinement < hitRefinements; ++refinement) {
        const float middle = 0.5 * (outside + inside);
        if (sampleSdf(arguments, ray.at(middle)) > 0.0)
            outside = middle;
        else
            inside = middle;
    }
    return 0.5 * (outside + inside);
}

Trace traceSurface(Arguments arguments, Ray ray) {
    Trace trace;
    trace.hit = false;
    trace.crossing = false;
    trace.distance = ray.far;
    trace.bracketed = false;
    trace.lowest = 0.0;
    trace.lowerBound = 0.0;
    trace.upperBound = 0.0;
    if (ray.near >= ray.far)
        return trace;
    float before = ray.near;
    float distance = ray.near;
    float value = sampleSdf(arguments, ray.at(distance));
    float beforeValue = value;
    if (value <= 0.0) {
        trace.hit = true;
        trace.distance = distance;
        return trace;
    }
    for (uint count = 0; count < maximumSteps && distance < ray.far; ++count) {
        const float end = min(distance + max(value, arguments.step), ray.far);
        const float endValue = sampleSdf(arguments, ray.at(end));
        if (endValue <= 0.0) {
            trace.hit = true;
            trace.crossing = true;
            trace.distance = refineHit(arguments, ray, distance, end);
            return trace;
        }
        if (value < beforeValue && value <= endValue && (!trace.bracketed || value < trace.lowest)) {
            trace.bracketed = true;
            trace.lowest = value;
            trace.lowerBound = before;
            trace.upperBound = end;
        }
        before = distance;
        beforeValue = value;
        distance = end;
        value = endValue;
    }
    return trace;
}

//! Bisection on the directional derivative of the SDF, as in Wang et al. 2024, section 4; their code takes 20 steps.
float findRelaxedPoint(Arguments arguments, Ray ray, Trace trace) {
    float lower = trace.lowerBound;
    float upper = trace.upperBound;
    for (uint refinement = 0; refinement < relaxedRefinements; ++refinement) {
        const float middle = 0.5 * (lower + upper);
        if (dot(sampleSdfSlope(arguments, ray.at(middle)), ray.direction) > 0.0)
            upper = middle;
        else
            lower = middle;
    }
    return 0.5 * (lower + upper);
}

//! The SDF threshold epsilon of Wang et al. 2024 scaled to the footprint of bandPixels pixels at the distance; the band then has the same width in pixels
//! everywhere, and the distance cancels from the estimator as in their equation 15.
float getThreshold(Arguments arguments, float distance) {
    return arguments.bandPixels * distance / arguments.intrinsics.x;
}

float3 getSurfaceColour(Arguments arguments, float3 position, float basis[shCoefficients]) {
    return sigmoid(getColourLogits(arguments, getCell(arguments.grid, position), basis));
}

float3 getRadiance(Arguments arguments, Ray ray, Trace trace, float basis[shCoefficients]) {
    if (!trace.hit)
        return getBackground(arguments, ray.direction);
    return getSurfaceColour(arguments, ray.at(trace.distance), basis);
}

void writeImage(Arguments arguments, uint2 pixel, float3 colour) {
    const uint index = (pixel.y * arguments.width + pixel.x) * 3;
    arguments.image[index] = colour.x;
    arguments.image[index + 1] = colour.y;
    arguments.image[index + 2] = colour.z;
}

void renderPixel(Arguments arguments, uint2 pixel) {
    const uint samples = arguments.samplesPerSide * arguments.samplesPerSide;
    float3 sum = float3(0.0);
    for (uint sample = 0; sample < samples; ++sample) {
        const Ray ray = getRay(arguments, getFilm(arguments, pixel, sample));
        float basis[shCoefficients];
        getShBasis(ray.direction, arguments.activeCoefficients, basis);
        sum += getRadiance(arguments, ray, traceSurface(arguments, ray), basis);
    }
    writeImage(arguments, pixel, sum / float(samples));
}

bool showsBackground(Arguments arguments, uint pixel) {
    if (arguments.emptyWeight == 0.0 || arguments.background.w != 0.0)
        return false;
    const uint packed = arguments.target[pixel];
    const float3 target = float3(float(packed & 0xff), float((packed >> 8) & 0xff), float((packed >> 16) & 0xff)) / 255.0;
    return all(abs(target - arguments.background.xyz) <= backgroundTolerance);
}

//! The hit moves along the fixed ray by dt = -dSDF / (grad SDF . d), as in Vicini et al. 2022; the colour follows it.
void addHitGradient(Arguments arguments, Ray ray, Trace trace, float basis[shCoefficients], float3 gradient, float emptyGradient) {
    const Cell cell = getCell(arguments.grid, ray.at(trace.distance));
    const float3 colour = sigmoid(getColourLogits(arguments, cell, basis));
    const float3 logitGradient = gradient * colour * (1.0 - colour);
    addColourGradient(arguments, cell, basis, logitGradient);
    if (!trace.crossing)
        return;
    const float3 slope = sampleSlope(arguments.sdf, cell, arguments.grid.origin.w);
    const float cosine = dot(slope, ray.direction);
    if (cosine > -minimumCosine * length(slope))
        return;
    const float distanceGradient = dot(logitGradient, getColourLogitSlope(arguments, cell, basis, ray.direction));
    addScalarGradient(arguments.sdfGradient, cell, -distanceGradient / cosine - emptyGradient);
}

//! The relaxed boundary term of Wang et al. 2024, equation 16: (1 / epsilon) v(y*) (L(y*) - L), with the normal velocity v = -dSDF(y*) / |grad SDF(y*)|.
void addBoundaryGradient(Arguments arguments, Ray ray, Trace trace, float basis[shCoefficients], float3 gradient, float3 radiance) {
    if (!trace.bracketed)
        return;
    const float distance = findRelaxedPoint(arguments, ray, trace);
    const float3 position = ray.at(distance);
    const Cell cell = getCell(arguments.grid, position);
    const float value = sampleScalar(arguments.sdf, cell);
    const float threshold = getThreshold(arguments, distance);
    if (value <= 0.0 || value >= threshold)
        return;
    const float3 slope = sampleSlope(arguments.sdf, cell, arguments.grid.origin.w);
    const float slopeLength = length(slope);
    if (abs(dot(slope, ray.direction)) > maximumGrazingCosine * slopeLength)
        return;
    const float3 difference = getSurfaceColour(arguments, position, basis) - radiance;
    addScalarGradient(arguments.sdfGradient, cell, -dot(gradient, difference) / (threshold * max(slopeLength, 1e-3)));
}

void backwardPixel(Arguments arguments, uint2 pixel) {
    const uint index = pixel.y * arguments.width + pixel.x;
    const float3 gradient = float3(arguments.imageGradient[index * 3], arguments.imageGradient[index * 3 + 1], arguments.imageGradient[index * 3 + 2]);
    const float emptyGradient = showsBackground(arguments, index) ? arguments.emptyWeight / float(arguments.width * arguments.height) : 0.0;
    if (all(gradient == 0.0) && emptyGradient == 0.0)
        return;
    const uint samples = arguments.samplesPerSide * arguments.samplesPerSide;
    const float3 sampleGradient = gradient / float(samples);
    for (uint sample = 0; sample < samples; ++sample) {
        const Ray ray = getRay(arguments, getFilm(arguments, pixel, sample));
        float basis[shCoefficients];
        getShBasis(ray.direction, arguments.activeCoefficients, basis);
        const Trace trace = traceSurface(arguments, ray);
        if (trace.hit)
            addHitGradient(arguments, ray, trace, basis, sampleGradient, emptyGradient / float(samples));
        else
            addBackgroundGradient(arguments, ray.direction, sampleGradient);
        if (arguments.boundaryTerm != 0)
            addBoundaryGradient(arguments, ray, trace, basis, sampleGradient, getRadiance(arguments, ray, trace, basis));
    }
}

float getCurvature(Arguments arguments, int3 node, int3 axis) {
    if (!isNode(arguments.grid, node - axis) || !isNode(arguments.grid, node + axis))
        return 0.0;
    const Grid grid = arguments.grid;
    return arguments.sdf[getNodeIndex(grid, node - axis)] - 2.0 * arguments.sdf[getNodeIndex(grid, node)] + arguments.sdf[getNodeIndex(grid, node + axis)];
}

//! The squared second differences of the SDF along each axis, which keep the zero level set smooth between the views.
void addCurvatureGradient(Arguments arguments, uint index) {
    const Grid grid = arguments.grid;
    const float scale = 2.0 * arguments.curvatureWeight / (float(grid.size.w) * grid.origin.w * grid.origin.w);
    const int3 node = getNodePosition(grid, index);
    float gradient = 0.0;
    [ForceUnroll]
    for (uint axisIndex = 0; axisIndex < 3; ++axisIndex) {
        const int3 axis = int3(axisIndex == 0 ? 1 : 0, axisIndex == 1 ? 1 : 0, axisIndex == 2 ? 1 : 0);
        gradient += getCurvature(arguments, node - axis, axis) - 2.0 * getCurvature(arguments, node, axis) + getCurvature(arguments, node + axis, axis);
    }
    atomicAdd(arguments.sdfGradient + index, scale * gradient);
}

void resampleNode(Arguments arguments, uint index) {
    const Grid grid = arguments.grid;
    const Cell cell = getCell(arguments.source, grid.origin.xyz + float3(getNodePosition(grid, index)) * grid.origin.w);
    arguments.sdf[index] = sampleScalar(arguments.sourceSdf, cell);
    for (uint channel = 0; channel < colourChannels; ++channel) {
        float value = 0.0;
        [ForceUnroll]
        for (uint corner = 0; corner < 8; ++corner)
            value += cell.getWeight(corner) * arguments.sourceColour[cell.getCorner(corner) * colourChannels + channel];
        arguments.colour[index * colourChannels + channel] = value;
    }
}
