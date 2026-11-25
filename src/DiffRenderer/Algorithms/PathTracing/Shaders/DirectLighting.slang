#include "DirectLighting.h"

typealias Arguments = DirectLighting::Arguments;

static const float pi = 3.14159265358979;
static const float rayOffset = 1e-4;
static const float screenOffset = 1e-3;
static const float lightOffset = 2e-4;

struct Parameters : IDifferentiable {
    float3 albedoObject;
    float3 albedoGround;
    float3 translation;
};

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

struct Light {
    float3 corner;
    float3 edgeU;
    float3 edgeV;
    float3 normal;
    float3 radiance;
    float area;

    __init(Arguments arguments) {
        corner = arguments.lightCorner.xyz;
        edgeU = arguments.lightEdgeU.xyz;
        edgeV = arguments.lightEdgeV.xyz;
        const float3 product = cross(edgeU, edgeV);
        area = length(product);
        normal = product / area;
        radiance = arguments.lightRadiance.xyz;
    }

    float3 getPoint(float2 uv) {
        return corner + uv.x * edgeU + uv.y * edgeV;
    }

    float2 getCoordinates(float3 position) {
        const float3 offset = position - corner;
        return float2(dot(offset, edgeU) / dot(edgeU, edgeU), dot(offset, edgeV) / dot(edgeV, edgeV));
    }
}

struct Triangle {
    float3 vertices[3];
    float3 normals[3];
    bool object;
    bool valid;
}

float3 loadPosition(Ptr<float> positions, uint index) {
    return float3(positions[3 * index], positions[3 * index + 1], positions[3 * index + 2]);
}

float3 loadObjectPosition(Arguments arguments, uint index, float3 translation) {
    return loadPosition(arguments.objectPositions, index) + translation;
}

float3 getFaceNormal(Arguments arguments, uint face) {
    const float3 a = loadPosition(arguments.objectPositions, arguments.objectIndices[3 * face]);
    const float3 b = loadPosition(arguments.objectPositions, arguments.objectIndices[3 * face + 1]);
    const float3 c = loadPosition(arguments.objectPositions, arguments.objectIndices[3 * face + 2]);
    return normalize(cross(b - a, c - a));
}

Parameters loadParameters(Ptr<float> values) {
    Parameters parameters;
    parameters.albedoObject = float3(values[0], values[1], values[2]);
    parameters.albedoGround = float3(values[3], values[4], values[5]);
    parameters.translation = float3(values[6], values[7], values[8]);
    return parameters;
}

float3 getCameraDirection(Arguments arguments, float2 screen) {
    const float tangent = arguments.cameraOrigin.w;
    const float2 ndc = float2(2.0 * screen.x / float(arguments.width) - 1.0, 1.0 - 2.0 * screen.y / float(arguments.height));
    return normalize(arguments.cameraForward.xyz + tangent * (ndc.x * arguments.cameraRight.xyz + ndc.y * arguments.cameraUp.xyz));
}

[Differentiable]
float2 projectToScreen(no_diff Arguments arguments, float3 position) {
    const float3 offset = position - arguments.cameraOrigin.xyz;
    const float depth = dot(offset, arguments.cameraForward.xyz) * arguments.cameraOrigin.w;
    const float2 ndc = float2(dot(offset, arguments.cameraRight.xyz), dot(offset, arguments.cameraUp.xyz)) / depth;
    return float2((ndc.x + 1.0) * 0.5 * float(arguments.width), (1.0 - ndc.y) * 0.5 * float(arguments.height));
}

Triangle trace(Arguments arguments, float3 origin, float3 direction, float3 translation) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 1e4;
    RayHit hit;
    Triangle triangle = {};
    triangle.valid = traceRayInline(arguments.scene, RAY_FLAG_NONE, 0xff, ray, hit);
    if (!triangle.valid)
        return triangle;
    triangle.object = hit.instanceId == DirectLighting::objectInstance;
    for (uint corner = 0; corner < 3; ++corner) {
        if (triangle.object) {
            const uint index = arguments.objectIndices[3 * hit.primitiveIndex + corner];
            triangle.vertices[corner] = loadObjectPosition(arguments, index, translation);
            triangle.normals[corner] = loadPosition(arguments.objectNormals, index);
        } else {
            triangle.vertices[corner] = loadPosition(arguments.groundPositions, arguments.groundIndices[3 * hit.primitiveIndex + corner]);
            triangle.normals[corner] = float3(0.0, 1.0, 0.0);
        }
    }
    return triangle;
}

float3 getRayOrigin(float3 position, float3 normal, float3 target) {
    return position + normal * (dot(normal, target - position) > 0.0 ? rayOffset : -rayOffset);
}

bool isVisible(Arguments arguments, float3 origin, float3 target) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = target - origin;
    ray.TMin = 0.0;
    ray.TMax = 1.0 - rayOffset;
    RayHit hit;
    return !traceRayInline(arguments.scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, ray, hit);
}

[Differentiable]
float3 intersect(no_diff float3 origin, no_diff float3 direction, float3 a, float3 b, float3 c) {
    const float3 normal = cross(b - a, c - a);
    return origin + direction * (dot(normal, a - origin) / dot(normal, direction));
}

[Differentiable]
float3 interpolateNormal(float3 position, float3 a, float3 b, float3 c, no_diff float3 normals[3]) {
    const float3 normal = cross(b - a, c - a);
    const float scale = 1.0 / dot(normal, normal);
    const float first = dot(cross(b - position, c - position), normal) * scale;
    const float second = dot(cross(c - position, a - position), normal) * scale;
    return normalize(first * normals[0] + second * normals[1] + (1.0 - first - second) * normals[2]);
}

[Differentiable]
float3 shade(float3 albedo, float3 position, float3 normal, no_diff float3 target, no_diff Light light) {
    const float3 toLight = target - position;
    const float distanceSquared = dot(toLight, toLight);
    const float geometry = dot(normal, toLight) * -dot(light.normal, toLight) / (distanceSquared * distanceSquared);
    return albedo / pi * light.radiance * geometry;
}

[Differentiable]
float3 estimate(Parameters parameters, no_diff Triangle triangle, no_diff float3 origin, no_diff float3 direction, no_diff float3 target,
                no_diff Light light) {
    const float3 offset = triangle.object ? parameters.translation - detach(parameters.translation) : float3(0.0);
    const float3 a = triangle.vertices[0] + offset;
    const float3 b = triangle.vertices[1] + offset;
    const float3 c = triangle.vertices[2] + offset;
    const float3 position = intersect(origin, direction, a, b, c);
    const float3 albedo = triangle.object ? parameters.albedoObject : parameters.albedoGround;
    const float3 normal = interpolateNormal(position, a, b, c, triangle.normals);
    return shade(albedo, position, normal, target, light) * light.area;
}

struct Surface {
    float3 position;
    float3 geometricNormal;
    float3 shadingNormal;
    bool facing;

    __init(Triangle triangle, float3 origin, float3 direction) {
        position = intersect(origin, direction, triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]);
        geometricNormal = normalize(cross(triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]));
        shadingNormal = interpolateNormal(position, triangle.vertices[0], triangle.vertices[1], triangle.vertices[2], triangle.normals);
        facing = dot(geometricNormal, direction) < 0.0;
    }

    bool isLit(Arguments arguments, Light light, float3 target) {
        const float3 toLight = target - position;
        return facing && dot(shadingNormal, toLight) > 0.0 && dot(light.normal, toLight) < 0.0
               && isVisible(arguments, getRayOrigin(position, geometricNormal, target), target);
    }
}

struct Shading {
    float3 target;
    bool lit;
}

Shading prepare(Arguments arguments, Triangle triangle, float3 origin, float3 direction, Light light, inout Random random) {
    const Surface surface = Surface(triangle, origin, direction);
    Shading shading;
    shading.target = light.getPoint(float2(random.next(), random.next()));
    shading.lit = surface.isLit(arguments, light, shading.target);
    return shading;
}

float3 getRadiance(Arguments arguments, Parameters parameters, float2 screen, Light light, inout Random random) {
    const float3 origin = arguments.cameraOrigin.xyz;
    const float3 direction = getCameraDirection(arguments, screen);
    const Triangle triangle = trace(arguments, origin, direction, parameters.translation);
    if (!triangle.valid)
        return float3(0.0);
    float3 radiance = float3(0.0);
    for (uint sample = 0; sample < arguments.lightSamples; ++sample) {
        const Shading shading = prepare(arguments, triangle, origin, direction, light, random);
        if (shading.lit)
            radiance += estimate(parameters, triangle, origin, direction, shading.target, light);
    }
    return radiance / float(arguments.lightSamples);
}

float3 loadWeight(Arguments arguments, int2 pixel) {
    const uint index = 3 * (uint(pixel.y) * arguments.width + uint(pixel.x));
    return float3(arguments.weights[index], arguments.weights[index + 1], arguments.weights[index + 2]);
}

bool isInside(Arguments arguments, int2 pixel) {
    return all(pixel >= 0) && pixel.x < int(arguments.width) && pixel.y < int(arguments.height);
}

void addReduced(Arguments arguments, uint offset, float3 value) {
    const float3 sum = WaveActiveSum(value);
    if (!WaveIsFirstLane())
        return;
    for (uint channel = 0; channel < 3; ++channel)
        if (sum[channel] != 0.0)
            atomicAdd(arguments.gradients + offset + channel, sum[channel]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void renderMain(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Parameters parameters = loadParameters(arguments.parameters);
    const Light light = Light(arguments);
    const uint pixel = id.y * arguments.width + id.x;
    float3 sum = float3(0.0);
    for (uint sample = 0; sample < arguments.samples; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        const float2 screen = float2(id.xy) + float2(random.next(), random.next());
        sum += getRadiance(arguments, parameters, screen, light, random);
    }
    const float3 color = sum / float(arguments.samples);
    for (uint channel = 0; channel < 3; ++channel)
        arguments.image[3 * pixel + channel] = color[channel];
}

[Differentiable]
float3 projectToLight(float3 position, float3 edgePoint, no_diff Light light) {
    const float scale = dot(light.normal, light.corner - position) / dot(light.normal, edgePoint - position);
    return position + scale * (edgePoint - position);
}

struct Segment {
    float3 start;
    float3 end;
    float startDepth;
    float endDepth;
    float clipStart;
    float clipEnd;
    uint edge;
}

bool clipToLight(Light light, inout Segment segment) {
    const float2 start = light.getCoordinates(segment.start);
    const float2 delta = light.getCoordinates(segment.end) - start;
    segment.clipStart = 0.0;
    segment.clipEnd = 1.0;
    for (uint axis = 0; axis < 2; ++axis) {
        if (abs(delta[axis]) < 1e-12) {
            if (start[axis] < 0.0 || start[axis] > 1.0)
                return false;
            continue;
        }
        const float first = -start[axis] / delta[axis];
        const float second = (1.0 - start[axis]) / delta[axis];
        segment.clipStart = max(segment.clipStart, min(first, second));
        segment.clipEnd = min(segment.clipEnd, max(first, second));
    }
    return segment.clipEnd > segment.clipStart;
}

bool isSilhouette(Arguments arguments, uint4 edge, float3 viewer, float3 edgeStart) {
    const float first = dot(getFaceNormal(arguments, edge.z), viewer - edgeStart);
    const float second = dot(getFaceNormal(arguments, edge.w), viewer - edgeStart);
    return (first > 0.0) != (second > 0.0);
}

bool projectEdge(Arguments arguments, uint index, float3 position, float3 translation, Light light, out Segment segment) {
    segment = {};
    const uint4 edge = arguments.objectEdges[index];
    const float3 a = loadObjectPosition(arguments, edge.x, translation);
    const float3 b = loadObjectPosition(arguments, edge.y, translation);
    if (!isSilhouette(arguments, edge, position, a))
        return false;
    const float planeDepth = dot(light.normal, light.corner - position);
    segment.startDepth = dot(light.normal, a - position);
    segment.endDepth = dot(light.normal, b - position);
    if (segment.startDepth / planeDepth <= 0.0 || segment.endDepth / planeDepth <= 0.0)
        return false;
    if (abs(segment.startDepth) >= abs(planeDepth) || abs(segment.endDepth) >= abs(planeDepth))
        return false;
    segment.start = projectToLight(position, a, light);
    segment.end = projectToLight(position, b, light);
    segment.edge = index;
    return clipToLight(light, segment);
}

float3 getShadowBoundary(Arguments arguments, Parameters parameters, Triangle triangle, float3 origin, float3 direction, Light light,
                         float3 weight, inout Random random) {
    const Surface surface = Surface(triangle, origin, direction);
    if (!surface.facing)
        return float3(0.0);
    const float3 normal = surface.geometricNormal;
    const float3 position = surface.position + normal * rayOffset;
    Segment chosen = {};
    float total = 0.0;
    for (uint index = 0; index < arguments.edgeCount; ++index) {
        Segment segment;
        if (!projectEdge(arguments, index, position, parameters.translation, light, segment))
            continue;
        const float weight = (segment.clipEnd - segment.clipStart) * length(segment.end - segment.start);
        total += weight;
        if (random.next() * total <= weight)
            chosen = segment;
    }
    if (total <= 0.0)
        return float3(0.0);
    const float along = lerp(chosen.clipStart, chosen.clipEnd, random.next());
    const float3 target = lerp(chosen.start, chosen.end, along);
    const float3 toLight = target - position;
    if (dot(surface.shadingNormal, toLight) <= 0.0 || dot(light.normal, toLight) >= 0.0)
        return float3(0.0);
    const float3 side = normalize(cross(light.normal, chosen.end - chosen.start)) * lightOffset;
    const float visibility = float(isVisible(arguments, position, target - side)) - float(isVisible(arguments, position, target + side));
    if (visibility == 0.0)
        return float3(0.0);
    const uint4 edge = arguments.objectEdges[chosen.edge];
    const float3 a = loadObjectPosition(arguments, edge.x, parameters.translation);
    const float3 b = loadObjectPosition(arguments, edge.y, parameters.translation);
    const float edgeParameter = along * chosen.startDepth / (chosen.endDepth - along * (chosen.endDepth - chosen.startDepth));
    const float3 edgePoint = lerp(a, b, edgeParameter);
    const float3 albedo = triangle.object ? parameters.albedoObject : parameters.albedoGround;
    const float3 integrand = shade(albedo, surface.position, surface.shadingNormal, target, light) * visibility * total;
    const float3 sideNormal = normalize(side);
    const float3 positionVelocity = triangle.object ? direction / dot(normal, direction) : float3(0.0);
    float3 velocity;
    for (uint axis = 0; axis < 3; ++axis) {
        float3 unit = float3(0.0);
        unit[axis] = 1.0;
        const float3 moved = fwd_diff(projectToLight)(diffPair(position, positionVelocity * normal[axis]), diffPair(edgePoint, unit), light).d;
        velocity[axis] = dot(moved, sideNormal);
    }
    return dot(integrand, weight) * velocity;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gradientMain(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.width || id.y >= arguments.height)
        return;
    const Parameters parameters = loadParameters(arguments.parameters);
    const Light light = Light(arguments);
    const uint pixel = id.y * arguments.width + id.x;
    const float3 weight = loadWeight(arguments, int2(id.xy)) / float(arguments.samples);
    const float3 origin = arguments.cameraOrigin.xyz;
    Parameters.Differential sum = Parameters.dzero();
    float3 boundary = float3(0.0);
    for (uint sample = 0; sample < arguments.samples; ++sample) {
        Random random = Random(pixel, sample, arguments.seed);
        const float3 direction = getCameraDirection(arguments, float2(id.xy) + float2(random.next(), random.next()));
        const Triangle triangle = trace(arguments, origin, direction, parameters.translation);
        if (!triangle.valid)
            continue;
        if ((arguments.terms & DirectLighting::shadowBoundaryTerm) != 0)
            boundary += getShadowBoundary(arguments, parameters, triangle, origin, direction, light, weight, random);
        if ((arguments.terms & DirectLighting::interiorTerm) == 0)
            continue;
        const Shading shading = prepare(arguments, triangle, origin, direction, light, random);
        if (!shading.lit)
            continue;
        var pair = diffPair(parameters);
        bwd_diff(estimate)(pair, triangle, origin, direction, shading.target, light, weight);
        sum = Parameters.dadd(sum, pair.d);
    }
    addReduced(arguments, 0, sum.albedoObject);
    addReduced(arguments, 3, sum.albedoGround);
    addReduced(arguments, 6, sum.translation + boundary);
}

float getWeightedRadiance(Arguments arguments, Parameters parameters, float2 screen, Light light, uint seed) {
    const int2 coordinates = int2(floor(screen));
    if (!isInside(arguments, coordinates))
        return 0.0;
    Random random = Random(seed, 0x9e3779b9u, arguments.seed);
    return dot(loadWeight(arguments, coordinates), getRadiance(arguments, parameters, screen, light, random));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void primaryBoundaryMain(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.edgeCount * arguments.edgeSamples)
        return;
    const Parameters parameters = loadParameters(arguments.parameters);
    const Light light = Light(arguments);
    const uint4 edge = arguments.objectEdges[id.x / arguments.edgeSamples];
    const float3 a = loadObjectPosition(arguments, edge.x, parameters.translation);
    const float3 b = loadObjectPosition(arguments, edge.y, parameters.translation);
    if (!isSilhouette(arguments, edge, arguments.cameraOrigin.xyz, a))
        return;
    Random random = Random(id.x, 0x85ebca6bu, arguments.seed);
    const float along = (float(id.x % arguments.edgeSamples) + random.next()) / float(arguments.edgeSamples);
    const float3 edgePoint = lerp(a, b, along);
    if (dot(edgePoint - arguments.cameraOrigin.xyz, arguments.cameraForward.xyz) <= 0.0)
        return;
    const DifferentialPair<float2> projected = fwd_diff(projectToScreen)(arguments, diffPair(edgePoint, b - a));
    const float2 screen = projected.p;
    const float2 tangent = projected.d;
    const float2 normal = normalize(float2(-tangent.y, tangent.x));
    const float innerValue = getWeightedRadiance(arguments, parameters, screen - normal * screenOffset, light, id.x);
    const float outerValue = getWeightedRadiance(arguments, parameters, screen + normal * screenOffset, light, id.x);
    const float scale = length(tangent) / float(arguments.edgeSamples);
    float3 gradient;
    for (uint axis = 0; axis < 3; ++axis) {
        float3 unit = float3(0.0);
        unit[axis] = 1.0;
        gradient[axis] = dot(fwd_diff(projectToScreen)(arguments, diffPair(edgePoint, unit)).d, normal) * scale * (innerValue - outerValue);
    }
    addReduced(arguments, 6, gradient);
}
