#include "MeshRasterizing.h"
#include "Cubemap.slang"

typealias Arguments = MeshRasterizing::Arguments;
typealias Projection = MeshRasterizing::Projection;

static const uint vertexChannels = MeshRasterizing::vertexChannels;
static const uint texelChannels = MeshRasterizing::texelChannels;
static const uint splitSumSize = MeshRasterizing::splitSumSize;
//! The bend of nvdiffrec turns normals that face away from the camera towards the face normal below this cosine.
static const float normalThreshold = 0.1;
//! The standard deviation of the uv jitter of the kd smoothness term.
static const float smoothJitter = 0.005;

float3 loadVector(Ptr<float> values, uint index) {
    return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]);
}

void storeVector(Ptr<float> values, uint index, float3 value) {
    values[index * 3] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

void addVector(AtomicPtr<float> values, uint index, float3 value) {
    atomicAdd(values + index * 3, value.x);
    atomicAdd(values + index * 3 + 1, value.y);
    atomicAdd(values + index * 3 + 2, value.z);
}

[Differentiable]
float4 project(float3 position, no_diff Projection projection) {
    const float3 local = float3(dot(projection.row0.xyz, position) + projection.row0.w, dot(projection.row1.xyz, position) + projection.row1.w,
                                dot(projection.row2.xyz, position) + projection.row2.w);
    return float4(local.xy * projection.lens.xy + projection.lens.zw * local.z, projection.depth.x * local.z + projection.depth.y, local.z);
}

float3 getEdgeFunctions(float4 p0, float4 p1, float4 p2, float2 pixel) {
    const float2 q0 = p0.xy - pixel * p0.w;
    const float2 q1 = p1.xy - pixel * p1.w;
    const float2 q2 = p2.xy - pixel * p2.w;
    return float3(q1.x * q2.y - q1.y * q2.x, q2.x * q0.y - q2.y * q0.x, q0.x * q1.y - q0.y * q1.x);
}

[Differentiable]
float2 getBarycentrics(float4 p0, float4 p1, float4 p2, no_diff float2 pixel) {
    const float2 q0 = p0.xy - pixel * p0.w;
    const float2 q1 = p1.xy - pixel * p1.w;
    const float2 q2 = p2.xy - pixel * p2.w;
    const float a0 = q1.x * q2.y - q1.y * q2.x;
    const float a1 = q2.x * q0.y - q2.y * q0.x;
    const float a2 = q0.x * q1.y - q0.y * q1.x;
    return float2(a0, a1) / (a0 + a1 + a2);
}

[Differentiable]
float3 interpolate(float2 barycentrics, float3 c0, float3 c1, float3 c2) {
    return barycentrics.x * c0 + barycentrics.y * c1 + (1.0 - barycentrics.x - barycentrics.y) * c2;
}

[Differentiable]
float getEdgeCrossing(float4 p1, float4 p2, no_diff float2 center, no_diff float2 halfSize, no_diff bool down) {
    float2 a = p1.xy / p1.w;
    float2 b = p2.xy / p2.w;
    float2 scale = halfSize;
    float2 offset = center;
    if (down) {
        a = a.yx;
        b = b.yx;
        scale = scale.yx;
        offset = offset.yx;
    }
    const float2 v1 = a * scale - offset;
    const float2 v2 = b * scale - offset;
    const float2 delta = v2 - v1;
    return (v1.x * delta.y - v1.y * delta.x) / delta.y;
}

uint getTexelValueCount(Arguments arguments) {
    return arguments.textureWidth * arguments.textureHeight * texelChannels;
}

float2 getTextureSize(Arguments arguments) {
    return float2(arguments.textureWidth, arguments.textureHeight);
}

[Differentiable]
float2 getTexelPosition(float2 barycentrics, no_diff float2 uv0, no_diff float2 uv1, no_diff float2 uv2, no_diff float2 size) {
    return (barycentrics.x * uv0 + barycentrics.y * uv1 + (1.0 - barycentrics.x - barycentrics.y) * uv2) * size - 0.5;
}

struct TexelQuad {
    uint4 texels;
    float2 weight;

    float4 getWeights() {
        return float4((1.0 - weight.x) * (1.0 - weight.y), weight.x * (1.0 - weight.y), (1.0 - weight.x) * weight.y, weight.x * weight.y);
    }
};

TexelQuad findTexelQuad(Arguments arguments, float2 position) {
    const int2 size = int2(arguments.textureWidth, arguments.textureHeight);
    const float2 base = floor(position);
    const int2 low = clamp(int2(base), int2(0), size - 1);
    const int2 high = clamp(int2(base) + 1, int2(0), size - 1);
    TexelQuad quad;
    quad.texels = uint4(low.y * size.x + low.x, low.y * size.x + high.x, high.y * size.x + low.x, high.y * size.x + high.x);
    quad.weight = position - base;
    return quad;
}

float2 loadUv(Arguments arguments, uint corner) {
    return arguments.uvs[arguments.uvTriangles[corner]];
}

uint2 getSize(Arguments arguments) {
    return uint2(arguments.projection.width, arguments.projection.height);
}

uint3 getTriangle(Arguments arguments, uint triangle) {
    return uint3(arguments.triangles[triangle * 3], arguments.triangles[triangle * 3 + 1], arguments.triangles[triangle * 3 + 2]);
}

float2 getPixelCenter(Arguments arguments, uint2 pixel) {
    return (float2(pixel) * 2.0 + 1.0) / float2(getSize(arguments)) - 1.0;
}

void addClip(Arguments arguments, uint vertex, float4 value) {
    const AtomicPtr<float> target = arguments.clipGradient + vertex * 4;
    atomicAdd(target, value.x);
    atomicAdd(target + 1, value.y);
    atomicAdd(target + 3, value.w);
}

bool isSameSign(float a, float b) { return (asint(a) ^ asint(b)) >= 0; }

bool isRationalGreater(float n0, float n1, float d0, float d1) { return (n0 * d1 > n1 * d0) == isSameSign(d0, d1); }

uint getMaximumIndex(float3 numerators, float3 denominators) {
    const bool greater10 = isRationalGreater(numerators.y, numerators.x, denominators.y, denominators.x);
    const bool greater20 = isRationalGreater(numerators.z, numerators.x, denominators.z, denominators.x);
    const bool greater21 = isRationalGreater(numerators.z, numerators.y, denominators.z, denominators.y);
    if (greater20 && greater21)
        return 2;
    return greater10 ? 1 : 0;
}

struct Edge {
    bool found;
    float alpha;
    uint triangle;
    uint index;
    uint pixel0;
    uint pixel1;
    float2 center;

    uint getTarget() { return alpha > 0.0 ? pixel0 : pixel1; }
};

Edge findEdge(Arguments arguments, uint2 pixel, bool down) {
    Edge edge = {};
    edge.pixel0 = pixel.y * arguments.projection.width + pixel.x;
    edge.pixel1 = edge.pixel0 + (down ? arguments.projection.width : 1);
    const float4 raster0 = arguments.raster[edge.pixel0];
    const float4 raster1 = arguments.raster[edge.pixel1];
    if (raster0.w == raster1.w)
        return edge;
    const int triangle0 = int(raster0.w) - 1;
    const int triangle1 = int(raster1.w) - 1;
    int triangle = triangle0 >= 0 ? triangle0 : triangle1;
    if (triangle0 >= 0 && triangle1 >= 0)
        triangle = raster0.z < raster1.z ? triangle0 : triangle1;
    const bool first = triangle == triangle0;
    const uint2 owner = first ? pixel : pixel + (down ? uint2(0, 1) : uint2(1, 0));
    const uint3 vertices = getTriangle(arguments, triangle);
    const int3 opposites = int3(arguments.opposites[triangle * 3], arguments.opposites[triangle * 3 + 1], arguments.opposites[triangle * 3 + 2]);
    const float2 halfSize = float2(getSize(arguments)) * 0.5;
    const float2 center = float2(owner) + 0.5 - halfSize;
    float2 v[3];
    float2 o[3];
    [ForceUnroll]
    for (uint index = 0; index < 3; ++index) {
        const float4 position = arguments.clip[vertices[index]];
        const float4 opposite = opposites[index] < 0 ? position : arguments.clip[opposites[index]];
        if (position.w <= 0.0 || opposite.w <= 0.0)
            return edge;
        v[index] = position.xy / position.w * halfSize - center;
        o[index] = opposite.xy / opposite.w * halfSize - center;
    }
    const float area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[2].x - v[0].x) * (v[1].y - v[0].y);
    const float3 wings = float3((v[1].x - o[0].x) * (v[2].y - o[0].y) - (v[2].x - o[0].x) * (v[1].y - o[0].y),
                                (v[2].x - o[1].x) * (v[0].y - o[1].y) - (v[0].x - o[1].x) * (v[2].y - o[1].y),
                                (v[0].x - o[2].x) * (v[1].y - o[2].y) - (v[1].x - o[2].x) * (v[0].y - o[2].y));
    const bool3 silhouette = bool3(isSameSign(wings.x, area), isSameSign(wings.y, area), isSameSign(wings.z, area));
    if (!any(silhouette))
        return edge;
    if (down) {
        [ForceUnroll]
        for (uint index = 0; index < 3; ++index)
            v[index] = v[index].yx;
    }
    float3 dx = float3(v[2].x - v[1].x, v[0].x - v[2].x, v[1].x - v[0].x);
    float3 dy = float3(v[2].y - v[1].y, v[0].y - v[2].y, v[1].y - v[0].y);
    const float side = first ? 1.0 : -1.0;
    float3 distances = side * float3(v[1].x * dy.x - v[1].y * dx.x, v[2].x * dy.y - v[2].y * dx.y, v[0].x * dy.z - v[0].y * dx.z);
    [ForceUnroll]
    for (uint index = 0; index < 3; ++index) {
        if (!isSameSign(v[(index + 1) % 3].y, v[(index + 2) % 3].y))
            continue;
        distances[index] = -3.402823466e+38;
        dy[index] = 1.0;
    }
    const uint index = getMaximumIndex(distances, dy);
    if (!silhouette[index] || abs(dy[index]) < abs(dx[index]))
        return edge;
    const float crossing = distances[index] / dy[index];
    const float tolerance = 0.0625;
    if (crossing <= -tolerance || crossing >= 1.0 + tolerance)
        return edge;
    edge.found = true;
    edge.alpha = side * (0.5 - clamp(crossing, 0.0, 1.0));
    edge.triangle = triangle;
    edge.index = index;
    edge.center = center;
    return edge;
}

bool isInside(Arguments arguments, uint2 pixel) {
    return all(pixel < getSize(arguments));
}

bool hasNext(Arguments arguments, uint2 pixel, bool down) {
    return (down ? pixel.y + 1 : pixel.x + 1) < (down ? arguments.projection.height : arguments.projection.width);
}

bool hasPrevious(uint2 pixel, bool down) {
    return (down ? pixel.y : pixel.x) > 0;
}

uint2 getPrevious(uint2 pixel, bool down) {
    return pixel - (down ? uint2(0, 1) : uint2(1, 0));
}

float3 loadMaterial(Ptr<float> values, uint vertex, uint offset) {
    const uint base = vertex * vertexChannels + offset;
    return float3(values[base], values[base + 1], values[base + 2]);
}

void addMaterial(AtomicPtr<float> values, uint vertex, uint offset, float3 value) {
    const uint base = vertex * vertexChannels + offset;
    atomicAdd(values + base, value.x);
    atomicAdd(values + base + 1, value.y);
    atomicAdd(values + base + 2, value.z);
}

float3 loadTexel(Arguments arguments, uint texel, uint group) {
    const uint base = texel * texelChannels + group * 3;
    return float3(arguments.texels[base], arguments.texels[base + 1], arguments.texels[base + 2]);
}

[Differentiable]
float3 safeNormalize(float3 value) {
    const float size = length(value);
    return size > 0.0 ? value / size : float3(0.0);
}

[Differentiable]
float3 getAreaNormal(float3 p0, float3 p1, float3 p2) {
    return cross(p1 - p0, p2 - p0);
}

[Differentiable]
float toSrgb(float value) {
    return value <= 0.0031308 ? value * 12.92 : pow(max(value, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055;
}

[Differentiable]
float3 toSrgb(float3 value) {
    return float3(toSrgb(value.x), toSrgb(value.y), toSrgb(value.z));
}

//! The derivatives of a bilinear blend along the two fractions.
void getBilinearSlopes(float2 fraction, float3 v00, float3 v10, float3 v01, float3 v11, out float3 alongX, out float3 alongY) {
    alongX = (1.0 - fraction.y) * (v10 - v00) + fraction.y * (v11 - v01);
    alongY = (1.0 - fraction.x) * (v01 - v00) + fraction.x * (v11 - v10);
}

float3 blendTaps(float2 fraction, float3 v00, float3 v10, float3 v01, float3 v11) {
    const float4 weights = getTapWeights(fraction);
    return weights.x * v00 + weights.y * v10 + weights.z * v01 + weights.w * v11;
}

[Differentiable]
[BackwardDerivative(sampleTextureBackward)]
float3 sampleTexture(no_diff Arguments arguments, no_diff uint group, float2 position) {
    const TexelQuad quad = findTexelQuad(arguments, position);
    return no_diff blendTaps(quad.weight, loadTexel(arguments, quad.texels.x, group), loadTexel(arguments, quad.texels.y, group),
                     loadTexel(arguments, quad.texels.z, group), loadTexel(arguments, quad.texels.w, group));
}

void scatterTexture(Arguments arguments, uint group, float2 position, float3 gradient) {
    const TexelQuad quad = findTexelQuad(arguments, position);
    const float4 weights = quad.getWeights();
    [ForceUnroll]
    for (uint index = 0; index < 4; ++index) {
        const AtomicPtr<float> target = arguments.texelAccumulator + quad.texels[index] * texelChannels + group * 3;
        atomicAdd(target, weights[index] * gradient.x);
        atomicAdd(target + 1, weights[index] * gradient.y);
        atomicAdd(target + 2, weights[index] * gradient.z);
    }
}

void sampleTextureBackward(Arguments arguments, uint group, inout DifferentialPair<float2> position, float3 gradient) {
    const TexelQuad quad = findTexelQuad(arguments, position.p);
    scatterTexture(arguments, group, position.p, gradient);
    float3 alongX;
    float3 alongY;
    getBilinearSlopes(quad.weight, loadTexel(arguments, quad.texels.x, group), loadTexel(arguments, quad.texels.y, group),
                      loadTexel(arguments, quad.texels.z, group), loadTexel(arguments, quad.texels.w, group), alongX, alongY);
    position = diffPair(position.p, float2(dot(gradient, alongX), dot(gradient, alongY)));
}

float3 loadSpecular(Light light, uint index) {
    return light.specular[index].xyz;
}

float3 loadDiffuse(Light light, uint index) {
    return float3(light.diffuse[index * 3], light.diffuse[index * 3 + 1], light.diffuse[index * 3 + 2]);
}

void addLightTexel(AtomicPtr<float> values, uint index, float3 value) {
    atomicAdd(values + index * 3, value.x);
    atomicAdd(values + index * 3 + 1, value.y);
    atomicAdd(values + index * 3 + 2, value.z);
}

//! The direction gradient of a bilinear cube lookup, through the face coordinates of the face that the lookup reads.
float3 getCubeDirectionGradient(float3 direction, CubeTaps taps, uint size, float3 v00, float3 v10, float3 v01, float3 v11, float3 gradient) {
    float3 alongX;
    float3 alongY;
    getBilinearSlopes(taps.fraction, v00, v10, v01, v11, alongX, alongY);
    var pair = diffPair(direction);
    bwd_diff(getFaceCoordinate)(pair, taps.face, size, float2(dot(gradient, alongX), dot(gradient, alongY)));
    return pair.d;
}

[Differentiable]
[BackwardDerivative(sampleSpecularLevelBackward)]
float3 sampleSpecularLevel(no_diff Light light, no_diff uint level, float3 direction) {
    const uint size = getLevelSize(light, level);
    const uint offset = getLevelOffset(light, level);
    const CubeTaps taps = getCubeTaps(direction, size);
    return no_diff blendTaps(taps.fraction, loadSpecular(light, offset + taps.texels.x), loadSpecular(light, offset + taps.texels.y),
                     loadSpecular(light, offset + taps.texels.z), loadSpecular(light, offset + taps.texels.w));
}

void sampleSpecularLevelBackward(Light light, uint level, inout DifferentialPair<float3> direction, float3 gradient) {
    const uint size = getLevelSize(light, level);
    const uint offset = getLevelOffset(light, level);
    const CubeTaps taps = getCubeTaps(direction.p, size);
    const float4 weights = getTapWeights(taps.fraction);
    [ForceUnroll]
    for (uint index = 0; index < 4; ++index)
        addLightTexel(light.specularGradient, offset + taps.texels[index], weights[index] * gradient);
    direction = diffPair(direction.p, getCubeDirectionGradient(direction.p, taps, size, loadSpecular(light, offset + taps.texels.x),
                                                               loadSpecular(light, offset + taps.texels.y), loadSpecular(light, offset + taps.texels.z),
                                                               loadSpecular(light, offset + taps.texels.w), gradient));
}

[Differentiable]
[BackwardDerivative(sampleDiffuseBackward)]
float3 sampleDiffuse(no_diff Light light, float3 direction) {
    const uint size = getLevelSize(light, EnvironmentLight::levelCount - 1);
    const CubeTaps taps = getCubeTaps(direction, size);
    return no_diff blendTaps(taps.fraction, loadDiffuse(light, taps.texels.x), loadDiffuse(light, taps.texels.y), loadDiffuse(light, taps.texels.z),
                     loadDiffuse(light, taps.texels.w));
}

void sampleDiffuseBackward(Light light, inout DifferentialPair<float3> direction, float3 gradient) {
    const uint size = getLevelSize(light, EnvironmentLight::levelCount - 1);
    const CubeTaps taps = getCubeTaps(direction.p, size);
    const float4 weights = getTapWeights(taps.fraction);
    [ForceUnroll]
    for (uint index = 0; index < 4; ++index)
        addLightTexel(light.diffuseGradient, taps.texels[index], weights[index] * gradient);
    direction = diffPair(direction.p, getCubeDirectionGradient(direction.p, taps, size, loadDiffuse(light, taps.texels.x), loadDiffuse(light, taps.texels.y),
                                                               loadDiffuse(light, taps.texels.z), loadDiffuse(light, taps.texels.w), gradient));
}

//! The mip level of a roughness, as get_mip of nvdiffrec: linear up to the maximum roughness, then one more level up to 1.
[Differentiable]
float getMip(float roughness) {
    const float low = EnvironmentLight::minimumRoughness;
    const float high = EnvironmentLight::maximumRoughness;
    const float levels = float(EnvironmentLight::levelCount - 2);
    if (roughness < high)
        return (clamp(roughness, low, high) - low) / (high - low) * levels;
    return (clamp(roughness, high, 1.0) - high) / (1.0 - high) + levels;
}

[Differentiable]
float3 sampleSpecular(no_diff Light light, float3 direction, float mip) {
    const float level = clamp(mip, 0.0, float(EnvironmentLight::levelCount - 1));
    const uint low = min(uint(detach(level)), EnvironmentLight::levelCount - 2);
    const float blend = level - float(low);
    const float3 lower = sampleSpecularLevel(light, low, direction);
    return lower + (sampleSpecularLevel(light, low + 1, direction) - lower) * blend;
}

float2 loadSplitSum(Ptr<float2> table, int2 texel) {
    const int2 clamped = clamp(texel, int2(0), int2(int(splitSumSize) - 1));
    return table[uint(clamped.y) * splitSumSize + uint(clamped.x)];
}

//! A bilinear lookup with clamped taps, as dr.texture with boundary_mode clamp.
[Differentiable]
float2 sampleSplitSum(no_diff Ptr<float2> table, float2 uv) {
    const float2 position = uv * float(splitSumSize) - 0.5;
    const int2 base = int2(floor(detach(position)));
    const float2 fraction = position - float2(base);
    const float2 v00 = loadSplitSum(table, base);
    const float2 v10 = loadSplitSum(table, base + int2(1, 0));
    const float2 v01 = loadSplitSum(table, base + int2(0, 1));
    const float2 v11 = loadSplitSum(table, base + int2(1, 1));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

struct ShadingFrame {
    float3 view;
    float3 geometric;
    float3 tangent;
    bool perturb;
}

//! prepare_shading_normal of nvdiffrec: the normal map in the smooth tangent frame (OpenGL), two sided, bent towards the face normal at grazing views.
[Differentiable]
float3 getShadingNormal(float3 smoothNormal, float3 perturbed, no_diff ShadingFrame frame) {
    const float3 normal = safeNormalize(smoothNormal);
    float3 shading = normal;
    if (frame.perturb) {
        const float3 bitangent = safeNormalize(cross(frame.tangent, normal));
        shading = safeNormalize(frame.tangent * perturbed.x - bitangent * perturbed.y + normal * max(perturbed.z, 0.0));
    }
    float3 geometric = frame.geometric;
    if (dot(frame.view, geometric) < 0.0) {
        shading = -shading;
        geometric = -geometric;
    }
    const float bend = clamp(dot(frame.view, shading) / normalThreshold, 0.0, 1.0);
    return geometric * (1.0 - bend) + shading * bend;
}

//! EnvironmentLight.shade of nvdiffrec: the diffuse irradiance and the split-sum specular term, darkened by the occlusion in ks.x; then the sRGB transfer.
[Differentiable]
float3 shadeSurface(float3 kd, float3 ks, float3 perturbed, float3 smoothNormal, no_diff ShadingFrame frame, no_diff Arguments arguments) {
    const float3 normal = getShadingNormal(smoothNormal, perturbed, frame);
    const float roughness = ks.y;
    const float metallic = ks.z;
    const float3 specularColour = (1.0 - metallic) * 0.04 + kd * metallic;
    const float3 diffuseColour = kd * (1.0 - metallic);
    float3 colour = sampleDiffuse(arguments.light, normal) * diffuseColour;
    const float cosine = max(dot(frame.view, normal), 1e-4);
    const float2 splitSum = sampleSplitSum(arguments.splitSum, float2(cosine, roughness));
    const float3 reflected = safeNormalize(2.0 * dot(frame.view, normal) * normal - frame.view);
    colour += sampleSpecular(arguments.light, reflected, getMip(roughness)) * (specularColour * splitSum.x + splitSum.y);
    return toSrgb(colour * (1.0 - ks.x));
}

[Differentiable]
float3 shadeVertexSurface(float2 barycentrics, float3 n0, float3 n1, float3 n2, float3 kd0, float3 kd1, float3 kd2, float3 ks0, float3 ks1, float3 ks2,
                          no_diff ShadingFrame frame, no_diff Arguments arguments) {
    return shadeSurface(interpolate(barycentrics, kd0, kd1, kd2), interpolate(barycentrics, ks0, ks1, ks2), float3(0.0, 0.0, 1.0),
                        interpolate(barycentrics, n0, n1, n2), frame, arguments);
}

[Differentiable]
float3 shadeTexturedSurface(float2 barycentrics, float3 n0, float3 n1, float3 n2, no_diff float2 uv0, no_diff float2 uv1, no_diff float2 uv2,
                            no_diff ShadingFrame frame, no_diff Arguments arguments) {
    const float2 position = getTexelPosition(barycentrics, uv0, uv1, uv2, getTextureSize(arguments));
    return shadeSurface(sampleTexture(arguments, 0, position), sampleTexture(arguments, 1, position), sampleTexture(arguments, 2, position),
                        interpolate(barycentrics, n0, n1, n2), frame, arguments);
}

ShadingFrame getFrame(Arguments arguments, uint3 vertices, float2 barycentrics) {
    const float3 p0 = loadVector(arguments.positions, vertices.x);
    const float3 p1 = loadVector(arguments.positions, vertices.y);
    const float3 p2 = loadVector(arguments.positions, vertices.z);
    ShadingFrame frame;
    frame.view = safeNormalize(arguments.cameraPosition.xyz - interpolate(barycentrics, p0, p1, p2));
    frame.geometric = safeNormalize(getAreaNormal(p0, p1, p2));
    frame.perturb = arguments.textureWidth > 0;
    frame.tangent = frame.perturb ? safeNormalize(interpolate(barycentrics, loadVector(arguments.tangents, vertices.x), loadVector(arguments.tangents, vertices.y),
                                                              loadVector(arguments.tangents, vertices.z)))
                                  : float3(0.0);
    return frame;
}

uint hashValue(uint value) {
    const uint state = value * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float2 getGaussianPair(uint pixel, uint seed) {
    const uint first = hashValue(pixel ^ hashValue(seed + 0x9e3779b9u));
    const uint second = hashValue(first);
    const float u = max(float(first >> 8) / 16777216.0, 1e-7);
    const float v = float(second >> 8) / 16777216.0;
    return sqrt(-2.0 * log(u)) * float2(cos(2.0 * pi * v), sin(2.0 * pi * v));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clear(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x < arguments.vertexCount * 4)
        atomicStore(arguments.clipGradient + id.x, 0.0);
    if (id.x < arguments.vertexCount * vertexChannels)
        atomicStore(arguments.vertexMaterialAccumulator + id.x, 0.0);
    if (id.x < arguments.vertexCount * 3) {
        atomicStore(arguments.normalSums + id.x, 0.0);
        atomicStore(arguments.tangentSums + id.x, 0.0);
        atomicStore(arguments.normalGradient + id.x, 0.0);
        atomicStore(arguments.shadingPositionGradient + id.x, 0.0);
    }
    if (id.x < getTexelValueCount(arguments))
        atomicStore(arguments.texelAccumulator + id.x, 0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void projectVertices(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.vertexCount)
        return;
    arguments.clip[id.x] = project(loadVector(arguments.positions, id.x), arguments.projection);
}

//! auto_normals and compute_tangents of nvdiffrec: area weighted face normals, and the uv tangents of the faces.
[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateNormals(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.triangleCount)
        return;
    const uint3 vertices = getTriangle(arguments, id.x);
    const float3 p0 = loadVector(arguments.positions, vertices.x);
    const float3 p1 = loadVector(arguments.positions, vertices.y);
    const float3 p2 = loadVector(arguments.positions, vertices.z);
    const float3 normal = getAreaNormal(p0, p1, p2);
    float3 tangent = float3(0.0);
    if (arguments.textureWidth > 0) {
        const float2 uv0 = loadUv(arguments, id.x * 3);
        const float2 e1 = loadUv(arguments, id.x * 3 + 1) - uv0;
        const float2 e2 = loadUv(arguments, id.x * 3 + 2) - uv0;
        const float denominator = e1.x * e2.y - e1.y * e2.x;
        tangent = ((p1 - p0) * e2.y - (p2 - p0) * e1.y) / (denominator > 0.0 ? max(denominator, 1e-6) : min(denominator, -1e-6));
    }
    [ForceUnroll]
    for (uint corner = 0; corner < 3; ++corner) {
        addVector(arguments.normalSums, vertices[corner], normal);
        if (arguments.textureWidth > 0)
            addVector(arguments.tangentSums, vertices[corner], tangent);
    }
}

float3 loadAtomicVector(AtomicPtr<float> values, uint index) {
    return float3(atomicLoad(values + index * 3), atomicLoad(values + index * 3 + 1), atomicLoad(values + index * 3 + 2));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void finishNormals(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.vertexCount)
        return;
    const float3 normal = safeNormalize(loadAtomicVector(arguments.normalSums, id.x));
    storeVector(arguments.normals, id.x, normal);
    if (arguments.textureWidth == 0)
        return;
    const float3 tangent = safeNormalize(loadAtomicVector(arguments.tangentSums, id.x));
    storeVector(arguments.tangents, id.x, safeNormalize(tangent - dot(tangent, normal) * normal));
}

struct RasterVaryings {
    float4 position : SV_Position;
    nointerpolation uint triangle : TRIANGLE;
};

[shader("vertex")]
RasterVaryings rasterVertex(uint index : SV_VertexID) {
    const Arguments arguments = rootArguments<Arguments>();
    const float4 clip = arguments.clip[arguments.triangles[index]];
    RasterVaryings output;
    output.position = float4(clip.x, -clip.y, clip.z, clip.w);
    output.triangle = index / 3;
    return output;
}

[shader("fragment")]
float4 rasterFragment(RasterVaryings input) : SV_Target0 {
    return float4(float(input.triangle + 1), 0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint pixel = id.y * arguments.projection.width + id.x;
    const float triangleId = arguments.ids.Load(int2(id.xy)).x;
    float4 result = float4(0.0);
    if (triangleId > 0.0) {
        const uint3 vertices = getTriangle(arguments, uint(triangleId) - 1);
        const float4 p0 = arguments.clip[vertices.x];
        const float4 p1 = arguments.clip[vertices.y];
        const float4 p2 = arguments.clip[vertices.z];
        const float3 edges = getEdgeFunctions(p0, p1, p2, getPixelCenter(arguments, id.xy));
        const float area = edges.x + edges.y + edges.z;
        const float z = dot(float3(p0.z, p1.z, p2.z), edges) / dot(float3(p0.w, p1.w, p2.w), edges);
        float2 barycentrics = area == 0.0 ? float2(1.0 / 3.0) : saturate(edges.xy / area);
        barycentrics /= max(barycentrics.x + barycentrics.y, 1.0);
        result = float4(barycentrics, z, triangleId);
    }
    arguments.raster[pixel] = result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shadePixels(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint pixel = id.y * arguments.projection.width + id.x;
    const float4 raster = arguments.raster[pixel];
    float3 color = arguments.background.xyz;
    if (raster.w > 0.0) {
        const uint triangle = uint(raster.w) - 1;
        const uint3 vertices = getTriangle(arguments, triangle);
        const ShadingFrame frame = getFrame(arguments, vertices, raster.xy);
        const float3 n0 = loadVector(arguments.normals, vertices.x);
        const float3 n1 = loadVector(arguments.normals, vertices.y);
        const float3 n2 = loadVector(arguments.normals, vertices.z);
        if (arguments.textureWidth > 0)
            color = shadeTexturedSurface(raster.xy, n0, n1, n2, loadUv(arguments, triangle * 3), loadUv(arguments, triangle * 3 + 1),
                                         loadUv(arguments, triangle * 3 + 2), frame, arguments);
        else
            color = shadeVertexSurface(raster.xy, n0, n1, n2, loadMaterial(arguments.vertexMaterials, vertices.x, 0),
                                       loadMaterial(arguments.vertexMaterials, vertices.y, 0), loadMaterial(arguments.vertexMaterials, vertices.z, 0),
                                       loadMaterial(arguments.vertexMaterials, vertices.x, 3), loadMaterial(arguments.vertexMaterials, vertices.y, 3),
                                       loadMaterial(arguments.vertexMaterials, vertices.z, 3), frame, arguments);
    }
    storeVector(arguments.shaded, pixel, color);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void antialias(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint pixel = id.y * arguments.projection.width + id.x;
    float3 color = loadVector(arguments.shaded, pixel);
    if (arguments.antialiasing != 0) {
        [ForceUnroll]
        for (uint direction = 0; direction < 2; ++direction) {
            const bool down = direction == 1;
            if (hasNext(arguments, id.xy, down)) {
                const Edge edge = findEdge(arguments, id.xy, down);
                if (edge.found && edge.getTarget() == pixel)
                    color += edge.alpha * (loadVector(arguments.shaded, edge.pixel1) - loadVector(arguments.shaded, edge.pixel0));
            }
            if (hasPrevious(id.xy, down)) {
                const Edge edge = findEdge(arguments, getPrevious(id.xy, down), down);
                if (edge.found && edge.getTarget() == pixel)
                    color += edge.alpha * (loadVector(arguments.shaded, edge.pixel1) - loadVector(arguments.shaded, edge.pixel0));
            }
        }
    }
    storeVector(arguments.image, pixel, color);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void antialiasBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint pixel = id.y * arguments.projection.width + id.x;
    float3 gradient = loadVector(arguments.imageGradient, pixel);
    if (arguments.antialiasing != 0) {
        const float2 halfSize = float2(getSize(arguments)) * 0.5;
        [ForceUnroll]
        for (uint direction = 0; direction < 2; ++direction) {
            const bool down = direction == 1;
            if (hasPrevious(id.xy, down)) {
                const Edge edge = findEdge(arguments, getPrevious(id.xy, down), down);
                if (edge.found)
                    gradient += edge.alpha * loadVector(arguments.imageGradient, edge.getTarget());
            }
            if (!hasNext(arguments, id.xy, down))
                continue;
            const Edge edge = findEdge(arguments, id.xy, down);
            if (!edge.found)
                continue;
            const float3 targetGradient = loadVector(arguments.imageGradient, edge.getTarget());
            gradient -= edge.alpha * targetGradient;
            const float weight = dot(targetGradient, loadVector(arguments.shaded, edge.pixel1) - loadVector(arguments.shaded, edge.pixel0));
            if (weight == 0.0 || abs(edge.alpha) >= 0.5)
                continue;
            const uint3 vertices = getTriangle(arguments, edge.triangle);
            const uint vertex1 = vertices[(edge.index + 1) % 3];
            const uint vertex2 = vertices[(edge.index + 2) % 3];
            var p1 = diffPair(arguments.clip[vertex1]);
            var p2 = diffPair(arguments.clip[vertex2]);
            bwd_diff(getEdgeCrossing)(p1, p2, edge.center, halfSize, down, -weight);
            addClip(arguments, vertex1, p1.d);
            addClip(arguments, vertex2, p2.d);
        }
    }
    storeVector(arguments.shadedGradient, pixel, gradient);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shadeBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (!isInside(arguments, id.xy))
        return;
    const uint pixel = id.y * arguments.projection.width + id.x;
    const float4 raster = arguments.raster[pixel];
    if (raster.w <= 0.0)
        return;
    const uint triangle = uint(raster.w) - 1;
    const uint3 vertices = getTriangle(arguments, triangle);
    const float3 gradient = loadVector(arguments.shadedGradient, pixel);
    const ShadingFrame frame = getFrame(arguments, vertices, raster.xy);
    var barycentrics = diffPair(raster.xy);
    var n0 = diffPair(loadVector(arguments.normals, vertices.x));
    var n1 = diffPair(loadVector(arguments.normals, vertices.y));
    var n2 = diffPair(loadVector(arguments.normals, vertices.z));
    if (arguments.textureWidth > 0) {
        const float2 uv0 = loadUv(arguments, triangle * 3);
        const float2 uv1 = loadUv(arguments, triangle * 3 + 1);
        const float2 uv2 = loadUv(arguments, triangle * 3 + 2);
        bwd_diff(shadeTexturedSurface)(barycentrics, n0, n1, n2, uv0, uv1, uv2, frame, arguments, gradient);
        const float2 position = getTexelPosition(raster.xy, uv0, uv1, uv2, getTextureSize(arguments));
        if (arguments.smoothWeight > 0.0) {
            const float2 jittered = position + getGaussianPair(pixel, arguments.seed) * smoothJitter * getTextureSize(arguments);
            const float3 difference = sampleTexture(arguments, 0, jittered) - sampleTexture(arguments, 0, position);
            const float3 smoothGradient = sign(difference) * arguments.smoothWeight / 3.0;
            scatterTexture(arguments, 0, jittered, smoothGradient);
            scatterTexture(arguments, 0, position, -smoothGradient);
        }
        scatterTexture(arguments, 1, position, float3(arguments.occlusionWeight, 0.0, 0.0));
    } else {
        var kd0 = diffPair(loadMaterial(arguments.vertexMaterials, vertices.x, 0));
        var kd1 = diffPair(loadMaterial(arguments.vertexMaterials, vertices.y, 0));
        var kd2 = diffPair(loadMaterial(arguments.vertexMaterials, vertices.z, 0));
        var ks0 = diffPair(loadMaterial(arguments.vertexMaterials, vertices.x, 3));
        var ks1 = diffPair(loadMaterial(arguments.vertexMaterials, vertices.y, 3));
        var ks2 = diffPair(loadMaterial(arguments.vertexMaterials, vertices.z, 3));
        bwd_diff(shadeVertexSurface)(barycentrics, n0, n1, n2, kd0, kd1, kd2, ks0, ks1, ks2, frame, arguments, gradient);
        const float3 weights = float3(raster.xy, 1.0 - raster.x - raster.y);
        const float3 occlusion = float3(arguments.occlusionWeight, 0.0, 0.0);
        addMaterial(arguments.vertexMaterialAccumulator, vertices.x, 0, kd0.d);
        addMaterial(arguments.vertexMaterialAccumulator, vertices.y, 0, kd1.d);
        addMaterial(arguments.vertexMaterialAccumulator, vertices.z, 0, kd2.d);
        addMaterial(arguments.vertexMaterialAccumulator, vertices.x, 3, ks0.d + weights.x * occlusion);
        addMaterial(arguments.vertexMaterialAccumulator, vertices.y, 3, ks1.d + weights.y * occlusion);
        addMaterial(arguments.vertexMaterialAccumulator, vertices.z, 3, ks2.d + weights.z * occlusion);
    }
    addVector(arguments.normalGradient, vertices.x, n0.d);
    addVector(arguments.normalGradient, vertices.y, n1.d);
    addVector(arguments.normalGradient, vertices.z, n2.d);
    var p0 = diffPair(arguments.clip[vertices.x]);
    var p1 = diffPair(arguments.clip[vertices.y]);
    var p2 = diffPair(arguments.clip[vertices.z]);
    bwd_diff(getBarycentrics)(p0, p1, p2, getPixelCenter(arguments, id.xy), barycentrics.d);
    addClip(arguments, vertices.x, p0.d);
    addClip(arguments, vertices.y, p1.d);
    addClip(arguments, vertices.z, p2.d);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void normalBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.vertexCount)
        return;
    var sum = diffPair(loadAtomicVector(arguments.normalSums, id.x));
    bwd_diff(safeNormalize)(sum, loadAtomicVector(arguments.normalGradient, id.x));
    storeVector(arguments.normalSumGradient, id.x, sum.d);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void faceNormalBackward(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x >= arguments.triangleCount)
        return;
    const uint3 vertices = getTriangle(arguments, id.x);
    const float3 gradient = loadVector(arguments.normalSumGradient, vertices.x) + loadVector(arguments.normalSumGradient, vertices.y)
                            + loadVector(arguments.normalSumGradient, vertices.z);
    var p0 = diffPair(loadVector(arguments.positions, vertices.x));
    var p1 = diffPair(loadVector(arguments.positions, vertices.y));
    var p2 = diffPair(loadVector(arguments.positions, vertices.z));
    bwd_diff(getAreaNormal)(p0, p1, p2, gradient);
    addVector(arguments.shadingPositionGradient, vertices.x, p0.d);
    addVector(arguments.shadingPositionGradient, vertices.y, p1.d);
    addVector(arguments.shadingPositionGradient, vertices.z, p2.d);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gatherGradients(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x < arguments.vertexCount * vertexChannels)
        arguments.vertexMaterialGradient[id.x] = atomicLoad(arguments.vertexMaterialAccumulator + id.x);
    if (id.x >= arguments.vertexCount)
        return;
    const AtomicPtr<float> clip = arguments.clipGradient + id.x * 4;
    const float4 clipGradient = float4(atomicLoad(clip), atomicLoad(clip + 1), atomicLoad(clip + 2), atomicLoad(clip + 3));
    var position = diffPair(loadVector(arguments.positions, id.x));
    bwd_diff(project)(position, arguments.projection, clipGradient);
    storeVector(arguments.positionGradient, id.x, position.d + loadAtomicVector(arguments.shadingPositionGradient, id.x));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gatherTexelGradients(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    if (id.x < getTexelValueCount(arguments))
        arguments.texelGradient[id.x] = atomicLoad(arguments.texelAccumulator + id.x);
}

//! The clamps of nvdiffrec after every step: kd in [0, 1], ks in [(0, minimum roughness, 0), 1], the normal map in [(-1, -1, 0), 1].
[shader("compute")]
[numthreads(256, 1, 1)]
void clampMaterials(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const bool textured = arguments.textureWidth > 0;
    const uint count = textured ? arguments.textureWidth * arguments.textureHeight : arguments.vertexCount;
    if (id.x >= count)
        return;
    const Ptr<float> values = textured ? arguments.texels + id.x * texelChannels : arguments.vertexMaterials + id.x * vertexChannels;
    const float lower[texelChannels] = { 0.0, 0.0, 0.0, 0.0, arguments.minimumRoughness, 0.0, -1.0, -1.0, 0.0 };
    [ForceUnroll]
    for (uint channel = 0; channel < texelChannels; ++channel)
        if (channel < (textured ? texelChannels : vertexChannels))
            values[channel] = clamp(values[channel], lower[channel], 1.0);
}
