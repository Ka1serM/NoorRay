#include "Cubemap.slang"

typealias Arguments = EnvironmentLight::Arguments;

float3 loadTexel(Ptr<float> values, uint index) {
    return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]);
}

float3 loadAtomicTexel(AtomicPtr<float> values, uint index) {
    return float3(atomicLoad(values + index * 3), atomicLoad(values + index * 3 + 1), atomicLoad(values + index * 3 + 2));
}

void storeTexel(Ptr<float> values, uint index, float3 value) {
    values[index * 3] = value.x;
    values[index * 3 + 1] = value.y;
    values[index * 3 + 2] = value.z;
}

struct LevelTexel {
    uint face;
    uint2 texel;
}

LevelTexel getLevelTexel(uint index, uint size) {
    LevelTexel result;
    result.face = index / (size * size);
    result.texel = uint2(index % size, (index / size) % size);
    return result;
}

uint getLastLevel() {
    return EnvironmentLight::levelCount - 1;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void downsample(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint size = getLevelSize(light, light.level);
    if (id.x >= getLevelTexelCount(light, light.level))
        return;
    const LevelTexel target = getLevelTexel(id.x, size);
    const uint sourceSize = size * 2;
    const uint sourceOffset = getLevelOffset(light, light.level - 1);
    float3 sum = float3(0.0);
    [ForceUnroll]
    for (uint corner = 0; corner < 4; ++corner)
        sum += loadTexel(light.levels, sourceOffset + getTexelIndex(target.face, target.texel * 2 + uint2(corner & 1, corner >> 1), sourceSize));
    storeTexel(light.levels, getLevelOffset(light, light.level) + id.x, sum * 0.25);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void filterSpecular(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint size = getLevelSize(light, light.level);
    if (id.x >= getLevelTexelCount(light, light.level))
        return;
    const LevelTexel target = getLevelTexel(id.x, size);
    const float3 axis = getTexelDirection(target.face, target.texel, size);
    const float roughness = getLevelRoughness(light.level);
    const float alphaSquared = roughness * roughness * roughness * roughness;
    const float cutoff = getLevelCutoff(light, light.level);
    const uint offset = getLevelOffset(light, light.level);
    float3 sum = float3(0.0);
    float weights = 0.0;
    for (uint face = 0; face < cubeFaces; ++face) {
        const int4 bounds = getConeBounds(axis, cutoff, face, size);
        for (int y = bounds.z; y <= bounds.w; ++y)
            for (int x = bounds.x; x <= bounds.y; ++x) {
                const float3 source = getTexelDirection(face, uint2(x, y), size);
                if (dot(source, axis) < cutoff)
                    continue;
                const float weight = getSpecularWeight(axis, source, alphaSquared, getTexelArea(uint2(x, y), size));
                sum += weight * loadTexel(light.levels, offset + getTexelIndex(face, uint2(x, y), size));
                weights += weight;
            }
    }
    light.specular[offset + id.x] = float4(weights > 0.0 ? sum / weights : float3(0.0), weights);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void filterDiffuse(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint last = getLastLevel();
    const uint size = getLevelSize(light, last);
    if (id.x >= getLevelTexelCount(light, last))
        return;
    const LevelTexel target = getLevelTexel(id.x, size);
    const float3 normal = getTexelDirection(target.face, target.texel, size);
    const uint offset = getLevelOffset(light, last);
    float3 sum = float3(0.0);
    for (uint face = 0; face < cubeFaces; ++face)
        for (uint y = 0; y < size; ++y)
            for (uint x = 0; x < size; ++x)
                sum += getDiffuseWeight(normal, getTexelDirection(face, uint2(x, y), size), getTexelArea(uint2(x, y), size))
                       * loadTexel(light.levels, offset + getTexelIndex(face, uint2(x, y), size));
    storeTexel(light.diffuse, id.x, sum);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradients(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint specularValues = getLevelOffset(light, EnvironmentLight::levelCount) * 3;
    if (id.x < specularValues)
        atomicStore(light.specularGradient + id.x, 0.0);
    if (id.x < getLevelTexelCount(light, getLastLevel()) * 3)
        atomicStore(light.diffuseGradient + id.x, 0.0);
}

//! The gradient of a source texel gathers over the outputs whose cone holds it; the cone test is symmetric, so the same bounds find them.
[shader("compute")]
[numthreads(64, 1, 1)]
void filterSpecularBackward(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint size = getLevelSize(light, light.level);
    if (id.x >= getLevelTexelCount(light, light.level))
        return;
    const LevelTexel source = getLevelTexel(id.x, size);
    const float3 direction = getTexelDirection(source.face, source.texel, size);
    const float area = getTexelArea(source.texel, size);
    const float roughness = getLevelRoughness(light.level);
    const float alphaSquared = roughness * roughness * roughness * roughness;
    const float cutoff = getLevelCutoff(light, light.level);
    const uint offset = getLevelOffset(light, light.level);
    float3 gradient = float3(0.0);
    for (uint face = 0; face < cubeFaces; ++face) {
        const int4 bounds = getConeBounds(direction, cutoff, face, size);
        for (int y = bounds.z; y <= bounds.w; ++y)
            for (int x = bounds.x; x <= bounds.y; ++x) {
                const float3 axis = getTexelDirection(face, uint2(x, y), size);
                if (dot(direction, axis) < cutoff)
                    continue;
                const uint output = offset + getTexelIndex(face, uint2(x, y), size);
                const float weights = light.specular[output].w;
                if (weights <= 0.0)
                    continue;
                gradient += getSpecularWeight(axis, direction, alphaSquared, area) / weights * loadAtomicTexel(light.specularGradient, output);
            }
    }
    storeTexel(light.levelGradient, offset + id.x, gradient);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void filterDiffuseBackward(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint last = getLastLevel();
    const uint size = getLevelSize(light, last);
    if (id.x >= getLevelTexelCount(light, last))
        return;
    const LevelTexel source = getLevelTexel(id.x, size);
    const float3 direction = getTexelDirection(source.face, source.texel, size);
    const float area = getTexelArea(source.texel, size);
    float3 gradient = float3(0.0);
    for (uint face = 0; face < cubeFaces; ++face)
        for (uint y = 0; y < size; ++y)
            for (uint x = 0; x < size; ++x)
                gradient += getDiffuseWeight(getTexelDirection(face, uint2(x, y), size), direction, area)
                            * loadAtomicTexel(light.diffuseGradient, getTexelIndex(face, uint2(x, y), size));
    const uint index = getLevelOffset(light, last) + id.x;
    storeTexel(light.levelGradient, index, loadTexel(light.levelGradient, index) + gradient);
}

//! The box filter of downsample spreads the gradient of a texel evenly over its four sources.
[shader("compute")]
[numthreads(256, 1, 1)]
void upsampleGradient(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    const uint size = getLevelSize(light, light.level);
    if (id.x >= getLevelTexelCount(light, light.level))
        return;
    const LevelTexel target = getLevelTexel(id.x, size);
    const uint parent = getLevelOffset(light, light.level + 1) + getTexelIndex(target.face, target.texel / 2, size / 2);
    const uint index = getLevelOffset(light, light.level) + id.x;
    storeTexel(light.levelGradient, index, loadTexel(light.levelGradient, index) + 0.25 * loadTexel(light.levelGradient, parent));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void finishBaseGradient(uint3 id : SV_DispatchThreadID) {
    const Arguments arguments = rootArguments<Arguments>();
    const Light light = arguments.light;
    const uint count = getLevelTexelCount(light, 0);
    if (id.x >= count)
        return;
    const float3 value = loadTexel(light.levels, id.x);
    const float3 signs = sign(value - (value.x + value.y + value.z) / 3.0);
    const float3 white = (signs - (signs.x + signs.y + signs.z) / 3.0) * arguments.whiteWeight / float(count * 3);
    storeTexel(light.levelGradient, id.x, (loadTexel(light.levelGradient, id.x) + white) * arguments.gradientScale);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clampBase(uint3 id : SV_DispatchThreadID) {
    const Light light = rootArguments<Arguments>().light;
    if (id.x < getLevelTexelCount(light, 0) * 3)
        light.levels[id.x] = max(light.levels[id.x], 0.0);
}
