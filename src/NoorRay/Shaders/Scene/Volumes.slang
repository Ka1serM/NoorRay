import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Frame;
import Materials.Graph;

//! The bake samples the graph only at voxel centres, so the true largest density can lie a little above the baked one.
static const float graphMajorantMargin = 1.2;

uint volumeKind(VolumeRecord volume) { return volume.kind & ((1u << lightSetShift) - 1u); }

//! The parametric range of a world ray inside the volume box; false when it misses.
bool intersectVolume(VolumeRecord volume, float3 origin, float3 direction, float tMax, out float tNear, out float tFar) {
    const float3 o = transformPoint(volume.worldToObject, origin);
    const float3 d = transformVector(volume.worldToObject, direction);
    const float3 inverse = float3(d.x != 0.0 ? 1.0 / d.x : floatMax, d.y != 0.0 ? 1.0 / d.y : floatMax, d.z != 0.0 ? 1.0 / d.z : floatMax);
    const float3 t0 = (volume.boundsMin.xyz - o) * inverse;
    const float3 t1 = (volume.boundsMax.xyz - o) * inverse;
    const float3 lower = min(t0, t1);
    const float3 upper = max(t0, t1);
    tNear = max(max(lower.x, lower.y), max(lower.z, 0.0));
    tFar = min(min(upper.x, upper.y), min(upper.z, tMax));
    return tNear < tFar;
}

bool insideVolume(VolumeRecord volume, float3 worldPoint) {
    const float3 p = transformPoint(volume.worldToObject, worldPoint);
    return all(p >= volume.boundsMin.xyz) && all(p <= volume.boundsMax.xyz);
}

float gridValue(Ptr<uint> values, uint3 resolution, uint3 cell) {
    const uint index = (cell.z * resolution.y + cell.y) * resolution.x + cell.x;
    const uint word = values[index >> 1];
    return f16tof32((index & 1) != 0 ? word >> 16 : word & 0xffffu);
}

//! A point of a graph volume as Shaders/Entries/BakeGraph.slang shades a voxel: Generated over the bounds from 0 to 1, the normal the object +z.
ShadingContext volumeContext(VolumeRecord volume, float3 worldPoint, float3 local, float3 generated) {
    const float3 r0 = volume.worldToObject[0].xyz;
    const float3 r1 = volume.worldToObject[1].xyz;
    const float3 r2 = volume.worldToObject[2].xyz;
    //! The columns of the inverse 3x3 are the cross products of the rows over the determinant.
    const float inverseDeterminant = 1.0 / dot(r0, cross(r1, r2));
    const float3 column0 = cross(r1, r2) * inverseDeterminant;
    const float3 column1 = cross(r2, r0) * inverseDeterminant;
    const float3 column2 = cross(r0, r1) * inverseDeterminant;
    const float3 translation = float3(volume.worldToObject[0].w, volume.worldToObject[1].w, volume.worldToObject[2].w);
    const float3 origin = -(column0 * translation.x + column1 * translation.y + column2 * translation.z);
    const float3 normal = normalize(column2);
    ShadingContext context;
    context.position = float4(worldPoint, 1.0);
    context.normal = float4(normal, 0.0);
    context.geometricNormal = float4(normal, 0.0);
    context.tangent = float4(normalize(column0), 1.0);
    context.incoming = float4(normal, 1.0);
    context.uv01 = float4(generated.xy, generated.xy);
    context.uv23 = float4(generated.xy, generated.xy);
    context.color = float4(1.0, 1.0, 1.0, 1.0);
    context.uvGradients = float4(0.0, 0.0, 0.0, 0.0);
    context.dPdx = float4(0.0, 0.0, 0.0, 0.0);
    context.dPdy = float4(0.0, 0.0, 0.0, 0.0);
    context.objectPosition = float4(local, 0.0);
    context.generated = float4(generated, 0.0);
    context.objectLocation = float4(origin, 0.0);
    context.objectDx = float4(0.0, 0.0, 0.0, 0.0);
    context.objectDy = float4(0.0, 0.0, 0.0, 0.0);
    context.generatedScale = float4(1.0, 1.0, 1.0, 0.0);
    context.worldToObject0 = volume.worldToObject[0];
    context.worldToObject1 = volume.worldToObject[1];
    context.worldToObject2 = volume.worldToObject[2];
    context.objectToWorld0 = float4(column0.x, column1.x, column2.x, origin.x);
    context.objectToWorld1 = float4(column0.y, column1.y, column2.y, origin.y);
    context.objectToWorld2 = float4(column0.z, column1.z, column2.z, origin.z);
    setCameraRay(context);
    return context;
}

//! A grid of half pairs, trilinear over its cell centres.
float gridDensity(Ptr<uint> values, uint3 size, float3 uvw) {
    const uint3 resolution = max(size, uint3(1, 1, 1));
    const float3 position = uvw * float3(resolution) - 0.5;
    const float3 base = floor(position);
    const float3 f = position - base;
    const int3 cell = int3(base);
    const int3 last = int3(resolution) - 1;
    const uint3 c0 = uint3(clamp(cell, int3(0, 0, 0), last));
    const uint3 c1 = uint3(clamp(cell + 1, int3(0, 0, 0), last));
    const float d000 = gridValue(values, resolution, uint3(c0.x, c0.y, c0.z));
    const float d100 = gridValue(values, resolution, uint3(c1.x, c0.y, c0.z));
    const float d010 = gridValue(values, resolution, uint3(c0.x, c1.y, c0.z));
    const float d110 = gridValue(values, resolution, uint3(c1.x, c1.y, c0.z));
    const float d001 = gridValue(values, resolution, uint3(c0.x, c0.y, c1.z));
    const float d101 = gridValue(values, resolution, uint3(c1.x, c0.y, c1.z));
    const float d011 = gridValue(values, resolution, uint3(c0.x, c1.y, c1.z));
    const float d111 = gridValue(values, resolution, uint3(c1.x, c1.y, c1.z));
    const float d00 = lerp(d000, d100, f.x);
    const float d10 = lerp(d010, d110, f.x);
    const float d01 = lerp(d001, d101, f.x);
    const float d11 = lerp(d011, d111, f.x);
    return max(lerp(lerp(d00, d10, f.y), lerp(d01, d11, f.y), f.z), 0.0);
}

uint brickLevel(Ptr<uint> brick, uint3 s) {
    const uint sample = (s.z * volumeBrickSamples + s.y) * volumeBrickSamples + s.x;
    return (brick[sample >> 2] >> (8 * (sample & 3u))) & 0xffu;
}

//! One cell read, and for a cell with a brick a trilinear lookup of its corner samples, which never leaves the brick.
float sparseDensity(VolumeRecord volume, float3 uvw) {
    const uint3 cells = max(volume.resolution.xyz, uint3(1, 1, 1));
    const float3 position = uvw * float3(cells * volumeBrickSize);
    const uint3 cell = min(uint3(position) / volumeBrickSize, cells - 1);
    const uint2 record = volume.cells[(cell.z * cells.y + cell.y) * cells.x + cell.x];
    const float low = f16tof32(record.x & 0xffffu);
    if (record.y == invalidIndex)
        return low;
    const float3 local = position - float3(cell * volumeBrickSize);
    const uint3 s = min(uint3(local), uint3(volumeBrickSize - 1u, volumeBrickSize - 1u, volumeBrickSize - 1u));
    const float3 f = local - float3(s);
    const Ptr<uint> brick = volume.bricks + record.y * volumeBrickWords;
    const float d00 = lerp(float(brickLevel(brick, s)), float(brickLevel(brick, s + uint3(1, 0, 0))), f.x);
    const float d10 = lerp(float(brickLevel(brick, s + uint3(0, 1, 0))), float(brickLevel(brick, s + uint3(1, 1, 0))), f.x);
    const float d01 = lerp(float(brickLevel(brick, s + uint3(0, 0, 1))), float(brickLevel(brick, s + uint3(1, 0, 1))), f.x);
    const float d11 = lerp(float(brickLevel(brick, s + uint3(0, 1, 1))), float(brickLevel(brick, s + uint3(1, 1, 1))), f.x);
    const float level = lerp(lerp(d00, d10, f.y), lerp(d01, d11, f.y), f.z);
    return low + level * (1.0 / 255.0) * (f16tof32(record.x >> 16) - low);
}

float3 volumePosition(VolumeRecord volume, float3 local) {
    const float3 extent = volume.boundsMax.xyz - volume.boundsMin.xyz;
    return saturate((local - volume.boundsMin.xyz) / max(extent, float3(1e-20, 1e-20, 1e-20)));
}

//! The density of a grid inside the bounds: a grid trilinear over cell centres, a sparse grid in its bricks, a graph volume its graph at the point.
float filledDensity(VolumeRecord volume, uint kind, float3 worldPoint, float3 local, float3 uvw) {
    if (kind == volumeKindGrid)
        return gridDensity(volume.density, volume.resolution.xyz, uvw);
    if (kind == volumeKindSparse)
        return sparseDensity(volume, uvw);
    const MaterialGraphOutput output =
        evaluateMaterialGraph(volumeContext(volume, worldPoint, local, uvw), graphMaterial(volume.graphParameters, volume.graphTextures));
    return max(output.emission.x, 0.0);
}

//! Density at a world point times the density scale and the share that the volume fills.
float volumeDensity(VolumeRecord volume, float3 worldPoint) {
    const float scale = volume.boundsMin.w;
    const float3 p = transformPoint(volume.worldToObject, worldPoint);
    if (any(p < volume.boundsMin.xyz) || any(p > volume.boundsMax.xyz))
        return 0.0;
    const uint kind = volumeKind(volume);
    if (kind == volumeKindHomogeneous)
        return scale;
    const float3 uvw = volumePosition(volume, p);
    const float filled = volume.maskResolution.x != 0 ? gridDensity(volume.mask, volume.maskResolution.xyz, uvw) : 1.0;
    if (filled <= 0.0)
        return 0.0;
    return filled * filledDensity(volume, kind, worldPoint, p, uvw) * scale;
}

float3 gridEmission(VolumeRecord volume, float3 uvw) {
    const uint3 resolution = max(volume.resolution.xyz, uint3(1, 1, 1));
    const float3 position = uvw * float3(resolution) - 0.5;
    const float3 base = floor(position);
    const float3 f = position - base;
    const int3 last = int3(resolution) - 1;
    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint corner = 0; corner < 8; ++corner) {
        const int3 offset = int3(corner & 1u, (corner >> 1) & 1u, corner >> 2);
        const uint3 cell = uint3(clamp(int3(base) + offset, int3(0, 0, 0), last));
        const float3 weight = lerp(1.0 - f, f, float3(offset));
        sum += weight.x * weight.y * weight.z * unpackRgb9e5(volume.emissionGrid[(cell.z * resolution.y + cell.y) * resolution.x + cell.x]);
    }
    return sum;
}

//! The emission per metre at a world point that does not depend on the density.
float3 volumeEmissionRate(VolumeRecord volume, float3 worldPoint) {
    const float3 rate = float3(volume.absorption.w, volume.scattering.w, volume.emission.w);
    const float3 p = transformPoint(volume.worldToObject, worldPoint);
    if (volume.maskResolution.w == 0 || any(p < volume.boundsMin.xyz) || any(p > volume.boundsMax.xyz))
        return float3(0.0, 0.0, 0.0);
    const float3 uvw = volumePosition(volume, p);
    const float filled = volume.maskResolution.x != 0 ? gridDensity(volume.mask, volume.maskResolution.xyz, uvw) : 1.0;
    return filled * (rate + (volume.flags.x != 0 ? gridEmission(volume, uvw) : float3(0.0, 0.0, 0.0)));
}

//! An upper bound of the extinction (any wavelength) inside the volume: spectra from the RGB basis never exceed the largest RGB component.
float volumeMajorant(VolumeRecord volume) {
    const uint kind = volumeKind(volume);
    const float baked = kind == volumeKindHomogeneous ? 1.0 : asfloat(atomicLoad(volume.maximum));
    const float density = kind == volumeKindGraph ? baked * graphMajorantMargin : baked;
    return max(maxComponent(volume.absorption.xyz + volume.scattering.xyz) * density * volume.boundsMin.w, asfloat(volume.maskResolution.w));
}

//! cosTheta is between the propagation direction and the scattered direction; g > 0 scatters forward.
float henyeyGreenstein(float cosTheta, float g) {
    const float denominator = 1.0 + g * g - 2.0 * g * cosTheta;
    return invFourPi * (1.0 - g * g) / (denominator * safeSqrt(denominator));
}

//! The Draine phase function of Cycles; alpha 0 is Henyey-Greenstein.
float draine(float cosTheta, float g, float alpha) {
    const float denominator = 1.0 + g * g - 2.0 * g * cosTheta;
    const float shape = (1.0 + alpha * cosTheta * cosTheta) / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0);
    return invFourPi * (1.0 - g * g) * shape / (denominator * safeSqrt(denominator));
}

//! Samples the incident-to-outgoing direction about the propagation direction wo (the ray direction); pdf equals the phase value.
float3 sampleHenyeyGreenstein(float3 direction, float g, float2 u, out float pdf) {
    float cosTheta;
    if (abs(g) < 1e-3)
        cosTheta = 1.0 - 2.0 * u.x;
    else {
        const float s = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.x);
        cosTheta = (1.0 + g * g - s * s) / (2.0 * g);
    }
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    const float sinTheta = safeSqrt(1.0 - cosTheta * cosTheta);
    const float phi = twoPi * u.y;
    const float3 local = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    pdf = henyeyGreenstein(cosTheta, g);
    return makeFrame(direction).toWorld(local);
}

//! The phase function between the ray direction and a new direction.
float phaseValue(VolumeRecord volume, float3 direction, float3 next) { return draine(dot(direction, next), volume.boundsMax.w, volume.draineAlpha); }
