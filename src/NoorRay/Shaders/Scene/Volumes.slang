import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Frame;

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

float gridValue(VolumeRecord volume, uint3 cell) {
    const uint3 resolution = volume.resolution.xyz;
    const uint index = (cell.z * resolution.y + cell.y) * resolution.x + cell.x;
    const uint word = volume.density[index >> 1];
    return f16tof32((index & 1) != 0 ? word >> 16 : word & 0xffffu);
}

//! Density at a world point, trilinear over cell centres, times the density scale.
float volumeDensity(VolumeRecord volume, float3 worldPoint) {
    const float scale = volume.boundsMin.w;
    const float3 p = transformPoint(volume.worldToObject, worldPoint);
    if (any(p < volume.boundsMin.xyz) || any(p > volume.boundsMax.xyz))
        return 0.0;
    if (volume.kind != volumeKindGrid)
        return scale;
    const float3 extent = volume.boundsMax.xyz - volume.boundsMin.xyz;
    const float3 uvw = (p - volume.boundsMin.xyz) / max(extent, float3(1e-20, 1e-20, 1e-20));
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return 0.0;
    const uint3 resolution = max(volume.resolution.xyz, uint3(1, 1, 1));
    const float3 position = uvw * float3(resolution) - 0.5;
    const float3 base = floor(position);
    const float3 f = position - base;
    const int3 cell = int3(base);
    const int3 last = int3(resolution) - 1;
    const uint3 c0 = uint3(clamp(cell, int3(0, 0, 0), last));
    const uint3 c1 = uint3(clamp(cell + 1, int3(0, 0, 0), last));
    const float d000 = gridValue(volume, uint3(c0.x, c0.y, c0.z));
    const float d100 = gridValue(volume, uint3(c1.x, c0.y, c0.z));
    const float d010 = gridValue(volume, uint3(c0.x, c1.y, c0.z));
    const float d110 = gridValue(volume, uint3(c1.x, c1.y, c0.z));
    const float d001 = gridValue(volume, uint3(c0.x, c0.y, c1.z));
    const float d101 = gridValue(volume, uint3(c1.x, c0.y, c1.z));
    const float d011 = gridValue(volume, uint3(c0.x, c1.y, c1.z));
    const float d111 = gridValue(volume, uint3(c1.x, c1.y, c1.z));
    const float d00 = lerp(d000, d100, f.x);
    const float d10 = lerp(d010, d110, f.x);
    const float d01 = lerp(d001, d101, f.x);
    const float d11 = lerp(d011, d111, f.x);
    return max(lerp(lerp(d00, d10, f.y), lerp(d01, d11, f.y), f.z), 0.0) * scale;
}

//! An upper bound of the extinction (any wavelength) inside the volume: spectra from the RGB basis never exceed the largest RGB component.
float volumeMajorant(VolumeRecord volume) { return maxComponent(volume.absorption.xyz + volume.scattering.xyz) * volume.absorption.w * volume.boundsMin.w; }

//! cosTheta is between the propagation direction and the scattered direction; g > 0 scatters forward.
float henyeyGreenstein(float cosTheta, float g) {
    const float denominator = 1.0 + g * g - 2.0 * g * cosTheta;
    return invFourPi * (1.0 - g * g) / (denominator * safeSqrt(denominator));
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
float phaseValue(VolumeRecord volume, float3 direction, float3 next) { return henyeyGreenstein(dot(direction, next), volume.boundsMax.w); }
