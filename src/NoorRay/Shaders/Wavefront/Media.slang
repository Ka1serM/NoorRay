import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Random;
import Scene.Volumes;
import Spectral.Wavelengths;

static const uint mediumEventNone = 0u;
static const uint mediumEventScatter = 1u;
static const uint mediumEventAbsorb = 2u;
static const uint maxTrackingSteps = 1024u;

struct MediumEvent {
    uint kind;
    float t;
    uint volume;
};

struct MediumRange {
    float majorant;
    float tStart;
    float tEnd;
    //! The first volume box along the ray.
    uint volume;
    //! The same for the grid volumes alone, which ratio tracking steps through; homogeneous volumes attenuate in closed form.
    float gridMajorant;
    float gridStart;
    float gridEnd;
};

bool isGridVolume(VolumeRecord volume) { return volumeKind(volume) != volumeKindHomogeneous; }

//! The union of all volume boxes along the ray with the sum of their majorants.
MediumRange mediumRange(SceneRecord scene, float3 origin, float3 direction, float tMax) {
    MediumRange range;
    range.majorant = 0.0;
    range.tStart = tMax;
    range.tEnd = 0.0;
    range.volume = invalidIndex;
    range.gridMajorant = 0.0;
    range.gridStart = tMax;
    range.gridEnd = 0.0;
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
        float tNear;
        float tFar;
        if (!intersectVolume(volume, origin, direction, tMax, tNear, tFar))
            continue;
        range.majorant += volumeMajorant(volume);
        if (tNear < range.tStart)
            range.volume = index;
        range.tStart = min(range.tStart, tNear);
        range.tEnd = max(range.tEnd, tFar);
        if (!isGridVolume(volume))
            continue;
        range.gridMajorant += volumeMajorant(volume);
        range.gridStart = min(range.gridStart, tNear);
        range.gridEnd = max(range.gridEnd, tFar);
    }
    return range;
}

struct MediumCoefficients {
    float4 absorption;
    float4 scattering;
    float4 emission;
    uint volume;
};

MediumCoefficients mediumAt(SceneRecord scene, float3 point, Wavelengths wavelengths, bool gridOnly = false) {
    MediumCoefficients coefficients;
    coefficients.absorption = float4(0.0, 0.0, 0.0, 0.0);
    coefficients.scattering = float4(0.0, 0.0, 0.0, 0.0);
    coefficients.emission = float4(0.0, 0.0, 0.0, 0.0);
    coefficients.volume = invalidIndex;
    float strongest = 0.0;
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
        if (gridOnly && !isGridVolume(volume))
            continue;
        if (volume.maskResolution.w != 0)
            coefficients.emission += max(wavelengths.upsample(volumeEmissionRate(volume, point)), 0.0);
        const float density = volumeDensity(volume, point);
        if (density <= 0.0)
            continue;
        const float4 absorption = density * max(wavelengths.upsample(volume.absorption.xyz), 0.0);
        const float4 scattering = density * max(wavelengths.upsample(volume.scattering.xyz), 0.0);
        coefficients.absorption += absorption;
        coefficients.scattering += scattering;
        coefficients.emission += absorption * max(wavelengths.upsample(volume.emission.xyz), 0.0);
        const float strength = maxComponent(volume.scattering.xyz) * density;
        if (strength > strongest) {
            strongest = strength;
            coefficients.volume = index;
        }
    }
    return coefficients;
}

float spectralMean(float4 value, Wavelengths wavelengths) { return dot(value, wavelengths.weights) * 0.25; }

bool hasMedium(MediumRange range) { return range.majorant > 0.0 && range.tStart < range.tEnd; }

struct MediumProbabilities {
    float absorb;
    float scatter;
    float nullEvent;
};

MediumProbabilities mediumProbabilities(MediumCoefficients medium, float majorant, Wavelengths wavelengths) {
    MediumProbabilities probabilities;
    probabilities.absorb = saturate(spectralMean(medium.absorption, wavelengths) / majorant);
    probabilities.scatter = min(saturate(spectralMean(medium.scattering, wavelengths) / majorant), 1.0 - probabilities.absorb);
    probabilities.nullEvent = 1.0 - probabilities.absorb - probabilities.scatter;
    return probabilities;
}

//! The spectral transmittance of the homogeneous volumes between t0 and t1, in closed form.
float4 homogeneousTransmittance(SceneRecord scene, float3 origin, float3 direction, float t0, float t1, Wavelengths wavelengths) {
    float4 depth = float4(0.0, 0.0, 0.0, 0.0);
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
        float tNear;
        float tFar;
        if (isGridVolume(volume) || !intersectVolume(volume, origin, direction, t1, tNear, tFar))
            continue;
        const float overlap = max(tFar - max(tNear, t0), 0.0);
        depth += overlap * volume.boundsMin.w * max(wavelengths.upsample(volume.absorption.xyz) + wavelengths.upsample(volume.scattering.xyz), 0.0);
    }
    return exp(-depth);
}

//! Ratio tracking through the grid volumes between t0 and t1.
float4 gridTransmittance(SceneRecord scene, MediumRange range, float3 origin, float3 direction, float t0, float t1, Wavelengths wavelengths, uint seed) {
    float4 transmittance = float4(1.0, 1.0, 1.0, 1.0);
    const float end = min(t1, range.gridEnd);
    if (range.gridMajorant <= 0.0)
        return transmittance;
    Random random = makeRandom(seed);
    float t = max(t0, range.gridStart);
    for (uint step = 0; step < maxTrackingSteps; ++step) {
        t -= log(1.0 - random.next()) / range.gridMajorant;
        if (t >= end)
            return transmittance;
        const MediumCoefficients medium = mediumAt(scene, origin + direction * t, wavelengths, true);
        transmittance *= max(1.0 - (medium.absorption + medium.scattering) / range.gridMajorant, 0.0);
        if (all(transmittance <= 0.0))
            return transmittance;
    }
    return transmittance;
}

float4 transmittanceBetween(SceneRecord scene, MediumRange range, float3 origin, float3 direction, float t0, float t1, Wavelengths wavelengths, uint seed) {
    return homogeneousTransmittance(scene, origin, direction, t0, t1, wavelengths) *
           gridTransmittance(scene, range, origin, direction, t0, t1, wavelengths, seed);
}

//! Spectral delta tracking (Kutz 2017) by the wavelength mean; without scatters the media only attenuate, as Cycles past the volume bounces.
MediumEvent trackMedium(SceneRecord scene, MediumRange range, float3 origin, float3 direction, float tMax, Wavelengths wavelengths, uint seed,
                        bool scatters, inout float4 throughput, out float4 emitted) {
    MediumEvent event;
    event.kind = mediumEventNone;
    event.t = tMax;
    event.volume = invalidIndex;
    emitted = float4(0.0, 0.0, 0.0, 0.0);
    if (!hasMedium(range))
        return event;
    if (!scatters) {
        throughput *= transmittanceBetween(scene, range, origin, direction, 0.0, tMax, wavelengths, seed);
        return event;
    }
    Random random = makeRandom(seed);
    float t = range.tStart;
    for (uint step = 0; step < maxTrackingSteps; ++step) {
        t -= log(1.0 - random.next()) / range.majorant;
        if (t >= range.tEnd)
            return event;
        const MediumCoefficients medium = mediumAt(scene, origin + direction * t, wavelengths);
        //! Every collision takes the emission, so a volume that only emits does too.
        emitted += throughput * medium.emission / range.majorant;
        const MediumProbabilities probabilities = mediumProbabilities(medium, range.majorant, wavelengths);
        const float absorbProbability = probabilities.absorb;
        const float scatterProbability = probabilities.scatter;
        const float nullProbability = probabilities.nullEvent;
        const float u = random.next();
        if (u < absorbProbability) {
            event.kind = mediumEventAbsorb;
            event.t = t;
            return event;
        }
        if (u < absorbProbability + scatterProbability) {
            throughput *= medium.scattering / (range.majorant * scatterProbability);
            event.kind = mediumEventScatter;
            event.t = t;
            event.volume = medium.volume;
            return event;
        }
        if (nullProbability <= 0.0) {
            throughput = float4(0.0, 0.0, 0.0, 0.0);
            event.kind = mediumEventAbsorb;
            event.t = t;
            return event;
        }
        throughput *= max(range.majorant - medium.absorption - medium.scattering, 0.0) / (range.majorant * nullProbability);
    }
    return event;
}

//! A point of the ray with the estimate of the spectral transmittance to it.
struct MediumPoint {
    float4 transmittance;
    MediumCoefficients medium;
};

MediumPoint trackToPoint(SceneRecord scene, MediumRange range, float3 origin, float3 direction, float t, Wavelengths wavelengths, uint seed) {
    MediumPoint point;
    point.transmittance = transmittanceBetween(scene, range, origin, direction, 0.0, t, wavelengths, seed);
    point.medium = mediumAt(scene, origin + direction * t, wavelengths);
    return point;
}

//! Equiangular sampling (Kulla and Fajardo 2012) of a distance in [tStart, tEnd] towards a light point at offset along the ray and height from it.
struct Equiangular {
    float offset;
    float height;
    float thetaStart;
    float thetaEnd;

    bool isValid() { return height > 0.0 && thetaEnd - thetaStart > 1e-6; }

    float sample(float u, out float pdf) {
        const float along = height * tan(lerp(thetaStart, thetaEnd, u));
        pdf = height / ((thetaEnd - thetaStart) * (height * height + along * along));
        return offset + along;
    }

    float pdf(float t) {
        const float along = t - offset;
        return height / ((thetaEnd - thetaStart) * (height * height + along * along));
    }
};

Equiangular makeEquiangular(float3 origin, float3 direction, float3 light, float tStart, float tEnd) {
    Equiangular sampler;
    sampler.offset = dot(light - origin, direction);
    sampler.height = length(light - origin - direction * sampler.offset);
    sampler.thetaStart = atan2(tStart - sampler.offset, sampler.height);
    sampler.thetaEnd = atan2(tEnd - sampler.offset, sampler.height);
    return sampler;
}

//! The spectral transmittance estimate between origin and tMax.
float4 mediumTransmittance(SceneRecord scene, float3 origin, float3 direction, float tMax, Wavelengths wavelengths, uint seed) {
    const MediumRange range = mediumRange(scene, origin, direction, tMax);
    if (!hasMedium(range))
        return float4(1.0, 1.0, 1.0, 1.0);
    return transmittanceBetween(scene, range, origin, direction, 0.0, tMax, wavelengths, seed);
}

static const uint guideSteps = 16u;
static const uint guideBisections = 12u;

//! The mean RGB optical depth of the homogeneous volumes from the ray origin to t.
float homogeneousDepth(SceneRecord scene, float3 origin, float3 direction, float t) {
    float depth = 0.0;
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
        float tNear;
        float tFar;
        if (isGridVolume(volume) || !intersectVolume(volume, origin, direction, t, tNear, tFar))
            continue;
        depth += (tFar - tNear) * volume.boundsMin.w * dot(volume.absorption.xyz + volume.scattering.xyz, float3(1.0, 1.0, 1.0)) / 3.0;
    }
    return depth;
}

//! The RGB extinction and scattering of all volumes at a point.
float3 rgbExtinction(SceneRecord scene, float3 point, bool gridOnly, out float3 scattering) {
    float3 extinction = float3(0.0, 0.0, 0.0);
    scattering = float3(0.0, 0.0, 0.0);
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
        if (gridOnly && !isGridVolume(volume))
            continue;
        const float density = volumeDensity(volume, point);
        extinction += density * max(volume.absorption.xyz + volume.scattering.xyz, 0.0);
        scattering += density * max(volume.scattering.xyz, 0.0);
    }
    return extinction;
}

//! A fixed march, so that the guide of a camera ray in a medium stays the same from frame to frame; t is 0 when more than half passes before tEnd.
struct MediumGuide {
    float t;
    float3 albedo;
};

MediumGuide mediumGuide(SceneRecord scene, MediumRange range, float3 origin, float3 direction, float tEnd) {
    MediumGuide guide;
    guide.t = 0.0;
    guide.albedo = float3(1.0, 1.0, 1.0);
    const float end = min(tEnd, range.tEnd);
    const float gridStart = range.gridStart;
    const float gridEnd = max(min(range.gridEnd, end), gridStart);
    const float stepLength = (gridEnd - gridStart) / float(guideSteps);
    float gridDepth[guideSteps + 1];
    gridDepth[0] = 0.0;
    float3 scattering;
    for (uint step = 0; step < guideSteps; ++step) {
        const float3 point = origin + direction * (gridStart + (float(step) + 0.5) * stepLength);
        const float3 extinction = stepLength > 0.0 ? rgbExtinction(scene, point, true, scattering) : float3(0.0, 0.0, 0.0);
        gridDepth[step + 1] = gridDepth[step] + stepLength * dot(extinction, float3(1.0, 1.0, 1.0)) / 3.0;
    }
    const float halfDepth = log(2.0);
    float low = range.tStart;
    float high = end;
    if (!(high > low) || homogeneousDepth(scene, origin, direction, high) + gridDepth[guideSteps] < halfDepth)
        return guide;
    for (uint i = 0; i < guideBisections; ++i) {
        const float middle = 0.5 * (low + high);
        const float position = stepLength > 0.0 ? clamp((middle - gridStart) / stepLength, 0.0, float(guideSteps)) : 0.0;
        const uint index = min(uint(position), guideSteps - 1u);
        const float depth = homogeneousDepth(scene, origin, direction, middle) + lerp(gridDepth[index], gridDepth[index + 1], position - float(index));
        if (depth < halfDepth)
            low = middle;
        else
            high = middle;
    }
    guide.t = high;
    const float3 extinction = rgbExtinction(scene, origin + direction * high, false, scattering);
    guide.albedo = select(extinction > 0.0, saturate(scattering / max(extinction, 1e-20)), float3(1.0, 1.0, 1.0));
    return guide;
}
