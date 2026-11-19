import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Random;
import Scene.Volumes;
import Spectral.Wavelengths;

static const uint mediumEventNone = 0u;
static const uint mediumEventScatter = 1u;
static const uint mediumEventAbsorb = 2u;
static const uint maxTrackingSteps = 256u;

struct MediumEvent {
    uint kind;
    float t;
    uint volume;
};

struct MediumRange {
    float majorant;
    float tStart;
    float tEnd;
};

//! The union of all volume boxes along the ray with the sum of their majorants.
MediumRange mediumRange(SceneRecord scene, float3 origin, float3 direction, float tMax) {
    MediumRange range;
    range.majorant = 0.0;
    range.tStart = tMax;
    range.tEnd = 0.0;
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
        float tNear;
        float tFar;
        if (!intersectVolume(volume, origin, direction, tMax, tNear, tFar))
            continue;
        range.majorant += volumeMajorant(volume);
        range.tStart = min(range.tStart, tNear);
        range.tEnd = max(range.tEnd, tFar);
    }
    return range;
}

struct MediumCoefficients {
    float4 absorption;
    float4 scattering;
    float4 emission;
    uint volume;
};

MediumCoefficients mediumAt(SceneRecord scene, float3 point, Wavelengths wavelengths) {
    MediumCoefficients coefficients;
    coefficients.absorption = float4(0.0, 0.0, 0.0, 0.0);
    coefficients.scattering = float4(0.0, 0.0, 0.0, 0.0);
    coefficients.emission = float4(0.0, 0.0, 0.0, 0.0);
    coefficients.volume = invalidIndex;
    float strongest = 0.0;
    for (uint index = 0; index < scene.counts.w; ++index) {
        const VolumeRecord volume = scene.volumes[index];
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

//! Spectral delta tracking (Kutz et al. 2017) with event probabilities from the wavelength mean. Updates throughput and adds absorbed emission to emitted.
MediumEvent trackMedium(SceneRecord scene, float3 origin, float3 direction, float tMax, Wavelengths wavelengths, uint seed, inout float4 throughput,
                        out float4 emitted) {
    MediumEvent event;
    event.kind = mediumEventNone;
    event.t = tMax;
    event.volume = invalidIndex;
    emitted = float4(0.0, 0.0, 0.0, 0.0);
    const MediumRange range = mediumRange(scene, origin, direction, tMax);
    if (range.majorant <= 0.0 || range.tStart >= range.tEnd)
        return event;
    Random random = makeRandom(seed);
    float t = range.tStart;
    for (uint step = 0; step < maxTrackingSteps; ++step) {
        t -= log(1.0 - random.next()) / range.majorant;
        if (t >= range.tEnd)
            return event;
        const MediumCoefficients medium = mediumAt(scene, origin + direction * t, wavelengths);
        const float absorbProbability = saturate(spectralMean(medium.absorption, wavelengths) / range.majorant);
        const float scatterProbability = min(saturate(spectralMean(medium.scattering, wavelengths) / range.majorant), 1.0 - absorbProbability);
        const float nullProbability = 1.0 - absorbProbability - scatterProbability;
        const float u = random.next();
        if (u < absorbProbability) {
            emitted = throughput * medium.emission / (range.majorant * absorbProbability);
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

//! Ratio tracking: the spectral transmittance estimate between origin and tMax.
float4 mediumTransmittance(SceneRecord scene, float3 origin, float3 direction, float tMax, Wavelengths wavelengths, uint seed) {
    const MediumRange range = mediumRange(scene, origin, direction, tMax);
    float4 transmittance = float4(1.0, 1.0, 1.0, 1.0);
    if (range.majorant <= 0.0 || range.tStart >= range.tEnd)
        return transmittance;
    Random random = makeRandom(seed);
    float t = range.tStart;
    for (uint step = 0; step < maxTrackingSteps; ++step) {
        t -= log(1.0 - random.next()) / range.majorant;
        if (t >= range.tEnd)
            return transmittance;
        const MediumCoefficients medium = mediumAt(scene, origin + direction * t, wavelengths);
        transmittance *= max(1.0 - (medium.absorption + medium.scattering) / range.majorant, 0.0);
        if (maxComponent(transmittance.xyz) <= 0.0 && transmittance.w <= 0.0)
            return transmittance;
    }
    return transmittance;
}
