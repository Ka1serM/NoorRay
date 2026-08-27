#include "RaytracingAbi.slang"
#include "../Gaussian/Gaussian.slang"
#include "AnalyticLights.slang"
#include "Environment.slang"

float3 sampleGaussianIsotropic(inout Rng rng)
{
    const float z = 2.0 * rng.nextFloat() - 1.0;
    const float phi = 6.28318530717958647692 * rng.nextFloat();
    const float radius = sqrt(max(1.0 - z * z, 0.0));
    return float3(radius * cos(phi), z, radius * sin(phi));
}

// Gaussian closest-hit consumes the candidate accepted by GaussianAnyHit and
// performs only the compact SH/AOV evaluation for the winning splat.
[shader("closesthit")]
void main(inout RayPayload payload)
{
    const FrameParams frame = *root.arguments;
    ByteAddressBuffer records = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.gaussianRecords, 0));
    const uint gaussianId = InstanceIndex() - frame.gaussianInstanceBase;
    payload.hit = 2u;
    payload.gaussianId = gaussianId;
    payload.cryptomatte = gaussianId;
    // Recompute for the committed instance: traversal may visit accepted
    // candidates out of distance order, so payload state from any-hit is not
    // guaranteed to belong to the final closest proxy triangle.
    float exactT;
    gaussianDistanceSq(ObjectRayOrigin(), ObjectRayDirection(), RayTMin(), exactT);
    payload.t = exactT;
    payload.position = WorldRayOrigin() + WorldRayDirection() * payload.t;
    ByteAddressBuffer spectralTables = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.spectralTables, 0));
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(
        payload.wavelengthSample);
    if (uint(frame.sceneAddress) != 0xFFFFFFFFu)
    {
        ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
            uint2(uint(frame.sceneAddress), 0));
        const float3 rayDirection = normalize(WorldRayDirection());
        const AnalyticLightHit lightHit = intersectAnalyticLights(lights,
            spectralTables, WorldRayOrigin(), rayDirection, 0.001,
            payload.t, wavelengths);
        if (lightHit.valid && lightHit.distance < payload.t)
        {
            payload.hit = 4u;
            payload.t = lightHit.distance;
            payload.position = WorldRayOrigin() + rayDirection * lightHit.distance;
            payload.emissionSpectrum = lightHit.radiance;
            payload.lightPdf = lightHit.pdf;
            return;
        }
    }

    const uint coefficientCount = min(gaussianUint(records, gaussianId,
        GaussianShCountOffset), frame.gaussianShCoefficientCount & 0xFFu);
    const uint order = coefficientCount >= 16u ? 3u
        : coefficientCount >= 9u ? 2u
        : coefficientCount >= 4u ? 1u : 0u;
    payload.radiance = max(evaluateGaussianSh(records, gaussianId, order,
        -WorldRayDirection()), 0.0);
    payload.albedo = payload.radiance;

    const float3 axis0 = gaussianRecordVector(records, gaussianId, 0u);
    const float3 axis1 = gaussianRecordVector(records, gaussianId, 1u);
    const float3 axis2 = gaussianRecordVector(records, gaussianId, 2u);
    const float3 relative = payload.position - gaussianRecordCenter(records,
        gaussianId);
    const float3 localPoint = float3(dot(relative, axis0) / max(dot(axis0, axis0), 1e-12),
        dot(relative, axis1) / max(dot(axis1, axis1), 1e-12),
        dot(relative, axis2) / max(dot(axis2, axis2), 1e-12));
    payload.normal = normalize(localPoint.x * axis0 / max(dot(axis0, axis0), 1e-12)
        + localPoint.y * axis1 / max(dot(axis1, axis1), 1e-12)
        + localPoint.z * axis2 / max(dot(axis2, axis2), 1e-12));
    // Acceptance in the any-hit stage is stochastic and therefore already
    // carries the Gaussian's density in expectation, in both shading modes. A
    // path that reaches closest hit is fully covered; republishing the density
    // as alpha would composite it a second time.
    payload.alpha = 1.0;

    payload.bsdfWeight = float4(0.0);
    payload.emissionSpectrum = float4(0.0);
    payload.nextDirection = float3(0.0);
    payload.sampleValid = 0u;
    payload.directSpectrum = float4(0.0);
    payload.shadowDirection = float3(0.0);
    payload.shadowDistance = 0.0;
    payload.shadowValid = 0u;
    payload.bsdfPdf = 0.0;

    // Direct colour reproduces 3DGS display semantics and terminates. GI
    // mirrors the renderer path: SH is isotropic spectral albedo, direct light
    // evaluates albedo/(4*pi), and continuation samples the sphere uniformly.
    if ((frame.gaussianShCoefficientCount & 0x80000000u) != 0u)
        return;

    payload.hit = 3u;
    const SampledSpectrum gaussianAlbedo = rgbAlbedoToSpectrum(
        spectralTables, payload.albedo, wavelengths);
    Rng rng;
    rng.state = payload.seed ^ (0xC2B2AE35u * (payload.depth + 1u));

    if (uint(frame.sceneAddress) != 0xFFFFFFFFu
        && frame.environmentBuffer != 0xFFFFFFFFu)
    {
        ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
            uint2(uint(frame.sceneAddress), 0));
        StructuredBuffer<EnvironmentRecord> environments =
            DescriptorHandle<StructuredBuffer<EnvironmentRecord>>(
                uint2(frame.environmentBuffer, 0));
        const EnvironmentRecord environment = environments[0];
        const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
        const float environmentWeight = max(environment.importanceWeight, 0.0);
        const float totalWeight = finiteWeight + environmentWeight;
        const float finiteProbability = totalWeight > 0.0
            ? finiteWeight / totalWeight : 0.0;
        const bool chooseFinite = finiteProbability > 0.0
            && (environmentWeight <= 0.0 || rng.nextFloat() < finiteProbability);
        SampledSpectrum radianceOverPdf = SampledSpectrum(0.0);
        float3 lightDirection = float3(0.0);
        float lightDistance = 0.0;
        bool lightValid = false;
        bool meshLight = false;
        if (chooseFinite)
        {
            const AnalyticLightSample light = sampleAnalyticLight(lights,
                spectralTables, payload.position, wavelengths, rng);
            lightDirection = light.direction;
            lightDistance = light.distance;
            radianceOverPdf = light.mesh
                ? SampledSpectrum(light.emissionMultiplier)
                : light.radianceOverSelectionPdf;
            radianceOverPdf /= max(finiteProbability, 1.0e-20);
            meshLight = light.mesh;
            payload.shadowInstance = light.instanceIndex;
            payload.shadowPrimitive = light.primitiveIndex;
            lightValid = light.valid;
        }
        else if (environmentWeight > 0.0)
        {
            const float environmentProbability = environmentWeight / totalWeight;
            const EnvironmentSample sample = sampleEnvironmentDirection(
                environment, rng);
            lightDirection = sample.direction;
            lightDistance = 1.0e16;
            radianceOverPdf = rgbIlluminantToSpectrum(spectralTables,
                environmentRadianceRgb(environment, sample.direction, false),
                wavelengths) / max(environmentProbability * sample.pdf, 1.0e-20);
            lightValid = sample.pdf > 0.0;
        }
        if (lightValid)
        {
            const SampledSpectrum direct = gaussianAlbedo * GaussianInv4Pi
                * radianceOverPdf;
            if (spectrumIsFinite(direct) && spectrumMaxComponent(direct) > 0.0)
            {
                payload.directSpectrum = direct;
                payload.shadowDirection = lightDirection;
                payload.shadowDistance = lightDistance;
                payload.shadowValid = meshLight ? 2u : 1u;
            }
        }
    }
    payload.bsdfWeight = gaussianAlbedo;
    payload.nextDirection = sampleGaussianIsotropic(rng);
    payload.sampleValid = spectrumIsFinite(gaussianAlbedo)
        && spectrumMaxComponent(gaussianAlbedo) > 0.0 ? 1u : 0u;
    payload.bsdfPdf = GaussianInv4Pi;
    payload.seed = rng.state;
}
