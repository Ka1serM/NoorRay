#include "ByteBuffer.slang"
#include "RaytracingAbi.slang"
#include "SceneSurface.slang"
#include "CompositeBsdf.slang"
#include "RgbToSpectrum.slang"
#include "AnalyticLights.slang"
#include "Environment.slang"

// Triangle closest-hit owns the complete surface front end. Raygen receives
// only this compact result payload and remains independent of scene layout.
// Materials are not evaluated yet: every surface is a neutral diffuse until
// this renderer runs the MaterialX shaders the realtime renderer compiles.
static const float3 kNeutralAlbedo = float3(0.8);
[shader("closesthit")]
void main(inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    const Frame frame = *root.arguments;
    payload.hit = 1u;
    payload.instance = InstanceID();
    payload.barycentrics = attributes.barycentrics;
    payload.t = RayTCurrent();
    payload.cryptomatte = payload.instance;
    payload.position = WorldRayOrigin() + WorldRayDirection() * payload.t;
    payload.normal = float3(0.0, 0.0, 1.0);
    payload.geometricNormal = float3(0.0, 0.0, 1.0);
    payload.albedo = float3(0.8);
    payload.radiance = float3(0.8);
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

    ByteBuffer energyLuts = byteBuffer(frame.energyLuts);
    ByteBuffer spectralTables = byteBuffer(frame.spectralTables);
    const float3 rayDirection = normalize(WorldRayDirection());
    const RegularSurface surface = loadRegularSurface(frame.scene,
        payload.instance, GeometryIndex(), PrimitiveIndex(), payload.barycentrics);
    payload.primitive = surface.primitiveIndex;

    NoorRayShaderData bsdf = makeShaderData(surface.geometricNormal,
        surface.shadingNormal, -rayDirection);
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(
        payload.wavelengthSample);
    const LightTable lights = makeLightTable(frame);
    if (hasAnalyticLights(frame))
    {
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
    payload.albedo = max(kNeutralAlbedo * surface.vertexColor.rgb, 0.0);
    payload.radiance = payload.albedo;
    payload.normal = surface.shadingNormal;
    payload.geometricNormal = surface.geometricNormal;
    payload.position = surface.position;
    DiffuseLobe lobe = makeDiffuseLobe();
    lobe.albedo = rgbAlbedoToSpectrum(spectralTables, payload.albedo, wavelengths);
    bsdf.addDiffuse(SampledSpectrum(1.0), lobe, float3(0.0));
    if (!bsdf.prepare(energyLuts))
        return;

    Rng bsdfRng;
    bsdfRng.state = payload.seed ^ (0x85EBCA6Bu * (payload.depth + 1u));
    if (hasAnalyticLights(frame)
        && frame.environment != nullptr)
    {
        Environment* environments = frame.environment;
        const Environment environment = environments[0];
        const float finiteWeight = max(lights.finiteWeight, 0.0);
        const float environmentWeight = max(environment.importanceWeight, 0.0);
        const float totalWeight = finiteWeight + environmentWeight;
        const float finiteProbability = totalWeight > 0.0
            ? finiteWeight / totalWeight : 0.0;
        const bool chooseFinite = finiteProbability > 0.0
            && (environmentWeight <= 0.0 || bsdfRng.nextFloat() < finiteProbability);
        SampledSpectrum directRadianceOverPdf = SampledSpectrum(0.0);
        float3 directDirection = float3(0.0);
        float directDistance = 0.0;
        bool directValid = false;
        bool directMesh = false;
        if (chooseFinite)
        {
            const AnalyticLightSample light = sampleAnalyticLight(lights,
                spectralTables, payload.position, wavelengths, bsdfRng);
            directDirection = light.direction;
            directDistance = light.distance;
            directRadianceOverPdf = light.radianceOverSelectionPdf;
            if (light.mesh && light.valid)
            {
                directMesh = true;
                // The emission-only ray type evaluates the sampled material
                // in a separate shader invocation. Here the BSDF is evaluated
                // against a unit illuminant and carries the geometry and
                // selection-PDF factor.
                directRadianceOverPdf = SampledSpectrum(light.emissionMultiplier);
                payload.shadowInstance = light.instanceIndex;
                payload.shadowPrimitive = light.primitiveIndex;
            }
            directRadianceOverPdf /= max(finiteProbability, 1.0e-20);
            directValid = light.valid;
        }
        else if (environmentWeight > 0.0)
        {
            const float environmentProbability = environmentWeight / totalWeight;
            const EnvironmentSample sample = sampleEnvironmentDirection(
                environment, bsdfRng);
            directDirection = sample.direction;
            directDistance = 1.0e16;
            directRadianceOverPdf = rgbIlluminantToSpectrum(spectralTables,
                environmentRadianceRgb(environment, sample.direction, false),
                wavelengths) / max(environmentProbability * sample.pdf, 1.0e-20);
            directValid = sample.pdf > 0.0;
        }
        if (directValid)
        {
            const SampledSpectrum direct = bsdf.evaluateDirect(energyLuts,
                directDirection, directRadianceOverPdf);
            if (spectrumIsFinite(direct) && spectrumMaxComponent(direct) > 0.0)
            {
                payload.directSpectrum = direct;
                payload.shadowDirection = directDirection;
                payload.shadowDistance = directDistance;
                payload.shadowValid = directMesh ? 2u : 1u;
            }
        }
    }
    const BsdfSample sample = bsdf.sample(energyLuts, bsdfRng);
    payload.seed = bsdfRng.state;
    if (sample.directionIsValid()
        && spectrumMaxAbsComponent(sample.weight) > 0.0)
    {
        payload.bsdfWeight = sample.weight;
        payload.nextDirection = sample.direction;
        payload.sampleValid = 1u;
        payload.bsdfPdf = sample.pdf;
    }
}
