#include "RaytracingAbi.slang"
#include "SceneSurface.slang"
#include "MaterialSVM.slang"
#include "AnalyticLights.slang"
#include "Environment.slang"

// Triangle closest-hit owns the complete surface/material front end. Raygen
// receives only this compact result payload and remains independent of scene
// layout and MaterialX bytecode size.
[shader("closesthit")]
void main(inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    const FrameParams frame = *root.arguments;
    payload.hit = 1u;
    payload.instance = InstanceID();
    payload.primitive = PrimitiveIndex();
    payload.barycentrics = attributes.barycentrics;
    payload.t = RayTCurrent();
    payload.cryptomatte = payload.instance;
    payload.position = WorldRayOrigin() + WorldRayDirection() * payload.t;
    payload.normal = float3(0.0, 0.0, 1.0);
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

    if (frame.sceneBuffer == 0xFFFFFFFFu)
        return;

    ByteAddressBuffer packedScene = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.sceneBuffer, 0));
    ByteAddressBuffer materials = DescriptorHandle<ByteAddressBuffer>(
        uint2(uint(frame.materialsAddress), 0));
    ByteAddressBuffer svmWords = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.svmWords, 0));
    ByteAddressBuffer energyLuts = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.energyLutBuffer, 0));
    ByteAddressBuffer spectralTables = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.spectralTables, 0));
    const float3 rayDirection = normalize(WorldRayDirection());
    const RegularSurface surface = loadRegularSurface(packedScene,
        payload.instance, payload.primitive, payload.barycentrics);

    MaterialSvmContext context;
    context.position = surface.position;
    context.geometricNormal = surface.geometricNormal;
    context.interpolatedNormal = surface.shadingNormal;
    context.tangent = surface.tangent;
    context.bitangent = normalize(cross(surface.shadingNormal,
        surface.tangent)) * surface.tangentSign;
    context.uv = surface.uv;
    context.vertexColor = surface.vertexColor;
    context.viewDirection = -rayDirection;
    context.time = 0.0;

    RaytracerMaterialSink sink;
    sink.bsdf = makeShaderData(context.geometricNormal,
        context.interpolatedNormal, context.viewDirection);
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(
        payload.wavelengthSample);
    if (uint(frame.sceneAddress) != 0xFFFFFFFFu)
    {
        ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
            uint2(uint(frame.sceneAddress), 0));
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
    const uint materialRecordOffset = surface.materialIndex * 24u;
    const uint wordOffset = materials.Load(materialRecordOffset);
    const uint wordCount = materials.Load(materialRecordOffset + 4u);
    const uint textureOffset = materials.Load(materialRecordOffset + 8u);
    const uint textureCount = materials.Load(materialRecordOffset + 12u);
    const uint stackSize = materials.Load(materialRecordOffset + 16u);
    DescriptorMaterialSvmTextures textures;
    textures.textureTable = svmWords;
    textures.textureOffset = textureOffset;
    textures.textureCount = textureCount;
    MaterialSvmResult material = {};
    const bool valid = wordCount != 0u && stackSize <= SVM_STACK_SIZE
        && evaluateMaterialSvm(textures, sink, svmWords, wordOffset,
            wordCount, context, wavelengths, spectralTables, energyLuts,
            dot(rayDirection, surface.geometricNormal) > 0.0, material);

    const float3 baseColor = valid ? material.albedo : float3(0.8);
    payload.albedo = max(baseColor * surface.vertexColor.rgb, 0.0);
    payload.radiance = payload.albedo + (valid
        ? material.emission * material.emissionStrength : float3(0.0));
    // Opacity is resolved stochastically in OpacityAnyHit, so a hit that
    // reaches closest hit is fully covered. Publishing the opacity again here
    // composited it twice and made the transparent-background alpha too low.
    payload.normal = normalize(valid && (material.flags & SVM_HAS_SHADING_NORMAL) != 0u
        ? material.shadingNormal : surface.shadingNormal);
    payload.position = surface.position;
    if (!valid)
        return;

    payload.emissionSpectrum = rgbIlluminantToSpectrum(spectralTables,
        material.emission * material.emissionStrength, wavelengths);
    Rng bsdfRng;
    bsdfRng.state = payload.seed ^ (0x85EBCA6Bu * (payload.depth + 1u));
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
            const SampledSpectrum direct = sink.bsdf.evaluateDirect(energyLuts,
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
    const BsdfSample sample = sink.bsdf.sample(energyLuts, bsdfRng);
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
