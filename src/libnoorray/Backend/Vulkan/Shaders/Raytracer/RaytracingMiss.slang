#include "RaytracingAbi.slang"
#include "AnalyticLights.slang"

[shader("miss")]
void main(inout RayPayload payload)
{
    const FrameParams frame = *root.arguments;
    if (uint(frame.sceneAddress) != 0xFFFFFFFFu)
    {
        ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
            uint2(uint(frame.sceneAddress), 0));
        ByteAddressBuffer spectralTables = DescriptorHandle<ByteAddressBuffer>(
            uint2(frame.spectralTables, 0));
        const SampledWavelengths wavelengths = sampleWavelengthsVisible(
            payload.wavelengthSample);
        const float3 direction = normalize(WorldRayDirection());
        const AnalyticLightHit lightHit = intersectAnalyticLights(lights,
            spectralTables, WorldRayOrigin(), direction, 0.001, 1.0e30,
            wavelengths);
        if (lightHit.valid)
        {
            payload.hit = 4u;
            payload.t = lightHit.distance;
            payload.position = WorldRayOrigin() + direction * lightHit.distance;
            payload.emissionSpectrum = lightHit.radiance;
            payload.lightPdf = lightHit.pdf;
            return;
        }
    }
    payload.hit = 0u;
    payload.radiance = float3(0.0);
    payload.albedo = float3(0.0);
    payload.normal = float3(0.0, 0.0, 1.0);
    payload.position = float3(0.0);
    payload.alpha = 1.0;
    payload.cryptomatte = 0xffffffffu;
    payload.t = 1.0e30;
}
