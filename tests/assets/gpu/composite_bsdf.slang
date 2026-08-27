#include "../../../src/libnoorray/Backend/Vulkan/Shaders/Raytracer/CompositeBsdf.slang"
#include "../../../src/libnoorray/Backend/Vulkan/Shaders/Raytracer/RgbToSpectrum.slang"

// Compile-time coverage for the complete tagged-storage BSDF surface. This
// deliberately exercises every lobe conversion and the mixture evaluator so
// Slang cannot discard invalid methods as unreferenced library code.
struct CompositeBsdfTestArgs
{
    uint output;
    uint energyLuts;
    uint spectralTables;
};

struct CompositeBsdfTestRoot
{
    CompositeBsdfTestArgs* arguments;
};

[[vk::push_constant]] ConstantBuffer<CompositeBsdfTestRoot> root;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (any(dispatchThreadId != uint3(0)))
        return;

    const CompositeBsdfTestArgs args = *root.arguments;
    RWStructuredBuffer<float4> output =
        DescriptorHandle<RWStructuredBuffer<float4>>(uint2(args.output, 0));
    ByteAddressBuffer energyLuts =
        DescriptorHandle<ByteAddressBuffer>(uint2(args.energyLuts, 0));
    ByteAddressBuffer spectralTables =
        DescriptorHandle<ByteAddressBuffer>(uint2(args.spectralTables, 0));

    NoorRayShaderData bsdf = makeShaderData(
        float3(0.0, 0.0, 1.0), float3(0.0, 0.0, 1.0),
        normalize(float3(0.2, 0.1, 1.0)));

    DiffuseLobe diffuse = makeDiffuseLobe();
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(0.37);
    diffuse.albedo = rgbAlbedoToSpectrum(spectralTables,
        float3(0.3, 0.5, 0.7), wavelengths);
    diffuse.roughness = 0.3;
    diffuse.burley = true;
    bsdf.addDiffuse(SampledSpectrum(0.6), diffuse, float3(0.0, 0.0, 1.0));

    ConductorLobe conductor = makeConductorLobe();
    conductor.eta = SampledSpectrum(0.2);
    conductor.extinction = SampledSpectrum(3.0);
    conductor.roughness = 0.25;
    bsdf.addConductor(SampledSpectrum(0.2), conductor,
        float3(0.0, 0.0, 1.0));

    DielectricLobe dielectric = makeDielectricLobe();
    dielectric.reflectionTint = SampledSpectrum(1.0);
    dielectric.transmissionTint = SampledSpectrum(0.25);
    dielectric.roughness = 0.15;
    dielectric.ior = 1.5;
    bsdf.addDielectric(SampledSpectrum(0.2), dielectric,
        float3(0.0, 0.0, 1.0));

    const bool prepared = bsdf.prepare(energyLuts);
    const BsdfEvaluation evaluation = bsdf.evaluate(energyLuts,
        normalize(float3(-0.3, 0.2, 1.0)));
    Rng rng;
    rng.state = 0x12345678u;
    const BsdfSample sampled = bsdf.sample(energyLuts, rng);
    output[0] = float4(evaluation.value.xyz,
        prepared && evaluation.isFinite() && sampled.directionIsValid()
            ? evaluation.pdf : -1.0);
}
