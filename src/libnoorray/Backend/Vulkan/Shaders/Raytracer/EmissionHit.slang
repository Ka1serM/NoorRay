#include "RaytracingAbi.slang"
#include "SceneSurface.slang"
#include "MaterialSVM.slang"

// Ray type 1 is used only for a sampled mesh-light visibility query. It has
// its own closest-hit stage so emission evaluation never adds a second SVM
// interpreter (and a second composite BSDF) to the regular surface shader.
[shader("closesthit")]
void main(inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    const uint expectedInstance = payload.instance;
    const uint expectedPrimitive = payload.primitive;
    payload.hit = 0u;
    payload.emissionSpectrum = float4(0.0);
    if (InstanceID() != expectedInstance || PrimitiveIndex() != expectedPrimitive)
        return;

    const FrameParams frame = *root.arguments;
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
    const RegularSurface surface = loadRegularSurface(packedScene,
        expectedInstance, expectedPrimitive, attributes.barycentrics);

    MaterialSvmContext context;
    context.position = surface.position;
    context.geometricNormal = surface.geometricNormal;
    context.interpolatedNormal = surface.shadingNormal;
    context.tangent = surface.tangent;
    context.bitangent = normalize(cross(surface.shadingNormal, surface.tangent))
        * surface.tangentSign;
    context.uv = surface.uv;
    context.vertexColor = surface.vertexColor;
    context.viewDirection = -normalize(WorldRayDirection());
    context.time = 0.0;

    const uint recordOffset = surface.materialIndex * 24u;
    const uint wordOffset = materials.Load(recordOffset);
    const uint wordCount = materials.Load(recordOffset + 4u);
    const uint textureOffset = materials.Load(recordOffset + 8u);
    const uint textureCount = materials.Load(recordOffset + 12u);
    const uint stackSize = materials.Load(recordOffset + 16u);
    DescriptorMaterialSvmTextures textures;
    textures.textureTable = svmWords;
    textures.textureOffset = textureOffset;
    textures.textureCount = textureCount;
    EmissionOnlyMaterialSink sink;
    MaterialSvmResult material = {};
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(
        payload.wavelengthSample);
    const bool valid = wordCount != 0u && stackSize <= SVM_STACK_SIZE
        && evaluateMaterialSvm(textures, sink, svmWords, wordOffset, wordCount,
            context, wavelengths, spectralTables, energyLuts, false, material);
    if (!valid)
        return;
    payload.emissionSpectrum = rgbIlluminantToSpectrum(spectralTables,
        material.emission * material.emissionStrength, wavelengths);
    payload.hit = spectrumMaxComponent(payload.emissionSpectrum) > 0.0 ? 1u : 0u;
}
