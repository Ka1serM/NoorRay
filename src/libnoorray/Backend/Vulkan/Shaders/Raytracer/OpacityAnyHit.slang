#include "RaytracingAbi.slang"
#include "SceneSurface.slang"
#include "MaterialSVM.slang"
#include "Sampling.slang"

// Stochastic opacity/transmission is resolved during traversal so primary,
// continuation, analytic-shadow, and mesh-emission rays all see the same
// transparent geometry semantics.
[shader("anyhit")]
void main(inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
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
        InstanceID(), PrimitiveIndex(), attributes.barycentrics);
    const uint recordOffset = surface.materialIndex * 24u;
    if (materials.Load(recordOffset + 20u) != 0u
        && surface.vertexColor.a >= 1.0)
        return;

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
            context, wavelengths, spectralTables, energyLuts,
            dot(WorldRayDirection(), surface.geometricNormal) > 0.0, material);
    if (!valid)
        return;
    const bool emissionTarget = payload.shadowValid == 2u
        && InstanceID() == payload.instance
        && PrimitiveIndex() == payload.primitive;
    const bool visibilityRay = payload.shadowValid != 0u && !emissionTarget;
    const float blockProbability = saturate(material.opacity
        * surface.vertexColor.a * (visibilityRay
            ? (1.0 - material.transmission) : 1.0));
    if (blockProbability >= 1.0)
        return;
    Rng rng;
    rng.state = hashCombine32(payload.seed,
        hashCombine32(InstanceID(), PrimitiveIndex() ^ (payload.depth * 0x9E3779B9u)));
    const bool ignore = rng.nextFloat() >= blockProbability;
    payload.seed = rng.state;
    if (ignore)
        IgnoreHit();
}
