#include "ByteBuffer.slang"
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
    const Frame frame = *root.arguments;
    ByteBuffer energyLuts = byteBuffer(frame.energyLuts);
    ByteBuffer spectralTables = byteBuffer(frame.spectralTables);
    const RegularSurface surface = loadRegularSurface(frame.scene,
        InstanceID(), PrimitiveIndex(), attributes.barycentrics);
    Material* material = frame.materials[surface.materialIndex];
    if (material->shadowOpaque != 0u && surface.vertexColor.a >= 1.0)
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
    DescriptorMaterialSvmTextures textures;
    textures.textureTable = material->textures;
    EmissionOnlyMaterialSink sink;
    MaterialSvmResult shading = {};
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(
        payload.wavelengthSample);
    const bool valid = material->bytecodeLength != 0u
        && material->stackSize <= SVM_STACK_SIZE
        && evaluateMaterialSvm(textures, sink, byteBuffer(material->bytecode), 0u,
            material->bytecodeLength, context, wavelengths, spectralTables, energyLuts,
            dot(WorldRayDirection(), surface.geometricNormal) > 0.0, shading);
    if (!valid)
        return;
    const bool emissionTarget = payload.shadowValid == 2u
        && InstanceID() == payload.instance
        && PrimitiveIndex() == payload.primitive;
    const bool visibilityRay = payload.shadowValid != 0u && !emissionTarget;
    const float blockProbability = saturate(shading.opacity
        * surface.vertexColor.a * (visibilityRay
            ? (1.0 - shading.transmission) : 1.0));
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
