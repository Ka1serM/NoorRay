#include "ByteBuffer.slang"
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

    const Frame frame = *root.arguments;
    ByteBuffer energyLuts = byteBuffer(frame.energyLuts);
    ByteBuffer spectralTables = byteBuffer(frame.spectralTables);
    const RegularSurface surface = loadRegularSurface(frame.scene,
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

    Material* material = frame.materials[surface.materialIndex];
    DescriptorMaterialSvmTextures textures;
    textures.textureTable = material->textures;
    EmissionOnlyMaterialSink sink;
    MaterialSvmResult shading = {};
    const SampledWavelengths wavelengths = sampleWavelengthsVisible(
        payload.wavelengthSample);
    const bool valid = material->bytecodeLength != 0u
        && material->stackSize <= SVM_STACK_SIZE
        && evaluateMaterialSvm(textures, sink, byteBuffer(material->bytecode), 0u,
            material->bytecodeLength, context, wavelengths, spectralTables,
            energyLuts, false, shading);
    if (!valid)
        return;
    payload.emissionSpectrum = rgbIlluminantToSpectrum(spectralTables,
        shading.emission * shading.emissionStrength, wavelengths);
    payload.hit = spectrumMaxComponent(payload.emissionSpectrum) > 0.0 ? 1u : 0u;
}
