import NoorRHI;
import SceneRecords;
import Common.Math;

//! The tangent of a material without one; packOctahedral never writes it, because it clamps to -32767.
static const uint closureNoTangent = 0x80008000u;
static const uint closureModeShift = 30u;

bool isBackfacing(ShadingContext context) { return context.normal.w > 0.5; }

uint packUnorm2x16(float2 value) {
    const uint2 quantized = uint2(round(saturate(value) * 65535.0));
    return quantized.x | (quantized.y << 16);
}

float2 unpackUnorm2x16(uint packed) { return float2(packed & 0xffffu, packed >> 16) / 65535.0; }

uint packUnorm3x10(float3 value) {
    const uint3 quantized = uint3(round(saturate(value) * 1023.0));
    return quantized.x | (quantized.y << 10) | (quantized.z << 20);
}

float3 unpackUnorm3x10(uint packed) { return float3(packed & 0x3ffu, (packed >> 10) & 0x3ffu, (packed >> 20) & 0x3ffu) / 1023.0; }

uint packHalfPair(float low, float high) { return packHalf2(float2(low, high)); }

//! The BSDF inputs in 96 bytes: the values that the BSDF clamps to [0, 1] as unorm, the rest as half; opacity is the alpha of the hit.
PackedClosure packClosure(BsdfInputs inputs, float opacity) {
    const float3 tangent = inputs.tangentTransmission.xyz;
    PackedClosure packed;
    packed.base = uint4(packUnorm2x16(inputs.baseColorAlpha.xy), packUnorm2x16(float2(inputs.baseColorAlpha.z, opacity)),
                        packUnorm2x16(inputs.metallicRoughnessIorSpecular.xy), packHalf2(inputs.metallicRoughnessIorSpecular.zw));
    packed.frame = uint4(packUnorm3x10(inputs.specularTintAnisotropic.xyz) | (uint(inputs.closure.w) << closureModeShift),
                         f32tof16(inputs.specularTintAnisotropic.w) | (uint(round(frac(inputs.normalRotation.w) * 65535.0)) << 16),
                         packOctahedral(safeNormalize(inputs.normalRotation.xyz)), dot(tangent, tangent) > 0.0 ? packOctahedral(tangent) : closureNoTangent);
    packed.coat = uint4(packUnorm2x16(float2(inputs.tangentTransmission.w, inputs.coat.y)), packOctahedral(safeNormalize(inputs.coatNormal.xyz)),
                        packUnorm2x16(float2(inputs.coat.x, 0.0)) | (f32tof16(inputs.coat.z) << 16), packUnorm3x10(inputs.coatTint.xyz));
    packed.sheen = uint4(f32tof16(max(inputs.sheen.x, 0.0)) | (packUnorm2x16(float2(inputs.sheen.y, 0.0)) << 16), packUnorm3x10(inputs.sheenTint.xyz),
                         packHalf2(min(max(inputs.emission.xy, 0.0), 65504.0)), asuint(inputs.emission.w));
    packed.film = uint4(packHalf2(inputs.thinFilmDispersion.xy), packHalfPair(inputs.thinFilmDispersion.z, min(max(inputs.emission.z, 0.0), 65504.0)),
                        packRgb9e5(inputs.closure.xyz), packOctahedral(safeNormalize(inputs.smoothNormal.xyz)));
    packed.subsurface = uint4(packRgb9e5(max(inputs.subsurface.xyz, 0.0)), packUnorm2x16(float2(inputs.subsurface.w, inputs.sheen.z)), 0u, 0u);
    return packed;
}

BsdfInputs unpackClosure(PackedClosure packed) {
    const float2 baseColor = unpackUnorm2x16(packed.base.x);
    const float2 blue = unpackUnorm2x16(packed.base.y);
    const float2 transmissionCoat = unpackUnorm2x16(packed.coat.x);
    const float2 film = unpackHalf2(packed.film.x);
    const float2 dispersionEmission = unpackHalf2(packed.film.y);
    BsdfInputs inputs;
    inputs.baseColorAlpha = float4(baseColor, blue.x, blue.y);
    inputs.metallicRoughnessIorSpecular = float4(unpackUnorm2x16(packed.base.z), unpackHalf2(packed.base.w));
    inputs.specularTintAnisotropic = float4(unpackUnorm3x10(packed.frame.x), f16tof32(packed.frame.y & 0xffffu));
    inputs.normalRotation = float4(unpackOctahedral(packed.frame.z), float(packed.frame.y >> 16) / 65535.0);
    const float3 tangent = packed.frame.w == closureNoTangent ? float3(0.0, 0.0, 0.0) : unpackOctahedral(packed.frame.w);
    inputs.tangentTransmission = float4(tangent, transmissionCoat.x);
    inputs.coat = float4(float(packed.coat.z & 0xffffu) / 65535.0, transmissionCoat.y, f16tof32(packed.coat.z >> 16), 0.0);
    inputs.coatTint = float4(unpackUnorm3x10(packed.coat.w), 0.0);
    inputs.coatNormal = float4(unpackOctahedral(packed.coat.y), 0.0);
    inputs.sheen = float4(f16tof32(packed.sheen.x & 0xffffu), float(packed.sheen.x >> 16) / 65535.0, unpackUnorm2x16(packed.subsurface.y).y, 0.0);
    inputs.sheenTint = float4(unpackUnorm3x10(packed.sheen.y), 0.0);
    inputs.emission = float4(unpackHalf2(packed.sheen.z), dispersionEmission.y, asfloat(packed.sheen.w));
    inputs.thinFilmDispersion = float4(film, dispersionEmission.x, 0.0);
    inputs.closure = float4(unpackRgb9e5(packed.film.z), float(packed.frame.x >> closureModeShift));
    inputs.smoothNormal = float4(unpackOctahedral(packed.film.w), 0.0);
    inputs.subsurface = float4(unpackRgb9e5(packed.subsurface.x), unpackUnorm2x16(packed.subsurface.y).x);
    return inputs;
}

float closureOpacity(PackedClosure packed) { return float(packed.base.y >> 16) / 65535.0; }

//! Whether a hit is a holdout, by its closure or by its object.
bool isHoldout(PackedClosure packed, uint instanceFlags) {
    return packed.frame.x >> closureModeShift == closureHoldout || (instanceFlags & instanceFlagHoldout) != 0;
}

//! The specular roughness after the textures and the graph, without the rest of the closure.
float closureRoughness(PackedClosure packed) { return unpackUnorm2x16(packed.base.z).y; }
