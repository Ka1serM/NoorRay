import NoorRHI;
import SceneRecords;
import Common.Math;

bool isBackfacing(ShadingContext context) { return context.normal.w > 0.5; }

float2 selectUv(ShadingContext context, uint uvSet) {
    switch (uvSet) {
    case 1: return context.uv01.zw;
    case 2: return context.uv23.xy;
    case 3: return context.uv23.zw;
    default: return context.uv01.xy;
    }
}

float2 transformUvDirection(MaterialTextureRecord record, float2 direction) {
    return float2(dot(record.row0.xy, direction), -dot(record.row1.xy, direction));
}

bool hasSlot(MaterialRecord material, uint slot) { return (material.textureMask & (1u << slot)) != 0; }

MaterialTextureRecord slotTexture(MaterialRecord material, uint slot) { return material.textures[countbits(material.textureMask & ((1u << slot) - 1u))]; }

//! A texture of a material. Images store v from the top, so v flips.
float4 sampleTexture(MaterialTextureRecord record, ShadingContext context) {
    const float2 uv = selectUv(context, record.uvSet);
    const float2 transformed = float2(dot(record.row0.xy, uv) + record.row0.z, dot(record.row1.xy, uv) + record.row1.z);
    TextureHandle texture;
    texture.value = record.texture;
    SamplerHandle sampler;
    sampler.value = record.sampler;
    return texture.SampleGrad(sampler, float2(transformed.x, 1.0 - transformed.y), transformUvDirection(record, context.uvGradients.xy),
                              transformUvDirection(record, context.uvGradients.zw));
}

static const uint materialTextureSlots = 11;

//! The texels of all slots, one where a slot has no texture. The loop is the only place that samples, because GPU compilers inline each call site of the
//! texture pool switch; on WebGPU eleven inlined copies made a ReSTIR kernel too large for the driver to compile.
struct MaterialTexels {
    float4 slots[materialTextureSlots];
    uint channels;

    __init(MaterialRecord material, ShadingContext context) {
        for (uint slot = 0; slot < materialTextureSlots; ++slot)
            slots[slot] = float4(1.0, 1.0, 1.0, 1.0);
        channels = 0;
        uint mask = material.textureMask;
        [loop]
        for (uint dense = 0; mask != 0; ++dense) {
            const uint slot = firstbitlow(mask);
            mask &= mask - 1u;
            const MaterialTextureRecord record = material.textures[dense];
            slots[slot] = sampleTexture(record, context);
            channels |= (record.channel & 3u) << (2u * slot);
        }
    }

    float4 color(uint slot) { return slots[slot]; }
    float scalar(uint slot, float factor) { return factor * slots[slot][(channels >> (2u * slot)) & 3u]; }
};

//! A tangent-space normal map; the normals in context face the viewer, so the map turns with them on a back face.
float3 mapNormal(ShadingContext context, float4 texel, bool mapped, float strength) {
    const float3 normal = context.normal.xyz;
    if (!mapped)
        return normal;
    const bool backfacing = isBackfacing(context);
    const float3 frameNormal = backfacing ? -normal : normal;
    const float3 tangent = context.tangent.xyz;
    const float3 bitangent = cross(frameNormal, tangent) * context.tangent.w;
    const float3 encoded = texel.xyz * 2.0 - 1.0;
    float3 bent = normalize(encoded.x * tangent + encoded.y * bitangent + encoded.z * frameNormal);
    if (backfacing)
        bent = -bent;
    return strength == 1.0 ? bent : normalize(normal + (bent - normal) * max(strength, 0.0));
}

BsdfInputs evaluateMaterial(SceneRecord scene, ShadingContext context, uint materialIndex) {
    const MaterialRecord material = scene.materials[materialIndex];
    const MaterialTexels texels = MaterialTexels(material, context);
    float3 baseColor = material.baseColorAlpha.xyz * texels.color(textureSlotBaseColor).xyz;
    if ((material.flags & materialFlagVertexColor) != 0)
        baseColor *= context.color.xyz;
    float roughness = texels.scalar(textureSlotRoughness, material.metallicRoughnessIorSpecular.y);
    if ((material.flags & materialFlagGlossiness) != 0)
        roughness = 1.0 - roughness;
    const float4 parameters = material.transmissionRotationCoat;
    const float3 sheenColor = material.sheenColorRoughness.xyz;

    BsdfInputs inputs;
    inputs.baseColorAlpha = float4(baseColor, 1.0);
    inputs.metallicRoughnessIorSpecular = float4(texels.scalar(textureSlotMetallic, material.metallicRoughnessIorSpecular.x), roughness,
                                                 material.metallicRoughnessIorSpecular.z,
                                                 texels.scalar(textureSlotSpecular, material.metallicRoughnessIorSpecular.w));
    inputs.specularTintAnisotropic = material.specularTintAnisotropic;
    inputs.normalRotation =
        float4(mapNormal(context, texels.color(textureSlotNormal), hasSlot(material, textureSlotNormal), material.normalScales.x), parameters.y);
    inputs.tangentTransmission = float4(context.tangent.xyz, texels.scalar(textureSlotTransmission, parameters.x));
    inputs.coat = float4(texels.scalar(textureSlotCoat, parameters.z), texels.scalar(textureSlotCoatRoughness, parameters.w), 1.5, 0.0);
    inputs.coatTint = float4(1.0, 1.0, 1.0, 0.0);
    inputs.coatNormal =
        float4(mapNormal(context, texels.color(textureSlotCoatNormal), hasSlot(material, textureSlotCoatNormal), material.normalScales.y), 0.0);
    inputs.sheen = float4(maxComponent(sheenColor) > 0.0 ? 1.0 : 0.0, material.sheenColorRoughness.w, 0.0, 0.0);
    inputs.sheenTint = float4(sheenColor, 0.0);
    inputs.emission = float4(material.emission.xyz * texels.color(textureSlotEmission).xyz, material.emission.w);
    inputs.thinFilmDispersion = float4(material.thinFilmDispersion.xyz, 0.0);
    return inputs;
}

float evaluateOpacity(SceneRecord scene, ShadingContext context, uint materialIndex) {
    const MaterialRecord material = scene.materials[materialIndex];
    if ((material.flags & materialFlagAlphaTest) == 0)
        return 1.0;
    float alpha = material.baseColorAlpha.w;
    if (hasSlot(material, textureSlotAlpha)) {
        const MaterialTextureRecord record = slotTexture(material, textureSlotAlpha);
        alpha *= sampleTexture(record, context)[record.channel];
    }
    if ((material.flags & materialFlagVertexColor) != 0)
        alpha *= context.color.w;
    if ((material.flags & materialFlagAlphaMask) != 0)
        return alpha < material.thinFilmDispersion.w ? 0.0 : 1.0;
    return saturate(alpha);
}
