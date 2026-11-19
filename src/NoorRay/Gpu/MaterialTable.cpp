#include "NoorRay/Gpu/MaterialTable.h"

#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/TextureTable.h"

namespace NoorRay::Gpu {

namespace {

NoorRHI::float4 widen(const Core::float3 value, const float w) {
    return {value.x, value.y, value.z, w};
}

std::uint32_t getMaterialFlags(const Scene::Material& material) {
    std::uint32_t flags = 0;
    if (material.alphaMode != Scene::AlphaMode::Opaque)
        flags |= Shaders::materialFlagAlphaTest;
    if (material.alphaMode == Scene::AlphaMode::Mask)
        flags |= Shaders::materialFlagAlphaMask;
    if (material.isEmissive())
        flags |= Shaders::materialFlagEmissive;
    if (material.transmission > 0.0f || material.texture(Scene::TextureSlot::Transmission))
        flags |= Shaders::materialFlagTransmissive;
    if (material.dispersion > 0.0f && (flags & Shaders::materialFlagTransmissive) != 0)
        flags |= Shaders::materialFlagDispersive;
    if (material.useVertexColor)
        flags |= Shaders::materialFlagVertexColor;
    if (material.roughnessIsGlossiness)
        flags |= Shaders::materialFlagGlossiness;
    return flags;
}

Shaders::MaterialRecord makeRecord(const Scene::Material& material, const std::uint32_t flags) {
    Shaders::MaterialRecord record{};
    record.baseColorAlpha = widen(material.baseColor, material.alpha);
    record.metallicRoughnessIorSpecular = {material.metallic, material.roughness, material.ior, material.specularIorLevel};
    record.specularTintAnisotropic = widen(material.specularTint, material.anisotropic);
    record.transmissionRotationCoat = {material.transmission, material.anisotropicRotation, material.coat, material.coatRoughness};
    record.sheenColorRoughness = widen(material.sheenColor, material.sheenRoughness);
    record.emission = widen(material.emissionColor, material.emissionStrength);
    record.thinFilmDispersion = {material.thinFilmThickness, material.thinFilmIor, material.dispersion, material.alphaCutoff};
    record.normalScales = {material.normalScale, material.coatNormalScale, 0.0f, 0.0f};
    record.flags = flags;
    return record;
}

Shaders::MaterialTextureRecord makeTextureRecord(const Scene::MaterialTexture& texture, const TextureTable& textures) {
    return {
        .texture = textures.getTexture(texture.texture).value,
        .sampler = textures.getSampler(texture.wrap, texture.filter).value,
        .uvSet = texture.uvSet,
        .channel = texture.channel,
        .row0 = widen(texture.transform.row0, 0.0f),
        .row1 = widen(texture.transform.row1, 0.0f),
    };
}

}

std::expected<void, Core::Error> MaterialTable::update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures) {
    const std::span<const Scene::Material> materials = scene.getMaterials().getValues();
    std::vector<Shaders::MaterialRecord> recordValues;
    std::vector<Shaders::MaterialTextureRecord> textureValues;
    std::vector<std::uint32_t> firstTextures;
    recordValues.reserve(materials.size());
    firstTextures.reserve(materials.size());
    infos.assign(materials.size(), {});
    flagUnion = 0;
    for (std::size_t i = 0; i < materials.size(); ++i) {
        const Scene::Material& material = materials[i];
        const std::uint32_t flags = getMaterialFlags(material);
        Shaders::MaterialRecord record = makeRecord(material, flags);
        firstTextures.push_back(static_cast<std::uint32_t>(textureValues.size()));
        for (std::uint32_t slot = 0; slot < Scene::textureSlotCount; ++slot) {
            const std::optional<Scene::MaterialTexture>& texture = material.textures[slot];
            if (!texture)
                continue;
            record.textureMask |= 1u << slot;
            textureValues.push_back(makeTextureRecord(*texture, textures));
        }
        recordValues.push_back(record);
        infos[i] = {flags, material.isEmissive() ? material.emissionColor * material.emissionStrength : Core::float3{}};
        flagUnion |= flags;
    }
    if (auto uploaded = uploadValues<Shaders::MaterialTextureRecord>(device, textureRecords, textureValues); !uploaded)
        return uploaded;
    for (std::size_t i = 0; i < recordValues.size(); ++i)
        recordValues[i].textures = textureRecords.getPtr() + firstTextures[i];
    return uploadValues<Shaders::MaterialRecord>(device, records, recordValues);
}

}
