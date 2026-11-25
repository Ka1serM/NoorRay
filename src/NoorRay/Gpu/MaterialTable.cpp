#include "NoorRay/Gpu/MaterialTable.h"

#include <algorithm>
#include <format>

#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Gpu/TextureTable.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/GraphShader.h"

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

//! The value of an output of a graph material when it follows from the constants and parameter values alone.
std::optional<Core::float4> uniformOutput(const Scene::Material& material, const Materials::GraphOutput output) {
    return material.graph->evaluateUniform(material.graph->getOutput(output), material.graphParameters);
}

//! The flags of a graph material: the graph replaces the inputs, so only the flags of the outputs that can change stay.
std::uint32_t getGraphFlags(const Scene::Material& material) {
    std::uint32_t flags = Shaders::materialFlagGraph;
    const std::optional<Core::float4> transmission = uniformOutput(material, Materials::GraphOutput::TangentTransmission);
    if (!transmission || transmission->w > 0.0f)
        flags |= Shaders::materialFlagTransmissive;
    const std::optional<Core::float4> transparency = uniformOutput(material, Materials::GraphOutput::Transparency);
    if (!transparency || transparency->x != 0.0f)
        flags |= Shaders::materialFlagAlphaTest;
    const std::optional<Core::float4> emission = uniformOutput(material, Materials::GraphOutput::Emission);
    if (!emission || (emission->w > 0.0f && std::max({emission->x, emission->y, emission->z}) > 0.0f))
        flags |= Shaders::materialFlagEmissive;
    const std::optional<Core::float4> thinFilm = uniformOutput(material, Materials::GraphOutput::ThinFilmDispersion);
    if (!thinFilm || thinFilm->z > 0.0f)
        flags |= Shaders::materialFlagDispersive;
    const std::optional<Core::float4> subsurface = uniformOutput(material, Materials::GraphOutput::Subsurface);
    if (!subsurface || subsurface->w > 0.0f)
        flags |= Shaders::materialFlagSubsurface;
    if (material.bumpsRestNormal)
        flags |= Shaders::materialFlagRestNormal;
    if (material.bumpsSurface)
        flags |= Shaders::materialFlagBumpsSurface;
    return flags;
}

//! The faces of the triangle lights of a material that light sampling skips, as the light flags of a triangle that keeps the winding of the mesh.
std::uint32_t skippedFaces(const Scene::EmissionSampling sampling) {
    if (sampling == Scene::EmissionSampling::Front)
        return Shaders::lightFlagNoBack;
    if (sampling == Scene::EmissionSampling::Back)
        return Shaders::lightFlagNoFront;
    return 0;
}

//! The radiance estimate of a graph material for the light sampling: the constant emission, else the fixed inputs, else one.
Core::float3 getGraphEmission(const Scene::Material& material) {
    if (const std::optional<Core::float4> emission = uniformOutput(material, Materials::GraphOutput::Emission))
        return Core::float3{emission->x, emission->y, emission->z} * std::max(emission->w, 0.0f);
    if (material.isEmissive())
        return material.emissionColor * material.emissionStrength;
    return {1.0f, 1.0f, 1.0f};
}

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

std::expected<void, Core::Error> MaterialTable::updateGraphCode(std::vector<std::shared_ptr<const Materials::Graph>> graphs, const std::string_view name,
                                                                GraphCodes& codes) {
    std::vector<std::shared_ptr<const Materials::GraphCode>> written;
    written.reserve(graphs.size());
    for (const std::shared_ptr<const Materials::Graph>& graph : graphs) {
        const auto same = [&](const auto& other) { return other == graph || other->hasSameProgram(*graph); };
        if (const auto found = std::ranges::find_if(codes.graphs, same); found != codes.graphs.end()) {
            written.push_back(codes.codes[static_cast<std::size_t>(found - codes.graphs.begin())]);
            continue;
        }
        std::expected<Materials::GraphCode, Core::Error> code = Materials::writeGraphCode(*graph, name);
        if (!code)
            return std::unexpected(std::move(code.error()));
        written.push_back(std::make_shared<const Materials::GraphCode>(std::move(*code)));
    }
    codes.codes = std::move(written);
    codes.graphs = std::move(graphs);
    return {};
}

std::expected<void, Core::Error> MaterialTable::update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures) {
    const std::span<const Scene::Material> materials = scene.getMaterials().getValues();
    std::vector<Shaders::MaterialRecord> recordValues;
    std::vector<Shaders::MaterialTextureRecord> textureValues;
    std::vector<std::uint32_t> firstTextures;
    std::vector<NoorRHI::float4> parameterValues;
    std::vector<std::shared_ptr<const Materials::Graph>> graphs;
    std::vector<std::shared_ptr<const Materials::Graph>> alphaGraphs;
    recordValues.reserve(materials.size());
    firstTextures.reserve(materials.size());
    infos.assign(materials.size(), {});
    flagUnion = 0;
    for (std::size_t i = 0; i < materials.size(); ++i) {
        const Scene::Material& material = materials[i];
        firstTextures.push_back(static_cast<std::uint32_t>(textureValues.size()));
        if (material.graph) {
            const Materials::Graph& graph = *material.graph;
            if (material.graphParameters.size() != graph.getParameterSlots() || material.graphTextures.size() != graph.getTextureCount())
                return Core::fail(Core::ErrorCode::InvalidArgument,
                                  std::format("the material {} has {} parameter slots and {} textures, but its graph has {} and {}", material.name,
                                              material.graphParameters.size(), material.graphTextures.size(), graph.getParameterSlots(),
                                              graph.getTextureCount()));
            const auto found = std::ranges::find_if(graphs, [&](const auto& other) { return other == material.graph || other->hasSameProgram(graph); });
            const auto index = static_cast<std::uint32_t>(found - graphs.begin());
            if (found == graphs.end())
                graphs.push_back(material.graph);
            const std::uint32_t flags = getGraphFlags(material);
            Shaders::MaterialRecord record = makeRecord(material, flags);
            const auto alphaFound = std::ranges::find(alphaGraphs, graphs[index]);
            const auto alphaIndex = static_cast<std::uint32_t>(alphaFound - alphaGraphs.begin());
            const bool alpha = (flags & Shaders::materialFlagAlphaTest) != 0;
            if (alpha && alphaFound == alphaGraphs.end())
                alphaGraphs.push_back(graphs[index]);
            record.graph = {index, static_cast<std::uint32_t>(parameterValues.size()), alpha ? alphaIndex : Shaders::invalidIndex, 0};
            for (const Core::float4& value : material.graphParameters)
                parameterValues.push_back({value.x, value.y, value.z, value.w});
            for (const Scene::MaterialTexture& texture : material.graphTextures)
                textureValues.push_back(makeTextureRecord(texture, textures));
            recordValues.push_back(record);
            const bool emissive = (flags & Shaders::materialFlagEmissive) != 0;
            const bool varies = emissive && !uniformOutput(material, Materials::GraphOutput::Emission);
            const bool sampled = material.emissionSampling != Scene::EmissionSampling::Disabled
                                 && !graph.reads(Materials::GraphOutput::Emission, Materials::GraphInput::Incoming)
                                 && !graph.reads(Materials::GraphOutput::Emission, Materials::GraphInput::Distance);
            infos[i] = {flags, emissive ? getGraphEmission(material) : Core::float3{}, varies ? index : Shaders::invalidIndex, sampled,
                        skippedFaces(material.emissionSampling)};
            flagUnion |= flags;
            continue;
        }
        const std::uint32_t flags = getMaterialFlags(material);
        Shaders::MaterialRecord record = makeRecord(material, flags);
        for (std::uint32_t slot = 0; slot < Scene::textureSlotCount; ++slot) {
            const std::optional<Scene::MaterialTexture>& texture = material.textures[slot];
            if (!texture)
                continue;
            record.textureMask |= 1u << slot;
            textureValues.push_back(makeTextureRecord(*texture, textures));
        }
        recordValues.push_back(record);
        infos[i] = {flags, material.isEmissive() ? material.emissionColor * material.emissionStrength : Core::float3{}, Shaders::invalidIndex,
                    material.emissionSampling != Scene::EmissionSampling::Disabled, skippedFaces(material.emissionSampling)};
        flagUnion |= flags;
    }
    if (auto written = updateGraphCode(std::move(graphs), Materials::graphFunctionName, graphCodes); !written)
        return written;
    if (auto written = updateGraphCode(std::move(alphaGraphs), Materials::alphaGraphFunctionName, alphaGraphCodes); !written)
        return written;
    if (auto uploaded = uploadValues<NoorRHI::float4>(device, graphParameters, parameterValues); !uploaded)
        return uploaded;
    recordValues.push_back(makeRecord(Scene::Material{}, 0));
    firstTextures.push_back(static_cast<std::uint32_t>(textureValues.size()));
    if (auto uploaded = uploadValues<Shaders::MaterialTextureRecord>(device, textureRecords, textureValues); !uploaded)
        return uploaded;
    for (std::size_t i = 0; i < recordValues.size(); ++i)
        recordValues[i].textures = textureRecords.getPtr() + firstTextures[i];
    return uploadValues<Shaders::MaterialRecord>(device, records, recordValues);
}

}
