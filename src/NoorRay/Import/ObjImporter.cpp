#include "NoorRay/Import/ObjImporter.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Core/Log.h"
#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Materials/TextureCache.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

namespace {

struct CornerKey {
    int vertex = 0;
    int normal = 0;
    int uv = 0;
    bool operator==(const CornerKey&) const = default;
};

struct CornerKeyHash {
    std::size_t operator()(const CornerKey& key) const { return Core::hashValue(key); }
};

Core::float3 toFloat3(const tinyobj::real_t* value) {
    return {static_cast<float>(value[0]), static_cast<float>(value[1]), static_cast<float>(value[2])};
}

class Importer {
public:
    Importer(const std::filesystem::path& path, Scene::Scene& scene) : directory(path.parent_path()), scene(scene), textureCache(scene) {}

    std::optional<Scene::MaterialTexture> resolveTexture(const std::string& name, const tinyobj::texture_option_t& option,
                                                         const Materials::ColorSpace colorSpace) {
        if (name.empty())
            return std::nullopt;
        std::string relative = name;
        std::ranges::replace(relative, '\\', '/');
        const std::filesystem::path file = directory / relative;
        std::expected<Scene::TextureHandle, Core::Error> handle = textureCache.load(file, colorSpace);
        if (!handle) {
            Core::Log::warning("OBJ texture '{}' is skipped: {}", file.string(), handle.error().message);
            return std::nullopt;
        }
        Scene::MaterialTexture texture;
        texture.texture = *handle;
        texture.wrap = option.clamp ? Scene::TextureWrap::Extend : Scene::TextureWrap::Repeat;
        texture.transform.row0 = {static_cast<float>(option.scale[0]), 0.0f, static_cast<float>(option.origin_offset[0])};
        texture.transform.row1 = {0.0f, static_cast<float>(option.scale[1]), static_cast<float>(option.origin_offset[1])};
        return texture;
    }

    Scene::MaterialHandle getMaterial(const std::vector<tinyobj::material_t>& sources, const int index) {
        if (const auto found = materials.find(index); found != materials.end())
            return found->second;
        Scene::Material material;
        material.name = "default";
        if (index >= 0 && static_cast<std::size_t>(index) < sources.size())
            material = convert(sources[index]);
        const Scene::MaterialHandle handle = scene.addMaterial(std::move(material));
        materials.emplace(index, handle);
        return handle;
    }

    Scene::Material convert(const tinyobj::material_t& source) {
        Scene::Material material;
        material.name = source.name;
        material.baseColor = toFloat3(source.diffuse);
        material.texture(Scene::TextureSlot::BaseColor) = resolveTexture(source.diffuse_texname, source.diffuse_texopt, Materials::ColorSpace::Srgb);
        if (material.texture(Scene::TextureSlot::BaseColor) && source.diffuse[0] == 0.0f && source.diffuse[1] == 0.0f && source.diffuse[2] == 0.0f)
            material.baseColor = {1.0f, 1.0f, 1.0f};
        const bool physical = source.roughness > 0.0f || source.metallic > 0.0f || !source.roughness_texname.empty() || !source.metallic_texname.empty();
        if (physical) {
            material.roughness = static_cast<float>(source.roughness);
            material.metallic = static_cast<float>(source.metallic);
            material.texture(Scene::TextureSlot::Roughness) =
                resolveTexture(source.roughness_texname, source.roughness_texopt, Materials::ColorSpace::NonColor);
            material.texture(Scene::TextureSlot::Metallic) = resolveTexture(source.metallic_texname, source.metallic_texopt, Materials::ColorSpace::NonColor);
            material.sheenColor = Core::float3{1.0f, 1.0f, 1.0f} * static_cast<float>(source.sheen);
            material.coat = static_cast<float>(source.clearcoat_thickness);
            material.coatRoughness = static_cast<float>(source.clearcoat_roughness);
            material.anisotropic = static_cast<float>(source.anisotropy);
            material.anisotropicRotation = static_cast<float>(source.anisotropy_rotation);
        } else {
            //! Blender's OBJ importer maps the Phong exponent this way.
            material.roughness = 1.0f - std::sqrt(std::clamp(static_cast<float>(source.shininess), 0.0f, 1000.0f) / 1000.0f);
            if (source.illum == 3)
                material.metallic = Core::luminance(toFloat3(source.specular)) > 0.5f ? 1.0f : 0.0f;
        }
        material.ior = source.ior > 1.0f ? static_cast<float>(source.ior) : 1.5f;
        if (source.illum == 4 || source.illum == 6 || source.illum == 7 || source.illum == 9)
            material.transmission = 1.0f;
        material.alpha = static_cast<float>(source.dissolve);
        material.texture(Scene::TextureSlot::Alpha) = resolveTexture(source.alpha_texname, source.alpha_texopt, Materials::ColorSpace::NonColor);
        if (material.alpha < 1.0f || material.texture(Scene::TextureSlot::Alpha))
            material.alphaMode = Scene::AlphaMode::Blend;
        material.emissionColor = toFloat3(source.emission);
        material.texture(Scene::TextureSlot::Emission) = resolveTexture(source.emissive_texname, source.emissive_texopt, Materials::ColorSpace::Srgb);
        if (material.texture(Scene::TextureSlot::Emission) && Core::maxComponent(material.emissionColor) == 0.0f)
            material.emissionColor = {1.0f, 1.0f, 1.0f};
        material.texture(Scene::TextureSlot::Normal) = resolveTexture(source.normal_texname, source.normal_texopt, Materials::ColorSpace::NonColor);
        if (!source.bump_texname.empty() && !material.texture(Scene::TextureSlot::Normal))
            Core::Log::debug("OBJ material '{}': bump maps are not imported", source.name);
        return material;
    }

    std::expected<void, Core::Error> run(const tinyobj::ObjReader& reader) {
        const tinyobj::attrib_t& attributes = reader.GetAttrib();
        const std::vector<tinyobj::material_t>& sources = reader.GetMaterials();
        const bool hasColors = !attributes.colors.empty() && std::ranges::any_of(attributes.colors, [](const tinyobj::real_t value) { return value != 1.0f; });
        for (const tinyobj::shape_t& shape : reader.GetShapes()) {
            const tinyobj::mesh_t& source = shape.mesh;
            if (source.indices.empty())
                continue;
            MeshData data;
            data.name = shape.name;
            const bool hasNormals = std::ranges::all_of(source.indices, [](const tinyobj::index_t& index) { return index.normal_index >= 0; });
            const bool hasUvs = std::ranges::all_of(source.indices, [](const tinyobj::index_t& index) { return index.texcoord_index >= 0; });
            std::vector<std::uint32_t> order(source.indices.size() / 3);
            for (std::uint32_t triangle = 0; triangle < order.size(); ++triangle)
                order[triangle] = triangle;
            std::ranges::stable_sort(order, [&](const std::uint32_t a, const std::uint32_t b) { return source.material_ids[a] < source.material_ids[b]; });
            std::unordered_map<CornerKey, std::uint32_t, CornerKeyHash> vertices;
            for (const std::uint32_t triangle : order) {
                const Scene::MaterialHandle material = getMaterial(sources, source.material_ids[triangle]);
                const std::uint32_t firstIndex = static_cast<std::uint32_t>(data.indices.size());
                for (std::uint32_t corner = 0; corner < 3; ++corner) {
                    const tinyobj::index_t& index = source.indices[triangle * 3 + corner];
                    const CornerKey key{index.vertex_index, index.normal_index, index.texcoord_index};
                    const auto [found, inserted] = vertices.try_emplace(key, static_cast<std::uint32_t>(data.positions.size()));
                    data.indices.push_back(found->second);
                    if (!inserted)
                        continue;
                    data.positions.push_back(toFloat3(&attributes.vertices[3 * index.vertex_index]));
                    if (hasNormals)
                        data.normals.push_back(Core::normalize(toFloat3(&attributes.normals[3 * index.normal_index])));
                    if (hasUvs)
                        data.uvs[0].push_back({static_cast<float>(attributes.texcoords[2 * index.texcoord_index]),
                                               static_cast<float>(attributes.texcoords[2 * index.texcoord_index + 1])});
                    if (hasColors) {
                        const Core::float3 color = toFloat3(&attributes.colors[3 * index.vertex_index]);
                        data.colors.push_back({color.x, color.y, color.z, 1.0f});
                    }
                }
                data.appendSubset(material, firstIndex, 3);
            }
            std::string name = data.name;
            const Scene::MeshHandle mesh = scene.addMesh(MeshBuilder::build(std::move(data)));
            scene.addNode({std::move(name), {}, {}, scene.addInstance({Core::Transform{}, Core::Transform{}, mesh, Scene::InstanceFlags::None, 0})});
        }
        return {};
    }

private:
    std::filesystem::path directory;
    Scene::Scene& scene;
    Materials::TextureCache textureCache;
    std::unordered_map<int, Scene::MaterialHandle> materials;
};

}

std::expected<void, Core::Error> ObjImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = true;
    config.mtl_search_path = path.parent_path().string();
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string(), config))
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("cannot read OBJ '{}': {}", path.string(), reader.Error()));
    if (!reader.Warning().empty())
        Core::Log::warning("OBJ '{}': {}", path.string(), reader.Warning());
    return Importer(path, scene).run(reader);
}

}
