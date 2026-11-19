#include "NoorRay/Format/SceneWriter.h"

#include <format>
#include <fstream>
#include <ranges>

#include "NoorRay/Format/SceneConversion.h"

namespace NoorRay::Format {

namespace {

Schema::TextureFormat toSchemaFormat(const NoorRHI::ImageFormat format) {
    switch (format) {
    case NoorRHI::ImageFormat::Rgba8Srgb:
        return Schema::TextureFormat::Rgba8Srgb;
    case NoorRHI::ImageFormat::Rgba16Float:
        return Schema::TextureFormat::Rgba16Float;
    case NoorRHI::ImageFormat::Bc4Unorm:
        return Schema::TextureFormat::Bc4Unorm;
    case NoorRHI::ImageFormat::Bc5Unorm:
        return Schema::TextureFormat::Bc5Unorm;
    case NoorRHI::ImageFormat::Bc6hUfloat:
        return Schema::TextureFormat::Bc6hUfloat;
    case NoorRHI::ImageFormat::Bc7Unorm:
        return Schema::TextureFormat::Bc7Unorm;
    case NoorRHI::ImageFormat::Bc7Srgb:
        return Schema::TextureFormat::Bc7Srgb;
    default:
        return Schema::TextureFormat::Rgba8Unorm;
    }
}

template <class T>
std::uint32_t indexOf(const Core::SlotMap<T>& map, const Core::Handle<T> handle) {
    return map.denseIndex(handle);
}

static_assert(std::variant_size_v<Scene::NodeObject> == static_cast<std::size_t>(Schema::NodeObjectKind::Camera) + 1);

std::uint32_t objectIndex(const Scene::Scene& scene, const Scene::NodeObject& object) {
    if (const auto* handle = std::get_if<Scene::InstanceHandle>(&object))
        return indexOf(scene.getInstances(), *handle);
    if (const auto* handle = std::get_if<Scene::LightHandle>(&object))
        return indexOf(scene.getLights(), *handle);
    if (const auto* handle = std::get_if<Scene::GaussianCloudHandle>(&object))
        return indexOf(scene.getGaussianClouds(), *handle);
    if (const auto* handle = std::get_if<Scene::VolumeHandle>(&object))
        return indexOf(scene.getVolumes(), *handle);
    if (const auto* handle = std::get_if<Scene::CameraHandle>(&object))
        return indexOf(scene.getCameras(), *handle);
    return 0;
}

//! Parents come before their children and siblings keep their order, so the reader rebuilds the tree in one pass.
std::vector<flatbuffers::Offset<Schema::Node>> writeNodes(flatbuffers::FlatBufferBuilder& builder, const Scene::Scene& scene) {
    std::vector<flatbuffers::Offset<Schema::Node>> nodes;
    std::vector<std::pair<Scene::NodeHandle, std::uint32_t>> pending;
    for (const Scene::NodeHandle root : std::views::reverse(scene.getRootNodes()))
        pending.emplace_back(root, Scene::NodeHandle::invalidIndex);
    while (!pending.empty()) {
        const auto [handle, parent] = pending.back();
        pending.pop_back();
        const Scene::Node& node = *scene.find(handle);
        const Schema::Transform transform = toSchema(node.localTransform);
        const auto kind = static_cast<Schema::NodeObjectKind>(node.object.index());
        const auto index = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(Schema::CreateNode(builder, builder.CreateString(node.name), &transform, parent, kind, objectIndex(scene, node.object)));
        for (const Scene::NodeHandle child : std::views::reverse(node.children))
            pending.emplace_back(child, index);
    }
    return nodes;
}

std::span<const Schema::Float3> asSchema(const std::vector<Core::float3>& values) {
    static_assert(sizeof(Schema::Float3) == sizeof(Core::float3));
    return {reinterpret_cast<const Schema::Float3*>(values.data()), values.size()};
}

}

std::vector<std::byte> SceneWriter::write(const Scene::Scene& scene) {
    flatbuffers::FlatBufferBuilder builder(1 << 20);

    std::vector<flatbuffers::Offset<Schema::Texture>> textures;
    for (const Scene::Texture& texture : scene.getTextures().getValues()) {
        builder.ForceVectorAlignment(texture.data.size(), 1, 16);
        const auto data = builder.CreateVector(reinterpret_cast<const std::uint8_t*>(texture.data.data()), texture.data.size());
        std::vector<Schema::TextureLevel> levels;
        for (const Scene::TextureLevel& level : texture.levels)
            levels.emplace_back(level.offset, level.size, level.width, level.height);
        const auto levelVector = builder.CreateVectorOfStructs(levels);
        textures.push_back(Schema::CreateTexture(builder, builder.CreateString(texture.name), toSchemaFormat(texture.format), texture.width, texture.height,
                                                 levelVector, data));
    }

    std::vector<flatbuffers::Offset<Schema::Material>> materials;
    for (const Scene::Material& material : scene.getMaterials().getValues()) {
        const Schema::MaterialParameters parameters(
            toSchema(material.baseColor), material.alpha, static_cast<Schema::AlphaMode>(material.alphaMode), material.useVertexColor,
            material.roughnessIsGlossiness, material.alphaCutoff, material.metallic, material.roughness, material.normalScale, material.ior,
            material.specularIorLevel, toSchema(material.specularTint), material.transmission, material.anisotropic, material.anisotropicRotation,
            material.coat, material.coatRoughness, material.coatNormalScale, toSchema(material.sheenColor), material.sheenRoughness,
            toSchema(material.emissionColor), material.emissionStrength, material.thinFilmThickness, material.thinFilmIor, material.dispersion);
        std::vector<Schema::MaterialTexture> slots;
        for (std::uint32_t slot = 0; slot < Scene::textureSlotCount; ++slot) {
            const std::optional<Scene::MaterialTexture>& texture = material.textures[slot];
            if (!texture)
                continue;
            slots.emplace_back(indexOf(scene.getTextures(), texture->texture), texture->uvSet, texture->channel, toSchema(texture->transform.row0),
                               toSchema(texture->transform.row1), static_cast<Schema::TextureSlot>(slot), static_cast<Schema::TextureWrap>(texture->wrap),
                               static_cast<Schema::TextureFilter>(texture->filter));
        }
        materials.push_back(Schema::CreateMaterial(builder, builder.CreateString(material.name), &parameters, builder.CreateVectorOfStructs(slots)));
    }

    std::vector<flatbuffers::Offset<Schema::Mesh>> meshes;
    for (const Scene::Mesh& mesh : scene.getMeshes().getValues()) {
        std::vector<flatbuffers::Offset<Schema::UvSet>> uvs;
        for (std::uint32_t set = 0; set < mesh.getUvSetCount(); ++set)
            uvs.push_back(Schema::CreateUvSet(builder, builder.CreateVector(mesh.uvs[set])));
        std::vector<Schema::MeshSubset> subsets;
        for (const Scene::MeshSubset& subset : mesh.subsets)
            subsets.emplace_back(subset.firstIndex, subset.indexCount, indexOf(scene.getMaterials(), subset.material));
        const std::span<const Schema::Float3> positions = asSchema(mesh.positions);
        const Schema::Bounds bounds = toSchema(mesh.bounds);
        meshes.push_back(Schema::CreateMesh(builder, builder.CreateString(mesh.name), builder.CreateVectorOfStructs(positions.data(), positions.size()),
                                            builder.CreateVector(mesh.normals), builder.CreateVector(mesh.tangents), builder.CreateVector(uvs),
                                            builder.CreateVector(mesh.colors), builder.CreateVector(mesh.indices), builder.CreateVectorOfStructs(subsets),
                                            &bounds));
    }

    std::vector<Schema::Instance> instances;
    for (const Scene::Instance& instance : scene.getInstances().getValues())
        instances.emplace_back(toSchema(instance.transform), indexOf(scene.getMeshes(), instance.mesh), static_cast<Schema::InstanceFlags>(instance.flags),
                               instance.id, indexOf(scene.getMaterials(), instance.material));

    std::vector<flatbuffers::Offset<Schema::GaussianCloud>> clouds;
    for (const Scene::GaussianCloud& cloud : scene.getGaussianClouds().getValues()) {
        const Schema::Transform transform = toSchema(cloud.transform);
        const std::span<const Schema::Float3> positions = asSchema(cloud.positions);
        const std::span<const Schema::Float3> scales = asSchema(cloud.scales);
        const auto rotations = builder.CreateVector(reinterpret_cast<const float*>(cloud.rotations.data()), cloud.rotations.size() * 4);
        clouds.push_back(Schema::CreateGaussianCloud(
            builder, builder.CreateString(cloud.name), &transform, builder.CreateVectorOfStructs(positions.data(), positions.size()),
            builder.CreateVectorOfStructs(scales.data(), scales.size()), rotations, builder.CreateVector(cloud.opacities),
            static_cast<std::uint8_t>(cloud.shDegree), builder.CreateVector(cloud.sh)));
    }

    std::vector<flatbuffers::Offset<Schema::Volume>> volumes;
    for (const Scene::Volume& volume : scene.getVolumes().getValues()) {
        const Schema::Transform transform = toSchema(volume.transform);
        const Schema::Bounds bounds = toSchema(volume.bounds);
        const Schema::Float3 absorption = toSchema(volume.absorption);
        const Schema::Float3 scattering = toSchema(volume.scattering);
        const Schema::Float3 emission = toSchema(volume.emission);
        const std::vector<std::uint32_t> resolution{volume.resolution.x, volume.resolution.y, volume.resolution.z};
        volumes.push_back(Schema::CreateVolume(builder, builder.CreateString(volume.name), static_cast<Schema::VolumeKind>(volume.kind), &transform, &bounds,
                                               &absorption, &scattering, &emission, volume.anisotropy, volume.densityScale, builder.CreateVector(resolution),
                                               builder.CreateVector(volume.density)));
    }

    std::vector<flatbuffers::Offset<Schema::Light>> lights;
    for (const Scene::Light& light : scene.getLights().getValues()) {
        const Schema::Transform transform = toSchema(light.transform);
        const Schema::Float3 color = toSchema(light.color);
        lights.push_back(Schema::CreateLight(builder, builder.CreateString(light.name), static_cast<Schema::LightType>(light.type), &transform, &color,
                                             light.power, light.radius, light.width, light.height, light.spotAngle, light.spotBlend, light.angle));
    }

    std::vector<flatbuffers::Offset<Schema::Camera>> cameras;
    for (const Scene::Camera& camera : scene.getCameras().getValues()) {
        const Schema::Transform transform = toSchema(camera.transform);
        std::vector<Schema::LensElement> lens;
        for (const Scene::LensElement& element : camera.lens.elements)
            lens.emplace_back(element.curvatureRadius, element.thickness, element.ior, element.abbe, element.apertureRadius);
        cameras.push_back(Schema::CreateCamera(builder, builder.CreateString(camera.name), static_cast<Schema::CameraType>(camera.type), &transform,
                                               camera.verticalFov, camera.orthographicHeight, camera.sensor.width, camera.sensor.height, camera.sensor.offset.x,
                                               camera.sensor.offset.y, camera.focusDistance, camera.fStop, static_cast<std::uint8_t>(camera.blades),
                                               camera.near, camera.far, builder.CreateVectorOfStructs(lens)));
    }

    const Scene::Environment& environment = scene.getEnvironment();
    const Schema::Float3 environmentColor = toSchema(environment.color);
    const std::uint32_t environmentTexture = indexOf(scene.getTextures(), environment.texture);
    const auto environmentOffset = Schema::CreateEnvironment(
        builder, environmentTexture == Core::Handle<Scene::Texture>::invalidIndex ? -1 : static_cast<std::int32_t>(environmentTexture), &environmentColor,
        environment.strength, environment.rotation, environment.visibleToCamera);
    const std::uint32_t activeCamera = indexOf(scene.getCameras(), scene.getActiveCamera());
    const auto nodes = writeNodes(builder, scene);

    const auto root =
        Schema::CreateScene(builder, builder.CreateVector(meshes), builder.CreateVectorOfStructs(instances), builder.CreateVector(materials),
                            builder.CreateVector(textures), builder.CreateVector(clouds), builder.CreateVector(volumes), builder.CreateVector(lights),
                            builder.CreateVector(cameras), environmentOffset, activeCamera == Core::Handle<Scene::Camera>::invalidIndex ? 0 : activeCamera,
                            builder.CreateVector(nodes));
    Schema::FinishSceneBuffer(builder, root);
    const std::span<const std::byte> bytes = std::as_bytes(std::span(builder.GetBufferPointer(), builder.GetSize()));
    return {bytes.begin(), bytes.end()};
}

std::expected<void, Core::Error> SceneWriter::write(const Scene::Scene& scene, const std::filesystem::path& path) {
    const std::vector<std::byte> bytes = write(scene);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot open '{}' for writing", path.string()));
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream)
        return Core::fail(Core::ErrorCode::Io, std::format("cannot write '{}'", path.string()));
    return {};
}

}
