#include "NoorRay/Format/SceneFile.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "NoorRay/Format/SceneConversion.h"

namespace NoorRay::Format {

namespace {

template <class T, class Source>
std::vector<T> copyVector(const flatbuffers::Vector<Source>* source) {
    if (!source)
        return {};
    return {source->begin(), source->end()};
}

std::vector<Core::float3> copyPositions(const flatbuffers::Vector<const Schema::Float3*>* source) {
    std::vector<Core::float3> result;
    if (!source)
        return result;
    result.resize(source->size());
    static_assert(sizeof(Schema::Float3) == sizeof(Core::float3));
    std::memcpy(result.data(), source->data(), result.size() * sizeof(Core::float3));
    return result;
}

std::string toString(const flatbuffers::String* value) {
    return value ? value->str() : std::string();
}

NoorRHI::ImageFormat toImageFormat(const Schema::TextureFormat format) {
    switch (format) {
    case Schema::TextureFormat::Rgba8Unorm:
        return NoorRHI::ImageFormat::Rgba8Unorm;
    case Schema::TextureFormat::Rgba8Srgb:
        return NoorRHI::ImageFormat::Rgba8Srgb;
    case Schema::TextureFormat::Rgba16Float:
        return NoorRHI::ImageFormat::Rgba16Float;
    case Schema::TextureFormat::Bc4Unorm:
        return NoorRHI::ImageFormat::Bc4Unorm;
    case Schema::TextureFormat::Bc5Unorm:
        return NoorRHI::ImageFormat::Bc5Unorm;
    case Schema::TextureFormat::Bc6hUfloat:
        return NoorRHI::ImageFormat::Bc6hUfloat;
    case Schema::TextureFormat::Bc7Unorm:
        return NoorRHI::ImageFormat::Bc7Unorm;
    case Schema::TextureFormat::Bc7Srgb:
        return NoorRHI::ImageFormat::Bc7Srgb;
    }
    return NoorRHI::ImageFormat::Undefined;
}

template <class Handle>
std::expected<Handle, Core::Error> lookup(const std::vector<Handle>& handles, const std::uint32_t index, const std::string_view kind) {
    if (index >= handles.size())
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the scene file references {} {}, which does not exist", kind, index));
    return handles[index];
}

struct ObjectStart {
    std::size_t instances = 0;
    std::size_t lights = 0;
    std::size_t gaussianClouds = 0;
    std::size_t volumes = 0;
    std::size_t cameras = 0;
};

template <class T>
std::expected<Scene::NodeObject, Core::Error> objectAt(const Core::SlotMap<T>& objects, const std::size_t first, const std::uint32_t index, const std::string_view kind) {
    if (first + index >= objects.size())
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("the scene file references {} {}, which does not exist", kind, index));
    return Scene::NodeObject{objects.keyAt(static_cast<std::uint32_t>(first + index))};
}

std::expected<Scene::NodeObject, Core::Error> nodeObject(const Scene::Scene& scene, const ObjectStart& start, const Schema::Node& node) {
    switch (node.object_kind()) {
    case Schema::NodeObjectKind::None:
        return Scene::NodeObject{};
    case Schema::NodeObjectKind::Instance:
        return objectAt(scene.getInstances(), start.instances, node.object(), "instance");
    case Schema::NodeObjectKind::Light:
        return objectAt(scene.getLights(), start.lights, node.object(), "light");
    case Schema::NodeObjectKind::GaussianCloud:
        return objectAt(scene.getGaussianClouds(), start.gaussianClouds, node.object(), "Gaussian cloud");
    case Schema::NodeObjectKind::Volume:
        return objectAt(scene.getVolumes(), start.volumes, node.object(), "volume");
    case Schema::NodeObjectKind::Camera:
        return objectAt(scene.getCameras(), start.cameras, node.object(), "camera");
    }
    return Core::fail(Core::ErrorCode::InvalidFile, "the scene file has a node with an unknown object kind");
}

std::expected<void, Core::Error> loadNodes(const Schema::Scene& source, Scene::Scene& scene, const ObjectStart& start) {
    if (!source.nodes())
        return {};
    std::vector<Scene::NodeHandle> nodes;
    nodes.reserve(source.nodes()->size());
    for (const Schema::Node* item : *source.nodes()) {
        std::expected<Scene::NodeObject, Core::Error> object = nodeObject(scene, start, *item);
        if (!object)
            return std::unexpected(std::move(object.error()));
        Scene::NodeHandle parent;
        if (item->parent() != Scene::NodeHandle::invalidIndex) {
            std::expected<Scene::NodeHandle, Core::Error> found = lookup(nodes, item->parent(), "parent node");
            if (!found)
                return std::unexpected(std::move(found.error()));
            parent = *found;
        }
        const Core::Transform transform = item->transform() ? toTransform(*item->transform()) : Core::Transform{};
        const Scene::NodeHandle node = scene.addNode({toString(item->name()), transform, parent, *object});
        if (!node)
            return Core::fail(Core::ErrorCode::InvalidFile, "the scene file gives one object to two nodes");
        nodes.push_back(node);
    }
    return {};
}

std::expected<void, Core::Error> loadMeshes(const Schema::Scene& source, Scene::Scene& scene, const std::vector<Scene::MaterialHandle>& materials,
                                            std::vector<Scene::MeshHandle>& meshes) {
    if (!source.meshes())
        return {};
    for (const Schema::Mesh* item : *source.meshes()) {
        Scene::Mesh mesh;
        mesh.name = toString(item->name());
        mesh.positions = copyPositions(item->positions());
        mesh.normals = copyVector<std::uint32_t>(item->normals());
        mesh.tangents = copyVector<std::uint32_t>(item->tangents());
        if (item->uvs()) {
            if (item->uvs()->size() > Scene::maxUvSets)
                return Core::fail(Core::ErrorCode::InvalidFile, std::format("mesh '{}' has more than {} UV sets", mesh.name, Scene::maxUvSets));
            for (std::uint32_t set = 0; set < item->uvs()->size(); ++set)
                mesh.uvs[set] = copyVector<std::uint32_t>(item->uvs()->Get(set)->values());
        }
        mesh.colors = copyVector<std::uint32_t>(item->colors());
        mesh.indices = copyVector<std::uint32_t>(item->indices());
        const std::size_t vertexCount = mesh.positions.size();
        const auto streamValid = [&](const std::vector<std::uint32_t>& stream) { return stream.empty() || stream.size() == vertexCount; };
        if (!streamValid(mesh.normals) || !streamValid(mesh.tangents) || !streamValid(mesh.colors) || !std::ranges::all_of(mesh.uvs, streamValid)
            || mesh.indices.size() % 3 != 0)
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("mesh '{}' has vertex streams of different lengths", mesh.name));
        if (std::ranges::any_of(mesh.indices, [&](const std::uint32_t index) { return index >= vertexCount; }))
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("mesh '{}' has an index past its vertices", mesh.name));
        if (item->subsets()) {
            for (const Schema::MeshSubset* subset : *item->subsets()) {
                std::expected<Scene::MaterialHandle, Core::Error> material = lookup(materials, subset->material(), "material");
                if (!material)
                    return std::unexpected(std::move(material.error()));
                if (std::size_t{subset->first_index()} + subset->index_count() > mesh.indices.size())
                    return Core::fail(Core::ErrorCode::InvalidFile, std::format("mesh '{}' has a subset past its indices", mesh.name));
                mesh.subsets.push_back({subset->first_index(), subset->index_count(), *material});
            }
        }
        mesh.bounds = toBounds(item->bounds());
        meshes.push_back(scene.addMesh(std::move(mesh)));
    }
    return {};
}

std::expected<void, Core::Error> loadMaterials(const Schema::Scene& source, Scene::Scene& scene, const std::vector<Scene::TextureHandle>& textures,
                                               std::vector<Scene::MaterialHandle>& materials) {
    if (!source.materials())
        return {};
    for (const Schema::Material* item : *source.materials()) {
        const Schema::MaterialParameters* parameters = item->parameters();
        if (!parameters)
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("material '{}' has no parameters", toString(item->name())));
        Scene::Material material;
        material.name = toString(item->name());
        material.baseColor = toFloat3(parameters->base_color());
        material.alpha = parameters->alpha();
        material.alphaMode = static_cast<Scene::AlphaMode>(parameters->alpha_mode());
        material.useVertexColor = parameters->use_vertex_color();
        material.roughnessIsGlossiness = parameters->roughness_is_glossiness();
        material.alphaCutoff = parameters->alpha_cutoff();
        material.metallic = parameters->metallic();
        material.roughness = parameters->roughness();
        material.normalScale = parameters->normal_scale();
        material.ior = parameters->ior();
        material.specularIorLevel = parameters->specular_ior_level();
        material.specularTint = toFloat3(parameters->specular_tint());
        material.transmission = parameters->transmission();
        material.anisotropic = parameters->anisotropic();
        material.anisotropicRotation = parameters->anisotropic_rotation();
        material.coat = parameters->coat();
        material.coatRoughness = parameters->coat_roughness();
        material.coatNormalScale = parameters->coat_normal_scale();
        material.sheenColor = toFloat3(parameters->sheen_color());
        material.sheenRoughness = parameters->sheen_roughness();
        material.emissionColor = toFloat3(parameters->emission_color());
        material.emissionStrength = parameters->emission_strength();
        material.thinFilmThickness = parameters->thin_film_thickness();
        material.thinFilmIor = parameters->thin_film_ior();
        material.dispersion = parameters->dispersion();
        if (item->textures()) {
            for (const Schema::MaterialTexture* slot : *item->textures()) {
                const auto slotIndex = static_cast<std::size_t>(slot->slot());
                if (slotIndex >= Scene::textureSlotCount)
                    return Core::fail(Core::ErrorCode::InvalidFile, std::format("material '{}' has the unknown texture slot {}", material.name, slotIndex));
                std::expected<Scene::TextureHandle, Core::Error> texture = lookup(textures, slot->texture(), "texture");
                if (!texture)
                    return std::unexpected(std::move(texture.error()));
                material.textures[slotIndex] = Scene::MaterialTexture{
                    .texture = *texture,
                    .wrap = static_cast<Scene::TextureWrap>(slot->wrap()),
                    .filter = static_cast<Scene::TextureFilter>(slot->filter()),
                    .uvSet = slot->uv_set(),
                    .channel = slot->channel(),
                    .transform = {toFloat3(slot->row0()), toFloat3(slot->row1())},
                };
            }
        }
        materials.push_back(scene.addMaterial(std::move(material)));
    }
    return {};
}

std::expected<void, Core::Error> loadTextures(const Schema::Scene& source, Scene::Scene& scene, std::vector<Scene::TextureHandle>& textures) {
    if (!source.textures())
        return {};
    for (const Schema::Texture* item : *source.textures()) {
        Scene::Texture texture;
        texture.name = toString(item->name());
        texture.format = toImageFormat(item->format());
        texture.width = item->width();
        texture.height = item->height();
        if (texture.format == NoorRHI::ImageFormat::Undefined || !item->data() || !item->levels())
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("texture '{}' has no data or an unknown format", texture.name));
        for (const Schema::TextureLevel* level : *item->levels()) {
            if (level->offset() + level->size() > item->data()->size())
                return Core::fail(Core::ErrorCode::InvalidFile, std::format("texture '{}' has a level past its data", texture.name));
            texture.levels.push_back({static_cast<std::size_t>(level->offset()), static_cast<std::size_t>(level->size()), level->width(), level->height()});
        }
        const std::span<const std::byte> data = std::as_bytes(std::span(item->data()->data(), item->data()->size()));
        texture.data.assign(data.begin(), data.end());
        textures.push_back(scene.addTexture(std::move(texture)));
    }
    return {};
}

void loadGaussians(const Schema::Scene& source, Scene::Scene& scene) {
    if (!source.gaussian_clouds())
        return;
    for (const Schema::GaussianCloud* item : *source.gaussian_clouds()) {
        Scene::GaussianCloud cloud;
        cloud.name = toString(item->name());
        if (item->transform())
            cloud.transform = toTransform(*item->transform());
        cloud.positions = copyPositions(item->positions());
        cloud.scales = copyPositions(item->scales());
        const std::vector<float> rotations = copyVector<float>(item->rotations());
        cloud.rotations.resize(rotations.size() / 4);
        std::memcpy(cloud.rotations.data(), rotations.data(), cloud.rotations.size() * sizeof(Core::float4));
        cloud.opacities = copyVector<float>(item->opacities());
        cloud.shDegree = item->sh_degree();
        cloud.sh = copyVector<std::uint16_t>(item->sh());
        scene.addGaussianCloud(std::move(cloud));
    }
}

void loadVolumes(const Schema::Scene& source, Scene::Scene& scene) {
    if (!source.volumes())
        return;
    for (const Schema::Volume* item : *source.volumes()) {
        Scene::Volume volume;
        volume.name = toString(item->name());
        volume.kind = item->kind() == Schema::VolumeKind::Grid ? Scene::VolumeKind::Grid : Scene::VolumeKind::Homogeneous;
        if (item->transform())
            volume.transform = toTransform(*item->transform());
        volume.bounds = toBounds(item->bounds());
        if (item->absorption())
            volume.absorption = toFloat3(*item->absorption());
        if (item->scattering())
            volume.scattering = toFloat3(*item->scattering());
        if (item->emission())
            volume.emission = toFloat3(*item->emission());
        volume.anisotropy = item->anisotropy();
        volume.densityScale = item->density_scale();
        if (item->resolution() && item->resolution()->size() == 3)
            volume.resolution = {item->resolution()->Get(0), item->resolution()->Get(1), item->resolution()->Get(2)};
        volume.density = copyVector<std::uint16_t>(item->density());
        scene.addVolume(std::move(volume));
    }
}

void loadLights(const Schema::Scene& source, Scene::Scene& scene) {
    if (!source.lights())
        return;
    for (const Schema::Light* item : *source.lights()) {
        Scene::Light light;
        light.name = toString(item->name());
        light.type = static_cast<Scene::LightType>(item->type());
        if (item->transform())
            light.transform = toTransform(*item->transform());
        if (item->color())
            light.color = toFloat3(*item->color());
        light.power = item->power();
        light.radius = item->radius();
        light.width = item->width();
        light.height = item->height();
        light.spotAngle = item->spot_angle();
        light.spotBlend = item->spot_blend();
        light.angle = item->angle();
        scene.addLight(std::move(light));
    }
}

std::vector<Scene::CameraHandle> loadCameras(const Schema::Scene& source, Scene::Scene& scene) {
    std::vector<Scene::CameraHandle> cameras;
    if (!source.cameras())
        return cameras;
    for (const Schema::Camera* item : *source.cameras()) {
        Scene::Camera camera;
        camera.name = toString(item->name());
        camera.type = static_cast<Scene::CameraType>(item->type());
        if (item->transform())
            camera.transform = toTransform(*item->transform());
        camera.previousTransform = camera.transform;
        camera.verticalFov = item->vertical_fov();
        camera.orthographicHeight = item->orthographic_height();
        camera.sensor = {item->sensor_width(), item->sensor_height(), {item->sensor_offset_x(), item->sensor_offset_y()}};
        camera.focusDistance = item->focus_distance();
        camera.fStop = item->f_stop();
        camera.blades = item->blades();
        camera.near = item->near();
        camera.far = item->far();
        if (item->lens())
            for (const Schema::LensElement* element : *item->lens())
                camera.lens.elements.push_back(
                    {element->curvature_radius(), element->thickness(), element->ior(), element->abbe(), element->aperture_radius()});
        cameras.push_back(scene.addCamera(std::move(camera)));
    }
    return cameras;
}

}

std::expected<void, Core::Error> SceneFile::open(const std::filesystem::path& path) {
    if (std::expected<void, Core::Error> opened = file.open(path); !opened)
        return opened;
    return verify(file.getBytes());
}

std::expected<void, Core::Error> SceneFile::verify(const std::span<const std::byte> bytes) {
    flatbuffers::Verifier::Options options;
    options.max_tables = 100'000'000;
    flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(), options);
    if (!Schema::VerifySceneBuffer(verifier))
        return Core::fail(Core::ErrorCode::InvalidFile, "the file is not a valid NoorRay scene (NRSC)");
    return {};
}

std::expected<void, Core::Error> SceneFile::load(const std::span<const std::byte> bytes, Scene::Scene& scene) {
    const Schema::Scene* source = Schema::GetScene(bytes.data());
    const ObjectStart start{scene.getInstances().size(), scene.getLights().size(), scene.getGaussianClouds().size(), scene.getVolumes().size(),
                            scene.getCameras().size()};
    std::vector<Scene::TextureHandle> textures;
    std::vector<Scene::MaterialHandle> materials;
    std::vector<Scene::MeshHandle> meshes;
    if (std::expected<void, Core::Error> loaded = loadTextures(*source, scene, textures); !loaded)
        return loaded;
    if (std::expected<void, Core::Error> loaded = loadMaterials(*source, scene, textures, materials); !loaded)
        return loaded;
    if (std::expected<void, Core::Error> loaded = loadMeshes(*source, scene, materials, meshes); !loaded)
        return loaded;
    if (source->instances()) {
        for (const Schema::Instance* item : *source->instances()) {
            std::expected<Scene::MeshHandle, Core::Error> mesh = lookup(meshes, item->mesh(), "mesh");
            if (!mesh)
                return std::unexpected(std::move(mesh.error()));
            Scene::MaterialHandle material;
            if (item->material() != Scene::MaterialHandle::invalidIndex) {
                std::expected<Scene::MaterialHandle, Core::Error> found = lookup(materials, item->material(), "material");
                if (!found)
                    return std::unexpected(std::move(found.error()));
                material = *found;
            }
            const Core::Transform transform = toTransform(item->transform());
            scene.addInstance({transform, transform, *mesh, static_cast<Scene::InstanceFlags>(item->flags()), item->id(), material});
        }
    }
    loadGaussians(*source, scene);
    loadVolumes(*source, scene);
    loadLights(*source, scene);
    const std::vector<Scene::CameraHandle> cameras = loadCameras(*source, scene);
    if (source->active_camera() < cameras.size())
        scene.setActiveCamera(cameras[source->active_camera()]);
    if (std::expected<void, Core::Error> loaded = loadNodes(*source, scene, start); !loaded)
        return loaded;
    if (const Schema::Environment* environment = source->environment()) {
        Scene::Environment value;
        if (environment->texture() >= 0) {
            std::expected<Scene::TextureHandle, Core::Error> texture = lookup(textures, static_cast<std::uint32_t>(environment->texture()), "texture");
            if (!texture)
                return std::unexpected(std::move(texture.error()));
            value.texture = *texture;
        }
        if (environment->color())
            value.color = toFloat3(*environment->color());
        value.strength = environment->strength();
        value.rotation = environment->rotation();
        value.visibleToCamera = environment->visible_to_camera();
        scene.setEnvironment(value);
    }
    return {};
}

}
