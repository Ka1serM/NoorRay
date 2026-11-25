#define TINYGLTF_IMPLEMENTATION
#include "NoorRay/Import/GltfImporter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <map>
#include <numbers>

#include <tiny_gltf.h>
//! tinygltf includes windows.h, whose near and far macros hide the camera fields.
#undef near
#undef far

#include "NoorRay/Core/Log.h"
#include "NoorRay/Import/GltfMaterial.h"
#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Materials/TextureCache.h"

namespace NoorRay::Import {

namespace {

//! KHR_lights_punctual uses photometric units; Blender converts with 683 lm/W.
constexpr float lumensPerWatt = 683.0f;

float readComponent(const unsigned char* data, const int componentType, const bool normalized) {
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        float value = 0.0f;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return normalized ? static_cast<float>(*data) / 255.0f : static_cast<float>(*data);
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
        const float value = static_cast<float>(static_cast<std::int8_t>(*data));
        return normalized ? std::max(value / 127.0f, -1.0f) : value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        std::uint16_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? static_cast<float>(value) / 65535.0f : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
        std::int16_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? std::max(static_cast<float>(value) / 32767.0f, -1.0f) : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        std::uint32_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return static_cast<float>(value);
    }
    default:
        return 0.0f;
    }
}

//! Reads an accessor as floats, `components` per element; missing components are zero, a missing fourth component is one.
std::expected<std::vector<float>, Core::Error> readFloats(const tinygltf::Model& model, const int index, const int components) {
    if (index < 0 || static_cast<std::size_t>(index) >= model.accessors.size())
        return Core::fail(Core::ErrorCode::InvalidFile, "a glTF primitive references a missing accessor");
    const tinygltf::Accessor& accessor = model.accessors[index];
    const int sourceComponents = tinygltf::GetNumComponentsInType(accessor.type);
    const int componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    if (sourceComponents <= 0 || componentSize <= 0)
        return Core::fail(Core::ErrorCode::InvalidFile, "a glTF accessor has an unknown type");
    std::vector<float> result(accessor.count * components, 0.0f);
    if (components == 4 && sourceComponents < 4)
        for (std::size_t element = 0; element < accessor.count; ++element)
            result[element * 4 + 3] = 1.0f;
    const auto readElement = [&](const unsigned char* source, const std::size_t element) {
        for (int component = 0; component < std::min(components, sourceComponents); ++component)
            result[element * components + component] = readComponent(source + component * componentSize, accessor.componentType, accessor.normalized);
    };
    if (accessor.bufferView >= 0) {
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[view.buffer];
        const int stride = accessor.ByteStride(view);
        const std::size_t offset = view.byteOffset + accessor.byteOffset;
        if (stride <= 0 || (accessor.count > 0 && offset + (accessor.count - 1) * stride + sourceComponents * componentSize > buffer.data.size()))
            return Core::fail(Core::ErrorCode::InvalidFile, "a glTF accessor reads past its buffer");
        for (std::size_t element = 0; element < accessor.count; ++element)
            readElement(buffer.data.data() + offset + element * stride, element);
    }
    if (accessor.sparse.isSparse) {
        const auto& sparse = accessor.sparse;
        const tinygltf::BufferView& indexView = model.bufferViews[sparse.indices.bufferView];
        const tinygltf::BufferView& valueView = model.bufferViews[sparse.values.bufferView];
        const unsigned char* indices = model.buffers[indexView.buffer].data.data() + indexView.byteOffset + sparse.indices.byteOffset;
        const unsigned char* values = model.buffers[valueView.buffer].data.data() + valueView.byteOffset + sparse.values.byteOffset;
        const int indexSize = tinygltf::GetComponentSizeInBytes(sparse.indices.componentType);
        for (int entry = 0; entry < sparse.count; ++entry) {
            const std::size_t element = static_cast<std::size_t>(readComponent(indices + entry * indexSize, sparse.indices.componentType, false));
            if (element < accessor.count)
                readElement(values + entry * sourceComponents * componentSize, element);
        }
    }
    return result;
}

std::expected<std::vector<std::uint32_t>, Core::Error> readIndices(const tinygltf::Model& model, const int index) {
    std::expected<std::vector<float>, Core::Error> values = readFloats(model, index, 1);
    if (!values)
        return std::unexpected(std::move(values.error()));
    const tinygltf::Accessor& accessor = model.accessors[index];
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT && accessor.bufferView >= 0) {
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        const unsigned char* data = model.buffers[view.buffer].data.data() + view.byteOffset + accessor.byteOffset;
        const int stride = accessor.ByteStride(view);
        std::vector<std::uint32_t> result(accessor.count);
        for (std::size_t element = 0; element < accessor.count; ++element)
            std::memcpy(&result[element], data + element * stride, sizeof(std::uint32_t));
        return result;
    }
    std::vector<std::uint32_t> result(values->size());
    for (std::size_t element = 0; element < values->size(); ++element)
        result[element] = static_cast<std::uint32_t>((*values)[element]);
    return result;
}

std::vector<std::uint32_t> triangulate(const int mode, const std::vector<std::uint32_t>& indices) {
    std::vector<std::uint32_t> result;
    if (mode == TINYGLTF_MODE_TRIANGLES)
        return indices;
    for (std::size_t index = 2; index < indices.size(); ++index) {
        if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            const bool odd = (index & 1) != 0;
            result.insert(result.end(), {indices[index - 2], indices[odd ? index : index - 1], indices[odd ? index - 1 : index]});
        } else if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
            result.insert(result.end(), {indices[0], indices[index - 1], indices[index]});
        }
    }
    return result;
}

Core::Transform getNodeTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        Core::Transform transform;
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 4; ++column)
                transform.rows[row][column] = static_cast<float>(node.matrix[column * 4 + row]);
        return transform;
    }
    Core::float3 translation{0.0f, 0.0f, 0.0f};
    Core::Quaternion rotation;
    Core::float3 scale{1.0f, 1.0f, 1.0f};
    if (node.translation.size() == 3)
        translation = {static_cast<float>(node.translation[0]), static_cast<float>(node.translation[1]), static_cast<float>(node.translation[2])};
    if (node.rotation.size() == 4)
        rotation = {static_cast<float>(node.rotation[0]), static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]),
                    static_cast<float>(node.rotation[3])};
    if (node.scale.size() == 3)
        scale = {static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]), static_cast<float>(node.scale[2])};
    return Core::Transform::fromTrs(translation, rotation, scale);
}

class Importer {
public:
    Importer(const std::filesystem::path& path, tinygltf::Model& model, Scene::Scene& scene) : path(path), model(model), scene(scene), textureCache(scene) {}

    std::expected<void, Core::Error> run() {
        const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        if (model.scenes.empty()) {
            for (std::size_t node = 0; node < model.nodes.size(); ++node)
                if (std::expected<void, Core::Error> visited = visit(static_cast<int>(node), {}, {}, 0); !visited)
                    return visited;
            return {};
        }
        for (const int node : model.scenes[sceneIndex].nodes)
            if (std::expected<void, Core::Error> visited = visit(node, {}, {}, 0); !visited)
                return visited;
        return {};
    }

private:
    std::expected<void, Core::Error> visit(const int index, const Scene::NodeHandle parent, const Core::Transform& parentWorld, const std::uint32_t depth) {
        if (index < 0 || static_cast<std::size_t>(index) >= model.nodes.size() || depth > 256)
            return Core::fail(Core::ErrorCode::InvalidFile, "the glTF node hierarchy is invalid");
        const tinygltf::Node& node = model.nodes[index];
        const Core::Transform local = getNodeTransform(node);
        const Core::Transform world = parentWorld * local;
        std::vector<Scene::NodeObject> objects;
        if (node.mesh >= 0) {
            std::expected<Scene::MeshHandle, Core::Error> mesh = getMesh(node.mesh);
            if (!mesh)
                return std::unexpected(std::move(mesh.error()));
            objects.emplace_back(scene.addInstance({world, world, *mesh, Scene::InstanceFlags::None, static_cast<std::uint32_t>(index)}));
        }
        if (node.camera >= 0 && static_cast<std::size_t>(node.camera) < model.cameras.size())
            objects.emplace_back(addCamera(model.cameras[node.camera], world));
        if (node.light >= 0 && static_cast<std::size_t>(node.light) < model.lights.size())
            objects.emplace_back(addLight(model.lights[node.light], world));
        const std::string name = nodeName(node, index);
        const Scene::NodeHandle handle = scene.addNode({name, local, parent, objects.empty() ? Scene::NodeObject{} : objects.front()});
        //! A glTF node may hold a mesh, a camera and a light at once; a scene node holds one object, so the others become its children.
        for (std::size_t extra = 1; extra < objects.size(); ++extra)
            scene.addNode({name, {}, handle, objects[extra]});
        for (const int child : node.children)
            if (std::expected<void, Core::Error> visited = visit(child, handle, world, depth + 1); !visited)
                return visited;
        return {};
    }

    std::string nodeName(const tinygltf::Node& node, const int index) const {
        if (!node.name.empty())
            return node.name;
        if (node.mesh >= 0 && static_cast<std::size_t>(node.mesh) < model.meshes.size() && !model.meshes[node.mesh].name.empty())
            return model.meshes[node.mesh].name;
        return std::format("Node {}", index);
    }

    std::expected<Scene::MeshHandle, Core::Error> getMesh(const int index) {
        if (const auto found = meshes.find(index); found != meshes.end())
            return found->second;
        if (static_cast<std::size_t>(index) >= model.meshes.size())
            return Core::fail(Core::ErrorCode::InvalidFile, "a glTF node references a missing mesh");
        const tinygltf::Mesh& source = model.meshes[index];
        MeshData data;
        data.name = source.name.empty() ? std::format("mesh{}", index) : source.name;
        bool hasNormals = true;
        bool hasTangents = true;
        bool hasColors = false;
        std::uint32_t uvSets = Scene::maxUvSets;
        for (const tinygltf::Primitive& primitive : source.primitives) {
            hasNormals = hasNormals && primitive.attributes.contains("NORMAL");
            hasTangents = hasTangents && primitive.attributes.contains("TANGENT");
            hasColors = hasColors || primitive.attributes.contains("COLOR_0");
            std::uint32_t sets = 0;
            while (sets < Scene::maxUvSets && primitive.attributes.contains(std::format("TEXCOORD_{}", sets)))
                ++sets;
            uvSets = std::min(uvSets, sets);
        }
        for (const tinygltf::Primitive& primitive : source.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != TINYGLTF_MODE_TRIANGLE_STRIP && primitive.mode != TINYGLTF_MODE_TRIANGLE_FAN) {
                Core::Log::warning("glTF mesh '{}' has a primitive that is not made of triangles; it is skipped", data.name);
                continue;
            }
            const auto position = primitive.attributes.find("POSITION");
            if (position == primitive.attributes.end())
                continue;
            std::expected<std::vector<float>, Core::Error> positions = readFloats(model, position->second, 3);
            if (!positions)
                return std::unexpected(std::move(positions.error()));
            const std::uint32_t base = static_cast<std::uint32_t>(data.positions.size());
            const std::size_t count = positions->size() / 3;
            for (std::size_t vertex = 0; vertex < count; ++vertex)
                data.positions.push_back({(*positions)[vertex * 3], (*positions)[vertex * 3 + 1], (*positions)[vertex * 3 + 2]});
            if (hasNormals) {
                std::expected<std::vector<float>, Core::Error> normals = readFloats(model, primitive.attributes.at("NORMAL"), 3);
                if (!normals || normals->size() != count * 3)
                    return Core::fail(Core::ErrorCode::InvalidFile, std::format("glTF mesh '{}' has invalid normals", data.name));
                for (std::size_t vertex = 0; vertex < count; ++vertex)
                    data.normals.push_back({(*normals)[vertex * 3], (*normals)[vertex * 3 + 1], (*normals)[vertex * 3 + 2]});
            }
            if (hasTangents) {
                std::expected<std::vector<float>, Core::Error> tangents = readFloats(model, primitive.attributes.at("TANGENT"), 4);
                if (!tangents || tangents->size() != count * 4)
                    return Core::fail(Core::ErrorCode::InvalidFile, std::format("glTF mesh '{}' has invalid tangents", data.name));
                //! glTF bitangents point along +v with v down the image; after the flip to v up they point the other way.
                for (std::size_t vertex = 0; vertex < count; ++vertex)
                    data.tangents.push_back({(*tangents)[vertex * 4], (*tangents)[vertex * 4 + 1], (*tangents)[vertex * 4 + 2], -(*tangents)[vertex * 4 + 3]});
            }
            for (std::uint32_t set = 0; set < uvSets; ++set) {
                std::expected<std::vector<float>, Core::Error> uvs = readFloats(model, primitive.attributes.at(std::format("TEXCOORD_{}", set)), 2);
                if (!uvs || uvs->size() != count * 2)
                    return Core::fail(Core::ErrorCode::InvalidFile, std::format("glTF mesh '{}' has invalid UVs", data.name));
                for (std::size_t vertex = 0; vertex < count; ++vertex)
                    data.uvs[set].push_back({(*uvs)[vertex * 2], 1.0f - (*uvs)[vertex * 2 + 1]});
            }
            if (hasColors) {
                const auto color = primitive.attributes.find("COLOR_0");
                if (color == primitive.attributes.end()) {
                    data.colors.insert(data.colors.end(), count, Core::float4{1.0f, 1.0f, 1.0f, 1.0f});
                } else {
                    std::expected<std::vector<float>, Core::Error> colors = readFloats(model, color->second, 4);
                    if (!colors || colors->size() != count * 4)
                        return Core::fail(Core::ErrorCode::InvalidFile, std::format("glTF mesh '{}' has invalid colours", data.name));
                    for (std::size_t vertex = 0; vertex < count; ++vertex)
                        data.colors.push_back({(*colors)[vertex * 4], (*colors)[vertex * 4 + 1], (*colors)[vertex * 4 + 2], (*colors)[vertex * 4 + 3]});
                }
            }
            std::vector<std::uint32_t> indices;
            if (primitive.indices >= 0) {
                std::expected<std::vector<std::uint32_t>, Core::Error> read = readIndices(model, primitive.indices);
                if (!read)
                    return std::unexpected(std::move(read.error()));
                indices = std::move(*read);
            } else {
                indices.resize(count);
                for (std::size_t vertex = 0; vertex < count; ++vertex)
                    indices[vertex] = static_cast<std::uint32_t>(vertex);
            }
            indices = triangulate(primitive.mode, indices);
            if (std::ranges::any_of(indices, [&](const std::uint32_t value) { return value >= count; }))
                return Core::fail(Core::ErrorCode::InvalidFile, std::format("glTF mesh '{}' has an index past its vertices", data.name));
            std::expected<Scene::MaterialHandle, Core::Error> material = getMaterial(primitive.material, hasColors);
            if (!material)
                return std::unexpected(std::move(material.error()));
            const std::uint32_t firstIndex = static_cast<std::uint32_t>(data.indices.size());
            for (const std::uint32_t value : indices)
                data.indices.push_back(base + value);
            data.appendSubset(*material, firstIndex, static_cast<std::uint32_t>(indices.size()));
        }
        const Scene::MeshHandle handle = scene.addMesh(MeshBuilder::build(std::move(data)));
        meshes.emplace(index, handle);
        return handle;
    }

    std::optional<Scene::MaterialTexture> resolveTexture(const int index, const Materials::ColorSpace colorSpace) {
        if (index < 0 || static_cast<std::size_t>(index) >= model.textures.size())
            return std::nullopt;
        const tinygltf::Texture& texture = model.textures[index];
        if (texture.source < 0 || static_cast<std::size_t>(texture.source) >= model.images.size())
            return std::nullopt;
        const tinygltf::Image& image = model.images[texture.source];
        const std::string name = image.uri.empty() ? std::format("{}#image{}", path.string(), texture.source) : (path.parent_path() / image.uri).string();
        std::expected<Scene::TextureHandle, Core::Error> handle =
            textureCache.load(name, std::as_bytes(std::span(image.image.data(), image.image.size())), colorSpace);
        if (!handle) {
            Core::Log::warning("glTF texture '{}' is skipped: {}", name, handle.error().message);
            return std::nullopt;
        }
        Scene::MaterialTexture result;
        result.texture = *handle;
        if (texture.sampler >= 0 && static_cast<std::size_t>(texture.sampler) < model.samplers.size()) {
            const tinygltf::Sampler& sampler = model.samplers[texture.sampler];
            if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE)
                result.wrap = Scene::TextureWrap::Extend;
            if (sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT)
                result.wrap = Scene::TextureWrap::Mirror;
            if (sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST)
                result.filter = Scene::TextureFilter::Closest;
        }
        return result;
    }

    std::expected<Scene::MaterialHandle, Core::Error> getMaterial(const int index, const bool vertexColors) {
        const std::pair key{index, vertexColors};
        if (const auto found = materials.find(key); found != materials.end())
            return found->second;
        Scene::Material material;
        material.name = "default";
        if (index >= 0 && static_cast<std::size_t>(index) < model.materials.size())
            material = GltfMaterial::convert(model.materials[index],
                                             [this](const int texture, const Materials::ColorSpace colorSpace) { return resolveTexture(texture, colorSpace); });
        material.useVertexColor = vertexColors;
        const Scene::MaterialHandle handle = scene.addMaterial(std::move(material));
        materials.emplace(key, handle);
        return handle;
    }

    Scene::CameraHandle addCamera(const tinygltf::Camera& source, const Core::Transform& transform) {
        Scene::Camera camera;
        camera.name = source.name;
        camera.transform = transform;
        camera.previousTransform = transform;
        if (source.type == "orthographic") {
            camera.type = Scene::CameraType::Orthographic;
            camera.orthographicHeight = 2.0f * static_cast<float>(source.orthographic.ymag);
            camera.near = static_cast<float>(source.orthographic.znear);
            camera.far = static_cast<float>(source.orthographic.zfar);
            if (source.orthographic.ymag > 0.0)
                camera.sensor.width = camera.sensor.height * static_cast<float>(source.orthographic.xmag / source.orthographic.ymag);
        } else {
            camera.verticalFov = static_cast<float>(source.perspective.yfov);
            camera.near = static_cast<float>(source.perspective.znear);
            if (source.perspective.zfar > 0.0)
                camera.far = static_cast<float>(source.perspective.zfar);
            if (source.perspective.aspectRatio > 0.0)
                camera.sensor.width = camera.sensor.height * static_cast<float>(source.perspective.aspectRatio);
        }
        const Scene::CameraHandle handle = scene.addCamera(std::move(camera));
        if (!scene.findActiveCamera())
            scene.setActiveCamera(handle);
        return handle;
    }

    Scene::LightHandle addLight(const tinygltf::Light& source, const Core::Transform& transform) {
        Scene::Light light;
        light.name = source.name;
        light.transform = transform;
        if (source.color.size() == 3)
            light.color = {static_cast<float>(source.color[0]), static_cast<float>(source.color[1]), static_cast<float>(source.color[2])};
        const float intensity = static_cast<float>(source.intensity);
        if (source.type == "directional") {
            light.type = Scene::LightType::Directional;
            light.power = intensity / lumensPerWatt;
        } else {
            light.type = source.type == "spot" ? Scene::LightType::Spot : Scene::LightType::Point;
            light.power = intensity * 4.0f * std::numbers::pi_v<float> / lumensPerWatt;
            const float outer = static_cast<float>(source.spot.outerConeAngle);
            light.spotAngle = 2.0f * outer;
            light.spotBlend = outer > 0.0f ? 1.0f - static_cast<float>(source.spot.innerConeAngle) / outer : 0.0f;
        }
        return scene.addLight(std::move(light));
    }

    const std::filesystem::path& path;
    tinygltf::Model& model;
    Scene::Scene& scene;
    Materials::TextureCache textureCache;
    std::map<int, Scene::MeshHandle> meshes;
    std::map<std::pair<int, bool>, Scene::MaterialHandle> materials;
};

bool keepEncodedImage(tinygltf::Image* image, const int, std::string*, std::string*, const int, const int, const unsigned char* bytes, const int size, void*) {
    image->image.assign(bytes, bytes + size);
    image->as_is = true;
    return true;
}

}

std::expected<void, Core::Error> GltfImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(keepEncodedImage, nullptr);
    tinygltf::Model model;
    std::string error;
    std::string warning;
    const bool binary = path.extension() == ".glb" || path.extension() == ".GLB";
    const bool loaded =
        binary ? loader.LoadBinaryFromFile(&model, &error, &warning, path.string()) : loader.LoadASCIIFromFile(&model, &error, &warning, path.string());
    if (!warning.empty())
        Core::Log::warning("glTF '{}': {}", path.string(), warning);
    if (!loaded)
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("cannot read glTF '{}': {}", path.string(), error));
    return Importer(path, model, scene).run();
}

}
