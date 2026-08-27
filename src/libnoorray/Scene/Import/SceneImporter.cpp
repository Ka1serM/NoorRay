#include "SceneImporter.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include "Rendering/Camera/CameraInstance.h"
#include "Scene/Resources/Texture.h"
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <iostream>
#include "Scene/Scene.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Log.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Scene/Objects/MeshInstance.h"
#include "glm/gtx/matrix_decompose.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Geometry/Mesh/Transform.h"
#include "Scene/CoordinateSystem.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Scene/Import/SceneReader.h"
#include "Scene/Import/SceneUsd.h"

namespace {
template<class Function>
void boundedParallelFor(const size_t count, Function&& function)
{
    if (count == 0)
        return;
    // Mesh preparation is memory-bandwidth and allocation heavy. Four workers
    // overlap accessor decoding well without allowing one import to consume
    // the application's complete TBB pool or issue dozens of concurrent
    // calls.
    constexpr size_t MaxMeshPreparationWorkers = 4;
    const int concurrency = static_cast<int>(
        std::min(count, MaxMeshPreparationWorkers));
    tbb::task_arena arena(concurrency);
    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, count, 1),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t index = range.begin(); index != range.end(); ++index)
                    function(index);
            });
    });
}

std::string lowerPath(std::string value)
{
    std::ranges::transform(value, value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// Resolves an asset path referenced from a scene JSON. If the path does not
// exist as given (e.g. relative to whatever the process cwd happens to be),
// fall back to resolving it relative to the compiled-in asset directory, so
// scenes can reference paths like "tests/utah_teapot.obj" regardless of cwd.
std::filesystem::path resolveAssetPath(const std::string& filepath)
{
    const std::filesystem::path direct(filepath);
    if (std::filesystem::exists(direct))
        return direct;
    const std::filesystem::path fallback = std::filesystem::path(NOORRAY_ASSET_DIR) / filepath;
    if (std::filesystem::exists(fallback))
        return fallback;
    return direct;
}

float gltfColorComponent(const tinygltf::Model& model,
    const tinygltf::Accessor& accessor, const size_t vertex,
    const size_t component)
{
    if (accessor.bufferView < 0
        || static_cast<size_t>(accessor.bufferView) >= model.bufferViews.size())
        return component == 3 ? 1.0f : 0.0f;
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    if (view.buffer < 0
        || static_cast<size_t>(view.buffer) >= model.buffers.size())
        return component == 3 ? 1.0f : 0.0f;
    const int componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    const int stride = accessor.ByteStride(view);
    if (componentSize <= 0 || stride <= 0)
        return component == 3 ? 1.0f : 0.0f;
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    const size_t offset = view.byteOffset + accessor.byteOffset
        + vertex * static_cast<size_t>(stride)
        + component * static_cast<size_t>(componentSize);
    if (offset + static_cast<size_t>(componentSize) > buffer.data.size())
        return component == 3 ? 1.0f : 0.0f;
    const unsigned char* data = buffer.data.data() + offset;
    switch (accessor.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<float>(*data) / 255.0f;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    {
        uint16_t value{};
        std::memcpy(&value, data, sizeof(value));
        return static_cast<float>(value) / 65535.0f;
    }
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
    {
        float value{};
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    default:
        return component == 3 ? 1.0f : 0.0f;
    }
}

glm::vec4 gltfVertexColor(const tinygltf::Model& model,
    const tinygltf::Accessor& accessor, const size_t vertex)
{
    const int components = tinygltf::GetNumComponentsInType(accessor.type);
    if (components != 3 && components != 4)
        return glm::vec4(1.0f);
    return glm::vec4(
        gltfColorComponent(model, accessor, vertex, 0),
        gltfColorComponent(model, accessor, vertex, 1),
        gltfColorComponent(model, accessor, vertex, 2),
        components == 4
            ? gltfColorComponent(model, accessor, vertex, 3) : 1.0f);
}

struct GltfFloatAccessor
{
    const unsigned char* data{};
    size_t stride{};
    size_t count{};
    int components{};

    explicit operator bool() const { return data != nullptr; }

    float component(const size_t element, const size_t index) const
    {
        float value{};
        std::memcpy(&value,
            data + element * stride + index * sizeof(float), sizeof(value));
        return value;
    }
};

GltfFloatAccessor gltfFloatAccessor(const tinygltf::Model& model,
    const tinygltf::Primitive& primitive, const char* attribute,
    const int requiredComponents)
{
    const auto found = primitive.attributes.find(attribute);
    if (found == primitive.attributes.end() || found->second < 0
        || static_cast<size_t>(found->second) >= model.accessors.size())
        return {};
    const tinygltf::Accessor& accessor = model.accessors[found->second];
    if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT
        || accessor.bufferView < 0
        || static_cast<size_t>(accessor.bufferView) >= model.bufferViews.size())
        return {};
    const int components = tinygltf::GetNumComponentsInType(accessor.type);
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const int stride = accessor.ByteStride(view);
    if (components < requiredComponents || stride <= 0 || view.buffer < 0
        || static_cast<size_t>(view.buffer) >= model.buffers.size())
        return {};
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    const size_t offset = view.byteOffset + accessor.byteOffset;
    const size_t elementBytes =
        static_cast<size_t>(requiredComponents) * sizeof(float);
    if (offset > buffer.data.size())
        return {};
    const size_t available = buffer.data.size() - offset;
    if (accessor.count != 0
        && (elementBytes > available
            || accessor.count - 1
                > (available - elementBytes) / static_cast<size_t>(stride)))
        return {};
    return {
        buffer.data.data() + offset,
        static_cast<size_t>(stride),
        accessor.count,
        components,
    };
}
}

void SceneImporter::ImportGltfScene(Scene& scene, const std::string& filepath)
{
    // Boilerplate: Load GLTF file from disk
    const std::filesystem::path filePath = resolveAssetPath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    const std::filesystem::path gltfDir = filePath.parent_path();
    const std::string resolvedFilepath = filePath.string();

    // A second import of the same file (e.g. two scene-graph entries
    // pointing at the same asset) clones the hierarchy this call already
    // built instead of re-parsing the glTF and re-uploading every mesh. The
    // scene-wide texture library is already shared by handle, so cloning does
    // not duplicate image resources either.
    if (const SceneObjectHandle cached = scene.findImportedFileRoot(resolvedFilepath);
        cached.isValid())
    {
        if (SceneObject* cachedRoot = scene.getObject(cached)) {
            const auto clone = scene.cloneHierarchy(cachedRoot);
            scene.setActiveObject(clone->getHandle());
            return;
        }
    }

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;

    const bool success = filePath.extension() == ".glb"
        ? loader.LoadBinaryFromFile(&model, &err, &warn, resolvedFilepath)
        : loader.LoadASCIIFromFile(&model, &err, &warn, resolvedFilepath);

    if (!success)
        throw std::runtime_error("Failed to load GLTF file: " + warn + err);
    if (!warn.empty())
        LOG_ERROR("TinyGLTF Warning: " << warn);
    if (!err.empty())
        LOG_ERROR("TinyGLTF Info/Error: " << err);

    // STEP 1: Import all unique assets (meshes and materials)

    // Load Materials
    std::vector<MaterialAuthoring> globalMaterials;
    // Keyed by (glTF image index, encoding) so two materials that sample the
    // same source image (a common texture-atlas pattern) upload it once
    // instead of decoding and uploading a duplicate copy per material.
    std::map<std::pair<int, int>, TextureHandle> imageTextureCache;
    for (const auto& mat : model.materials) {
        MaterialAuthoring material{};
        const auto& pbr = mat.pbrMetallicRoughness;
        material.albedo = glm::make_vec3(pbr.baseColorFactor.data());
        material.metallic = static_cast<float>(pbr.metallicFactor);
        material.roughness = static_cast<float>(pbr.roughnessFactor);
        material.emission = glm::make_vec3(mat.emissiveFactor.data());
        material.emissionStrength = (length2(material.emission) > 1e-6f) ? 1.0f : 0.0f;
        material.transmission = 0.0f;

        if (mat.alphaMode == "BLEND") {
             material.opacity = static_cast<float>(pbr.baseColorFactor[3]);
        } else {
             material.opacity = 1.0f;
        }

        if (mat.extensions.contains("KHR_materials_transmission")) {
            const auto& ext = mat.extensions.at("KHR_materials_transmission");
            if (ext.Has("transmissionFactor"))
                material.transmission = static_cast<float>(ext.Get("transmissionFactor").Get<double>());
        }
        if (mat.extensions.contains("KHR_materials_ior")) {
             const auto& ext = mat.extensions.at("KHR_materials_ior");
             if (ext.Has("ior"))
                 material.sellmeier = constantIorSellmeier(
                     static_cast<float>(ext.Get("ior").Get<double>()));
        }

        auto addTexture = [&](int textureIndex, int& materialIndex, TextureEncoding encoding) {
            if (textureIndex < 0) return;
            const tinygltf::Texture& tex = model.textures[textureIndex];
            if (tex.source < 0) return;
            const tinygltf::Image& image = model.images[tex.source];

            // Images named *_linear contain already-linear data; skip sRGB decode.
            if (encoding == TextureEncoding::Srgb8 &&
                (image.name.find("_linear") != std::string::npos ||
                 image.uri.find("_linear")   != std::string::npos))
                encoding = TextureEncoding::Linear8;

            const std::pair<int, int> cacheKey{tex.source, static_cast<int>(encoding)};
            if (const auto cached = imageTextureCache.find(cacheKey);
                cached != imageTextureCache.end())
            {
                materialIndex = static_cast<int>(cached->second.index());
                return;
            }

            if (!image.uri.empty()) {
                const std::filesystem::path texturePath = gltfDir / image.uri;
                if (std::filesystem::exists(texturePath)) {
                    const TextureHandle texture = scene.addTexture(
                        Texture(texturePath.string(), encoding));
                    materialIndex = static_cast<int>(texture.index());
                    imageTextureCache[cacheKey] = texture;
                } else {
                    LOG_ERROR("Warning: Texture file not found: " << texturePath.string());
                }
            } else if (!image.image.empty()) {
                const std::string texName = image.name.empty()
                    ? "texture_" + std::to_string(textureIndex)
                    : image.name;
                const TextureHandle texture = scene.addTexture(Texture(texName,
                    image.image.data(), image.width, image.height, encoding));
                materialIndex = static_cast<int>(texture.index());
                imageTextureCache[cacheKey] = texture;
            } else {
                LOG_ERROR("Warning: Embedded texture has no decoded data");
            }
        };

        addTexture(pbr.baseColorTexture.index, material.albedoIndex,
                   TextureEncoding::Srgb8);
        addTexture(pbr.metallicRoughnessTexture.index, material.roughnessIndex,
                   TextureEncoding::Linear8);
        addTexture(mat.normalTexture.index, material.normalIndex,
                   TextureEncoding::Linear8);
        addTexture(mat.emissiveTexture.index, material.emissionIndex,
                   TextureEncoding::Srgb8);

        globalMaterials.push_back(material);
    }
    if (globalMaterials.empty())
        globalMaterials.emplace_back(); // Default material

    // Phase 1: decode every primitive into its final managed payload. The
    // tinygltf model is immutable here and each worker owns one output slot,
    // so no worker mutates Scene or shares writable geometry.
    struct PrimitiveLocation
    {
        size_t meshIndex;
        size_t primitiveIndex;
    };
    struct PreparedPrimitive
    {
        MeshGeometry geometry;
        std::string name;
        size_t meshIndex{};
        size_t materialIndex{};
        bool ready{};
    };

    std::vector<PrimitiveLocation> locations;
    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex)
        for (size_t primitiveIndex = 0;
            primitiveIndex < model.meshes[meshIndex].primitives.size();
            ++primitiveIndex)
            locations.push_back({meshIndex, primitiveIndex});

    const std::string fallbackMeshName = nameFromPath(filepath);
    std::vector<PreparedPrimitive> prepared(locations.size());
    boundedParallelFor(locations.size(), [&](const size_t jobIndex) {
        const PrimitiveLocation location = locations[jobIndex];
        const tinygltf::Mesh& mesh = model.meshes[location.meshIndex];
        const tinygltf::Primitive& primitive =
            mesh.primitives[location.primitiveIndex];
        PreparedPrimitive& output = prepared[jobIndex];

        // The renderer currently consumes triangles. glTF's default primitive
        // mode is TRIANGLES when mode is omitted.
        if (primitive.mode != -1
            && primitive.mode != TINYGLTF_MODE_TRIANGLES)
            return;

        const GltfFloatAccessor positions =
            gltfFloatAccessor(model, primitive, "POSITION", 3);
        const GltfFloatAccessor normals =
            gltfFloatAccessor(model, primitive, "NORMAL", 3);
        const GltfFloatAccessor tangents =
            gltfFloatAccessor(model, primitive, "TANGENT", 4);
        const GltfFloatAccessor texcoords =
            gltfFloatAccessor(model, primitive, "TEXCOORD_0", 2);
        const tinygltf::Accessor* colors = nullptr;
        if (!positions)
            return;
        const size_t vertexCount = positions.count;
        if (const auto colorIt = primitive.attributes.find("COLOR_0");
            colorIt != primitive.attributes.end()
            && colorIt->second >= 0
            && static_cast<size_t>(colorIt->second) < model.accessors.size()
            && model.accessors[colorIt->second].count >= vertexCount)
            colors = &model.accessors[colorIt->second];

        output.geometry.vertices.reserve(vertexCount);
        for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
            Vertex vertex{};
            vertex.position = nr::coords::toOpenGlVector(
                vec3(positions.component(vertexIndex, 0),
                    positions.component(vertexIndex, 1),
                    positions.component(vertexIndex, 2)),
                nr::coords::OpenGlSpace);
            vertex.normal = normals && normals.count > vertexIndex
                ? normalize(nr::coords::toOpenGlVector(
                    vec3(normals.component(vertexIndex, 0),
                        normals.component(vertexIndex, 1),
                        normals.component(vertexIndex, 2)),
                    nr::coords::OpenGlSpace))
                : vec3(0, 1, 0);
            vertex.uv = texcoords && texcoords.count > vertexIndex
                ? vec2(texcoords.component(vertexIndex, 0),
                    1.0f - texcoords.component(vertexIndex, 1))
                : vec2(0);
            vertex.tangent = tangents && tangents.count > vertexIndex
                ? normalize(nr::coords::toOpenGlVector(
                    vec3(tangents.component(vertexIndex, 0),
                        tangents.component(vertexIndex, 1),
                        tangents.component(vertexIndex, 2)),
                    nr::coords::OpenGlSpace))
                : vec3(1, 0, 0);
            vertex.tangentSign = tangents && tangents.count > vertexIndex
                ? tangents.component(vertexIndex, 3) : 1.0f;
            if (colors)
                vertex.color = nr::vertex_color::packLinear(
                    gltfVertexColor(model, *colors, vertexIndex));
            output.geometry.vertices.push_back(vertex);
        }

        if (primitive.indices < 0) {
            // Device traversal still expects an explicit index buffer. Fill it
            // directly in final storage; this is the only remaining work for
            // a glTF primitive whose source indices are implicit.
            output.geometry.indices.reserve(vertexCount);
            for (size_t index = 0; index < vertexCount; ++index)
                output.geometry.indices.push_back(
                    static_cast<uint32_t>(index));
        } else {
            if (static_cast<size_t>(primitive.indices)
                >= model.accessors.size())
                return;
            const tinygltf::Accessor& indexAccessor =
                model.accessors[primitive.indices];
            if (indexAccessor.bufferView < 0
                || static_cast<size_t>(indexAccessor.bufferView)
                    >= model.bufferViews.size())
                return;
            const tinygltf::BufferView& indexBufferView =
                model.bufferViews[indexAccessor.bufferView];
            if (indexBufferView.buffer < 0
                || static_cast<size_t>(indexBufferView.buffer)
                    >= model.buffers.size())
                return;
            const tinygltf::Buffer& indexBuffer =
                model.buffers[indexBufferView.buffer];
            const size_t offset =
                indexBufferView.byteOffset + indexAccessor.byteOffset;
            const int componentBytes = tinygltf::GetComponentSizeInBytes(
                indexAccessor.componentType);
            const int stride = indexAccessor.ByteStride(indexBufferView);
            if (offset > indexBuffer.data.size() || componentBytes <= 0
                || stride < componentBytes)
                return;
            const size_t available = indexBuffer.data.size() - offset;
            if (indexAccessor.count != 0
                && (static_cast<size_t>(componentBytes) > available
                    || indexAccessor.count - 1
                        > (available - static_cast<size_t>(componentBytes))
                            / static_cast<size_t>(stride)))
                return;
            const unsigned char* dataPtr =
                indexBuffer.data.data() + offset;
            output.geometry.indices.reserve(indexAccessor.count);
            for (size_t index = 0; index < indexAccessor.count; ++index) {
                const unsigned char* source =
                    dataPtr + index * static_cast<size_t>(stride);
                switch (indexAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    output.geometry.indices.push_back(
                        static_cast<uint32_t>(*source));
                    break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                {
                    uint16_t value{};
                    std::memcpy(&value, source, sizeof(value));
                    output.geometry.indices.push_back(value);
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                {
                    uint32_t value{};
                    std::memcpy(&value, source, sizeof(value));
                    output.geometry.indices.push_back(
                        value);
                    break;
                }
                default:
                    return;
                }
            }
        }

        output.geometry.faces.reserve(output.geometry.indices.size() / 3);
        for (size_t face = 0; face < output.geometry.indices.size() / 3; ++face)
            output.geometry.faces.push_back(Face{0});
        const int material = primitive.material < 0 ? 0 : primitive.material;
        output.materialIndex =
            static_cast<size_t>(material) < globalMaterials.size()
            ? static_cast<size_t>(material) : 0;
        output.meshIndex = location.meshIndex;
        output.name = mesh.name.empty()
            ? fallbackMeshName + "_mesh"
                + std::to_string(location.meshIndex) + "_"
                + std::to_string(location.primitiveIndex)
            : mesh.name;
        output.ready = !output.geometry.vertices.empty()
            && output.geometry.indices.size() >= 3;
    });

    const size_t preparedMeshCount = static_cast<size_t>(std::ranges::count_if(
        prepared, [](const PreparedPrimitive& primitive) {
            return primitive.ready;
        }));
    scene.reserveForImport(preparedMeshCount, globalMaterials.size(),
        model.nodes.size() + preparedMeshCount + 1);

    // Phase 2: publish shared materials and prepared meshes in source order.
    // Registry and Scene mutation stays on this thread; BLAS builds are
    // enqueued only after a complete immutable CPU payload exists.
    std::vector<MaterialRef> globalMaterialRefs;
    globalMaterialRefs.reserve(globalMaterials.size());
    for (const MaterialAuthoring& material : globalMaterials)
        globalMaterialRefs.push_back(
            scene.addMaterial(nr::materialx::documentFromAuthoring(material)));

    std::vector<std::vector<MeshAssetRef>> loadedMeshAssets(model.meshes.size());
    for (PreparedPrimitive& primitive : prepared) {
        if (!primitive.ready)
            continue;
        std::vector<MaterialRef> materials;
        materials.push_back(globalMaterialRefs[primitive.materialIndex]);
        loadedMeshAssets[primitive.meshIndex].push_back(scene.add(MeshAsset(
            scene, std::move(primitive.name), std::move(primitive.geometry),
            std::move(materials))));
    }

    // STEP 2: Import nodes with final world transforms
    // First, traverse the GLTF hierarchy to compute the world transform for every node.
    // Then, create a flat list of SceneObjects, each with its correct world transform.
    // At this stage, all objects are temporarily added to the scene as root objects.

    std::vector<mat4> worldTransforms(model.nodes.size());
    std::vector<SceneObject*> nodeMap(model.nodes.size(), nullptr);

    std::function<void(int, const mat4&)> calculateWorldTransforms =
        [&](int nodeIndex, const mat4& parentWorld) {

        const auto& node = model.nodes[nodeIndex];
        mat4 localTransform;
        if (node.matrix.size() == 16) {
            localTransform = glm::make_mat4(node.matrix.data());
        } else {
            // Decompose TRS and combine. glTF spec is T * R * S.
            vec3 T = node.translation.size() == 3 ? vec3(glm::make_vec3(node.translation.data())) : vec3(0.0f);
            // glTF quat is [x, y, z, w], while GLM is [w, x, y, z]
            quat R = node.rotation.size() == 4 ? quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]), static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2])) : quat(1.0f, 0.0f, 0.0f, 0.0f);
            vec3 S = node.scale.size() == 3 ? vec3(glm::make_vec3(node.scale.data())) : vec3(1.0f);
            localTransform = translate(mat4(1.0f), T) * toMat4(R) * scale(mat4(1.0f), S);
        }
        localTransform = nr::coords::toOpenGlTransform(localTransform, nr::coords::OpenGlSpace);

        worldTransforms[nodeIndex] = parentWorld * localTransform;

        for (int childIndex : node.children) {
            calculateWorldTransforms(childIndex, worldTransforms[nodeIndex]);
        }
    };

    // Start traversal from the root nodes of the default scene
    if (model.defaultScene >= 0) {
        for (int nodeIndex : model.scenes[model.defaultScene].nodes) {
            calculateWorldTransforms(nodeIndex, mat4(1.0f));
        }
    }

    // Create all SceneObjects and attach their meshes
    for (size_t i = 0; i < model.nodes.size(); ++i) {
        const auto& node = model.nodes[i];
        std::string name = node.name.empty() ? ("Node_" + std::to_string(i)) : node.name;

        auto sceneObject = std::make_unique<SceneObject>(scene, name, Transform{worldTransforms[i]});
        SceneObject* objPtr = sceneObject.get();
        nodeMap[i] = objPtr;
        scene.add(std::move(sceneObject)); // Add as a root object for now

        // If the node has a mesh, create instances and attach them as children
        if (node.mesh >= 0 && node.mesh < loadedMeshAssets.size()) {
            // Check for GPU instancing extension (EXT_mesh_gpu_instancing)
            bool hasInstancing = node.extensions.contains("EXT_mesh_gpu_instancing");
            std::vector<mat4> perInstanceTransforms;

            if (hasInstancing) {
                const auto& instExt = node.extensions.at("EXT_mesh_gpu_instancing");
                if (instExt.Has("attributes")) {
                    const auto& attrs = instExt.Get("attributes");
                    int transAcc = attrs.Has("TRANSLATION") ? attrs.Get("TRANSLATION").Get<int>() : -1;
                    int rotAcc   = attrs.Has("ROTATION")    ? attrs.Get("ROTATION").Get<int>()    : -1;
                    int scaleAcc = attrs.Has("SCALE")       ? attrs.Get("SCALE").Get<int>()       : -1;
                    int matrixAcc = attrs.Has("MATRIX")     ? attrs.Get("MATRIX").Get<int>()      : -1;

                    auto readAccData = [&](int idx) -> const float* {
                        if (idx < 0) return nullptr;
                        const auto& acc = model.accessors[idx];
                        const auto& bv = model.bufferViews[acc.bufferView];
                        const auto& buf = model.buffers[bv.buffer];
                        return reinterpret_cast<const float*>(&buf.data[bv.byteOffset + acc.byteOffset]);
                    };

                    size_t instanceCount = 0;
                    if (matrixAcc >= 0)       instanceCount = model.accessors[matrixAcc].count;
                    else if (transAcc >= 0)   instanceCount = model.accessors[transAcc].count;
                    else if (rotAcc >= 0)     instanceCount = model.accessors[rotAcc].count;
                    else if (scaleAcc >= 0)   instanceCount = model.accessors[scaleAcc].count;

                    if (matrixAcc >= 0) {
                        const float* matrices = readAccData(matrixAcc);
                        for (size_t inst = 0; inst < instanceCount; ++inst)
                            perInstanceTransforms.push_back(nr::coords::toOpenGlTransform(glm::make_mat4(&matrices[inst * 16]), nr::coords::OpenGlSpace));
                    } else {
                        const float* translations = readAccData(transAcc);
                        const float* rotations    = readAccData(rotAcc);
                        const float* scales       = readAccData(scaleAcc);
                        for (size_t inst = 0; inst < instanceCount; ++inst) {
                            vec3 T = translations ? glm::make_vec3(&translations[inst * 3]) : vec3(0.0f);
                            quat R = rotations
                                ? quat(rotations[inst * 4 + 3], rotations[inst * 4], rotations[inst * 4 + 1], rotations[inst * 4 + 2])
                                : quat(1.0f, 0.0f, 0.0f, 0.0f);
                            vec3 S = scales ? glm::make_vec3(&scales[inst * 3]) : vec3(1.0f);
                            perInstanceTransforms.push_back(nr::coords::toOpenGlTransform(
                                translate(mat4(1.0f), T) * toMat4(R) * scale(mat4(1.0f), S), nr::coords::OpenGlSpace));
                        }
                    }
                }
            }

            if (!perInstanceTransforms.empty()) {
                // GPU instancing: one MeshInstance per entry
                for (size_t inst = 0; inst < perInstanceTransforms.size(); ++inst) {
                    const mat4 finalWorld = worldTransforms[i] * perInstanceTransforms[inst];
                    for (const MeshAssetRef& meshAsset : loadedMeshAssets[node.mesh]) {
                        auto instance = std::make_unique<MeshInstance>(
                            scene, meshAsset.get()->getName() + "_inst_" + std::to_string(inst), meshAsset, Transform{finalWorld});
                        const SceneObjectHandle instHandle = scene.add(std::move(instance));
                        scene.reparentObject(instHandle, objPtr->getHandle());
                    }
                }
            } else {
                // Single instance with the node's world transform
                for (const MeshAssetRef& meshAsset : loadedMeshAssets[node.mesh]) {
                    auto instance = std::make_unique<MeshInstance>(scene, meshAsset.get()->getName() + "_inst", meshAsset, Transform{worldTransforms[i]});
                    const SceneObjectHandle instHandle = scene.add(std::move(instance));
                    scene.reparentObject(instHandle, objPtr->getHandle());
                }
            }
        }
    }

    // STEP 3: Assemble the final scene hierarchy
    // With all nodes created, build the hierarchy by reparenting each node. The 'reparent'
    // operation should preserve the node's world transform by calculating the correct
    // new local transform relative to its parent.

    auto importRoot = std::make_unique<SceneObject>(scene, nameFromPath(filepath), Transform{});
    const std::string gltfType = lowerPath(filePath.extension().string()) == ".glb" ? "glb" : "gltf";
    importRoot->setSource(gltfType, filePath.string());
    SceneObject* importRootPtr = importRoot.get();
    const SceneObjectHandle rootHandle = scene.add(std::move(importRoot));

    // Reparent nodes to build the glTF hierarchy
    for (size_t i = 0; i < model.nodes.size(); ++i) {
        const auto& node = model.nodes[i];
        if (nodeMap[i] == nullptr) continue;

        for (int childIndex : node.children) {
            if (nodeMap[childIndex])
                scene.reparentObject(nodeMap[childIndex]->getHandle(), nodeMap[i]->getHandle());
        }
    }

    // Attach the scene's original root nodes to our new import object
    if (model.defaultScene >= 0)
        for (int rootNodeIndex : model.scenes[model.defaultScene].nodes)
            if (nodeMap[rootNodeIndex])
                scene.reparentObject(nodeMap[rootNodeIndex]->getHandle(), rootHandle);

    scene.setActiveObject(rootHandle);
    scene.registerImportedFileRoot(resolvedFilepath, rootHandle);
    loadedMeshAssets.clear();
    globalMaterialRefs.clear();
    scene.reclaimUnusedResources();
}

void SceneImporter::ImportObjScene(Scene& scene, const std::string& filepath, const MaterialAuthoring* materialOverride)
{
    const std::filesystem::path filePath = resolveAssetPath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    // The directory containing the .obj file, used for finding .mtl and textures
    const std::filesystem::path objDir = filePath.has_parent_path() ? filePath.parent_path() : ".";
    const std::string resolvedFilepath = filePath.string();

    // Same rationale as ImportGltfScene's cache: a repeat import of the same
    // path clones the previously built hierarchy instead of re-parsing the
    // OBJ and re-uploading its meshes/textures. Skipped when a material
    // override is given, since the override isn't part of the cache key and
    // could otherwise apply the wrong material to a cloned hierarchy built
    // under a different (or no) override.
    if (materialOverride == nullptr) {
        if (const SceneObjectHandle cached = scene.findImportedFileRoot(resolvedFilepath);
            cached.isValid())
        {
            if (SceneObject* cachedRoot = scene.getObject(cached)) {
                const auto clone = scene.cloneHierarchy(cachedRoot);
                scene.setActiveObject(clone->getHandle());
                return;
            }
        }
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, resolvedFilepath.c_str(), objDir.string().c_str(), true))
        throw std::runtime_error("Failed to load OBJ file: " + warn + err);

    if (!warn.empty())
       LOG_ERROR("TinyObjLoader Warning: " << warn);
    if (!err.empty())
       LOG_ERROR("TinyObjLoader Info/Error: " << err);

    // Load Global Materials from the MTL file (if it was found)
    std::vector<MaterialAuthoring> globalMaterials;
    // The materials below carry scene texture slot indices. Scene owns the
    // corresponding image data for the lifetime of this scene.
    // Keyed by (resolved texture path, encoding) so multiple materials that
    // reuse the same texture file upload it once instead of decoding and
    // uploading a duplicate copy per material.
    std::map<std::pair<std::string, int>, TextureHandle> imageTextureCache;
    for (const auto& mat : mats) {
        MaterialAuthoring material{};
        material.albedo = vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        material.specular = mat.specular[0];
        material.metallic = mat.metallic;
        material.roughness = mat.roughness;
        material.sellmeier = constantIorSellmeier(mat.ior);
        material.transmissionColor = vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        material.transmission = (mat.illum == 4 || mat.illum == 7) ? 1.0f : 0.0f;
        material.opacity = mat.dissolve;
        material.emission = vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        material.emissionStrength = (length2(material.emission) > 1e-6f) ? 1.0f : 0.0f;

        // Lambda to safely find and load a texture
        auto addTexture = [&](const std::string& texname, int& index, TextureEncoding encoding) {
            if (!texname.empty()) {
                // Images named *_linear contain already-linear data; skip sRGB decode.
                if (encoding == TextureEncoding::Srgb8 &&
                    texname.find("_linear") != std::string::npos)
                    encoding = TextureEncoding::Linear8;
                const std::filesystem::path texturePath = objDir / texname;
                const std::string resolvedTexturePath = texturePath.string();
                const std::pair<std::string, int> cacheKey{
                    resolvedTexturePath, static_cast<int>(encoding)};
                if (const auto cached = imageTextureCache.find(cacheKey);
                    cached != imageTextureCache.end())
                {
                    index = static_cast<int>(cached->second.index());
                    return;
                }
                if (std::filesystem::exists(texturePath)) {
                    const TextureHandle texture = scene.addTexture(
                        Texture(resolvedTexturePath, encoding));
                    index = static_cast<int>(texture.index());
                    imageTextureCache[cacheKey] = texture;
                } else
                   LOG_ERROR("Warning: Texture file not found: " << texturePath.string());
            }
        };

        addTexture(mat.diffuse_texname, material.albedoIndex,
                   TextureEncoding::Srgb8);
        addTexture(mat.specular_texname, material.specularIndex,
                   TextureEncoding::Srgb8);
        addTexture(mat.roughness_texname, material.roughnessIndex,
                   TextureEncoding::Linear8);
        addTexture(mat.normal_texname, material.normalIndex,
                   TextureEncoding::Linear8);
        addTexture(mat.alpha_texname, material.opacityIndex,
                   TextureEncoding::Linear8);
        addTexture(mat.emissive_texname, material.emissionIndex,
                   TextureEncoding::Srgb8);

        globalMaterials.push_back(material);
    }

    if (globalMaterials.empty())
        globalMaterials.emplace_back(); // Adds a default-constructed Material if the .mtl was missing

    if (materialOverride != nullptr)
        globalMaterials.assign(1, *materialOverride);

    struct PreparedObjShape
    {
        MeshGeometry geometry;
        std::vector<size_t> globalMaterialIndices;
        std::string name;
        vec3 center{};
        bool ready{};
    };

    // Phase 1: every worker reads immutable tinyobj arrays and writes only its
    // own final managed payload. Material IDs are kept as indices into the
    // shared global material table; Scene references are created later.
    const std::string parentName = nameFromPath(filepath);
    std::vector<PreparedObjShape> prepared(shapes.size());
    boundedParallelFor(shapes.size(), [&](const size_t shapeIndex) {
        const tinyobj::shape_t& shape = shapes[shapeIndex];
        PreparedObjShape& output = prepared[shapeIndex];
        if (shape.mesh.indices.empty())
            return;

        output.geometry.vertices.reserve(shape.mesh.indices.size());
        output.geometry.indices.reserve(shape.mesh.indices.size());
        output.geometry.faces.reserve(shape.mesh.num_face_vertices.size());

        std::unordered_map<int, int> materialRemap;
        auto localMaterialIndex = [&](const int globalMaterialId) {
            const int sanitized = globalMaterialId < 0
                || static_cast<size_t>(globalMaterialId)
                    >= globalMaterials.size()
                ? 0 : globalMaterialId;
            if (const auto found = materialRemap.find(sanitized);
                found != materialRemap.end())
                return found->second;
            const int local = static_cast<int>(
                output.globalMaterialIndices.size());
            output.globalMaterialIndices.push_back(
                static_cast<size_t>(sanitized));
            materialRemap.emplace(sanitized, local);
            return local;
        };

        size_t indexOffset = 0;
        for (size_t faceIndex = 0;
            faceIndex < shape.mesh.num_face_vertices.size(); ++faceIndex) {
            const unsigned int vertexCount =
                shape.mesh.num_face_vertices[faceIndex];
            if (vertexCount != 3
                || indexOffset + vertexCount > shape.mesh.indices.size()) {
                indexOffset += vertexCount;
                continue;
            }

            const int sourceMaterial =
                faceIndex < shape.mesh.material_ids.size()
                ? shape.mesh.material_ids[faceIndex] : -1;
            const Face face{localMaterialIndex(sourceMaterial)};

            uint32_t triangleIndices[3];
            bool validTriangle = true;
            Vertex triangleVertices[3]{};
            for (unsigned int corner = 0; corner < 3; ++corner) {
                const tinyobj::index_t& sourceIndex =
                    shape.mesh.indices[indexOffset + corner];
                if (sourceIndex.vertex_index < 0
                    || static_cast<size_t>(
                        sourceIndex.vertex_index * 3 + 2)
                        >= attrib.vertices.size()) {
                    validTriangle = false;
                    break;
                }
                Vertex& vertex = triangleVertices[corner];
                vertex.position = nr::coords::toOpenGlVector({
                    attrib.vertices[3 * sourceIndex.vertex_index],
                    attrib.vertices[3 * sourceIndex.vertex_index + 1],
                    attrib.vertices[3 * sourceIndex.vertex_index + 2],
                }, nr::coords::OpenGlSpace);
                vertex.normal = vec3(0.0f, 1.0f, 0.0f);
                if (sourceIndex.normal_index >= 0
                    && static_cast<size_t>(
                        sourceIndex.normal_index * 3 + 2)
                        < attrib.normals.size())
                    vertex.normal = nr::coords::toOpenGlVector({
                        attrib.normals[3 * sourceIndex.normal_index],
                        attrib.normals[3 * sourceIndex.normal_index + 1],
                        attrib.normals[3 * sourceIndex.normal_index + 2],
                    }, nr::coords::OpenGlSpace);
                vertex.uv = vec2(0.0f);
                if (sourceIndex.texcoord_index >= 0
                    && static_cast<size_t>(
                        sourceIndex.texcoord_index * 2 + 1)
                        < attrib.texcoords.size())
                    vertex.uv = {
                        attrib.texcoords[2 * sourceIndex.texcoord_index],
                        1.0f - attrib.texcoords[
                            2 * sourceIndex.texcoord_index + 1],
                    };
                vertex.tangent = vec3(0.0f);
                if (static_cast<size_t>(
                        sourceIndex.vertex_index * 3 + 2)
                    < attrib.colors.size())
                    vertex.color = nr::vertex_color::packLinear(glm::vec4(
                        attrib.colors[3 * sourceIndex.vertex_index],
                        attrib.colors[3 * sourceIndex.vertex_index + 1],
                        attrib.colors[3 * sourceIndex.vertex_index + 2],
                        1.0f));
            }
            indexOffset += vertexCount;
            if (!validTriangle)
                continue;

            const vec3 edge1 =
                triangleVertices[1].position - triangleVertices[0].position;
            const vec3 edge2 =
                triangleVertices[2].position - triangleVertices[0].position;
            const vec2 deltaUv1 =
                triangleVertices[1].uv - triangleVertices[0].uv;
            const vec2 deltaUv2 =
                triangleVertices[2].uv - triangleVertices[0].uv;
            const float determinant =
                deltaUv1.x * deltaUv2.y - deltaUv2.x * deltaUv1.y;
            const vec3 tangent = std::fabs(determinant) < 1e-8f
                ? vec3(1.0f, 0.0f, 0.0f)
                : (1.0f / determinant)
                    * (deltaUv2.y * edge1 - deltaUv1.y * edge2);

            for (unsigned int corner = 0; corner < 3; ++corner) {
                Vertex& vertex = triangleVertices[corner];
                vertex.tangent += tangent;
                if (length2(vertex.tangent) > 1e-8f)
                    vertex.tangent = normalize(vertex.tangent
                        - vertex.normal
                            * dot(vertex.normal, vertex.tangent));
                else {
                    const vec3 first =
                        cross(vertex.normal, vec3(0.0f, 0.0f, 1.0f));
                    const vec3 second =
                        cross(vertex.normal, vec3(0.0f, 1.0f, 0.0f));
                    vertex.tangent = normalize(
                        length2(first) > length2(second) ? first : second);
                }
                vertex.tangentSign = 1.0f;
                triangleIndices[corner] = static_cast<uint32_t>(
                    output.geometry.vertices.size());
                output.geometry.vertices.push_back(vertex);
                output.geometry.indices.push_back(triangleIndices[corner]);
            }
            output.geometry.faces.push_back(face);
        }

        if (output.geometry.vertices.empty())
            return;
        vec3 minimum = output.geometry.vertices.front().position;
        vec3 maximum = minimum;
        for (const Vertex& vertex : output.geometry.vertices) {
            minimum = min(minimum, vertex.position);
            maximum = max(maximum, vertex.position);
        }
        output.center = (minimum + maximum) * 0.5f;
        for (Vertex& vertex : output.geometry.vertices)
            vertex.position -= output.center;
        output.name = shape.name.empty()
            ? parentName + "_shape" : shape.name;
        output.ready = true;
    });

    const size_t preparedMeshCount = static_cast<size_t>(std::ranges::count_if(
        prepared, [](const PreparedObjShape& shape) { return shape.ready; }));
    scene.reserveForImport(preparedMeshCount, globalMaterials.size(),
        preparedMeshCount + 1);

    std::vector<MaterialRef> globalMaterialRefs;
    globalMaterialRefs.reserve(globalMaterials.size());
    for (const MaterialAuthoring& material : globalMaterials)
        globalMaterialRefs.push_back(
            scene.addMaterial(nr::materialx::documentFromAuthoring(material)));

    // Phase 2: publish in OBJ shape order and assemble the hierarchy serially.
    auto parentObject =
        std::make_unique<SceneObject>(scene, parentName, Transform{});
    parentObject->setSource("obj", filePath.string());
    const SceneObjectHandle parentHandle = scene.add(std::move(parentObject));

    for (PreparedObjShape& shape : prepared) {
        if (!shape.ready)
            continue;
        std::vector<MaterialRef> localMaterials;
        localMaterials.reserve(shape.globalMaterialIndices.size());
        for (const size_t materialIndex : shape.globalMaterialIndices)
            localMaterials.push_back(globalMaterialRefs[materialIndex]);

        const std::string instanceName = shape.name;
        const MeshAssetRef meshAsset = scene.add(MeshAsset(scene,
            std::move(shape.name), std::move(shape.geometry),
            std::move(localMaterials)), materialOverride == nullptr);
        Transform transform;
        transform.setPosition(shape.center);
        auto instance = std::make_unique<MeshInstance>(
            scene, instanceName, meshAsset, transform);
        const SceneObjectHandle instanceHandle =
            scene.add(std::move(instance));
        scene.reparentObject(instanceHandle, parentHandle);
    }

    scene.setActiveObject(parentHandle);
    if (materialOverride == nullptr)
        scene.registerImportedFileRoot(resolvedFilepath, parentHandle);
    globalMaterialRefs.clear();
    scene.reclaimUnusedResources();
}

std::string SceneImporter::nameFromPath(const std::string& path) {
    const size_t lastSlash = path.find_last_of("/\\");
    std::string name = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;

    if (const size_t lastDot = name.find_last_of('.'); lastDot != std::string::npos)
        name = name.substr(0, lastDot);

    return name;
}

std::vector<char> SceneImporter::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    const size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize> (fileSize));
    file.close();

    return buffer;
}

// ── JSON scene ────────────────────────────────────────────────────────────────

void SceneImporter::ImportJsonScene(Scene& scene, const std::string& filepath)
{
    SceneReader::Read(scene, filepath);
}

bool SceneImporter::IsGaussianFile(const std::string& filepath)
{
    const std::string path = lowerPath(filepath);
    return path.ends_with(".ply")
        || path.ends_with(".compressed.ply")
        || path.ends_with(".splat")
        || path.ends_with(".ksplat")
        || path.ends_with(".spz")
        || path.ends_with(".sog");
}

bool SceneImporter::IsSceneFile(const std::string& filepath)
{
    return lowerPath(filepath).ends_with(".nrscene") || nr::sceneio::isUsdFile(filepath);
}

void SceneImporter::ImportFile(Scene& scene, const std::string& filepath)
{
    const std::string path = lowerPath(filepath);

    if (path.ends_with(".gltf") || path.ends_with(".glb")) {
        ImportGltfScene(scene, filepath);
    } else if (path.ends_with(".pbrt")) {
        ImportPbrtScene(scene, filepath);
    } else if (path.ends_with(".obj")) {
        ImportObjScene(scene, filepath);
    } else if (IsGaussianFile(filepath)) {
        ImportGaussianScene(scene, filepath);
    } else if (path.ends_with(".png") || path.ends_with(".jpg") || path.ends_with(".jpeg") ||
               path.ends_with(".bmp") || path.ends_with(".tga") || path.ends_with(".psd") ||
               path.ends_with(".gif") || path.ends_with(".hdr") || path.ends_with(".exr") ||
               path.ends_with(".pic")) {
        // Standalone images have no material reference to retain them. Keep
        // an explicit scene-library owner so they remain available to the
        // HDRI picker after this importer returns.
        scene.addTexture(Texture(filepath));
    } else {
        throw std::runtime_error("Unsupported import file type: " + filepath);
    }
}

void SceneImporter::ImportGaussianScene(Scene& scene, const std::string& filepath)
{
    const std::filesystem::path filePath = resolveAssetPath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    const std::string name = nameFromPath(filePath.filename().string());
    const GaussianAssetRef asset =
        scene.add(GaussianAsset::CreateFromFile(scene, name, filePath.string()));
    auto instance = std::make_unique<GaussianInstance>(scene, name, asset, Transform{});
    instance->setSource("gaussian", filePath.string());
    scene.add(std::move(instance));
}
