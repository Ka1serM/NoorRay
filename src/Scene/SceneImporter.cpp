#include "SceneImporter.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <glaze/glaze.hpp>
#include "Camera/CameraInstance.h"
#include "CUDA/rstd/Allocator.h"
#include "Vulkan/Texture.h"
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
#include "MeshInstance.h"
#include "glm/gtx/matrix_decompose.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "Mesh/MeshAsset.h"
#include "Mesh/Transform.h"
#include <functional>

//(x, y, z) -> (x, -y, -z)

namespace {
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
}

void SceneImporter::ImportGltfScene(Scene& scene, const std::string& filepath)
{
    // Boilerplate: Load GLTF file from disk
    const std::filesystem::path filePath = resolveAssetPath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    const std::filesystem::path gltfDir = filePath.parent_path();
    const std::string resolvedFilepath = filePath.string();
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
    std::vector<Material> globalMaterials;
    for (const auto& mat : model.materials) {
        Material material{};
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

        auto addTexture = [&](int textureIndex, int& materialIndex, vk::Format fmt) {
            if (textureIndex < 0) return;
            const tinygltf::Texture& tex = model.textures[textureIndex];
            if (tex.source < 0) return;
            const tinygltf::Image& image = model.images[tex.source];

            // Images named *_linear contain already-linear data; skip sRGB decode.
            if (fmt == vk::Format::eR8G8B8A8Srgb &&
                (image.name.find("_linear") != std::string::npos ||
                 image.uri.find("_linear")   != std::string::npos))
                fmt = vk::Format::eR8G8B8A8Unorm;

            if (!image.uri.empty()) {
                const std::filesystem::path texturePath = gltfDir / image.uri;
                if (std::filesystem::exists(texturePath)) {
                    scene.add(Texture(scene.getContext(), texturePath.string(), fmt));
                    materialIndex = static_cast<int>(scene.getTextures().size() - 1);
                } else {
                    LOG_ERROR("Warning: Texture file not found: " << texturePath.string());
                }
            } else if (!image.image.empty()) {
                const std::string texName = image.name.empty()
                    ? "texture_" + std::to_string(textureIndex)
                    : image.name;
                scene.add(Texture(scene.getContext(), texName,
                    image.image.data(), image.width, image.height, fmt));
                materialIndex = static_cast<int>(scene.getTextures().size() - 1);
            } else {
                LOG_ERROR("Warning: Embedded texture has no decoded data");
            }
        };

        addTexture(pbr.baseColorTexture.index, material.albedoIndex,
                   vk::Format::eR8G8B8A8Srgb);
        addTexture(pbr.metallicRoughnessTexture.index, material.roughnessIndex,
                   vk::Format::eR8G8B8A8Unorm);
        addTexture(mat.normalTexture.index, material.normalIndex,
                   vk::Format::eR8G8B8A8Unorm);
        addTexture(mat.emissiveTexture.index, material.emissionIndex,
                   vk::Format::eR8G8B8A8Srgb);

        globalMaterials.push_back(material);
    }
    if (globalMaterials.empty())
        globalMaterials.emplace_back(); // Default material

    // Load Meshes
    nr::rstd::vector<nr::rstd::vector<uint32_t>> loadedMeshAssets(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i) {
        const auto& mesh = model.meshes[i];
        for (size_t j = 0; j < mesh.primitives.size(); ++j) {
            const auto& primitive = mesh.primitives[j];
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            const float* positions = nullptr;
            const float* normals = nullptr;
            const float* tangents = nullptr;
            const float* texcoords = nullptr;
            size_t vertexCount = 0;

            auto getBuffer = [&](const char* attribute) -> const float* {
                const auto it = primitive.attributes.find(attribute);
                if (it == primitive.attributes.end()) return nullptr;
                const auto& accessor = model.accessors[it->second];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                if (vertexCount == 0) vertexCount = accessor.count;
                return reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            };

            positions = getBuffer("POSITION");
            if (!positions) continue;
            normals = getBuffer("NORMAL");
            tangents = getBuffer("TANGENT");
            texcoords = getBuffer("TEXCOORD_0");

            vertices.resize(vertexCount);
            for (size_t v = 0; v < vertexCount; ++v) {
                vertices[v].position    = glm::make_vec3(&positions[v * 3]);
                vertices[v].normal      = normals ? normalize(glm::make_vec3(&normals[v * 3])) : vec3(0, 1, 0);
                vertices[v].uv          = texcoords ? vec2(texcoords[v * 2], 1.0f - texcoords[v * 2 + 1]) : vec2(0);
                vertices[v].tangent     = tangents ? normalize(vec3(glm::make_vec3(&tangents[v * 4]))) : vec3(1, 0, 0);
                vertices[v].tangentSign = tangents ? tangents[v * 4 + 3] : 1.0f;
            }

            const auto& indexAccessor = model.accessors[primitive.indices];
            indices.reserve(indexAccessor.count);
            const auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const auto& indexBuffer = model.buffers[indexBufferView.buffer];
            const void* dataPtr = &(indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);

            switch (indexAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  for (size_t k = 0; k < indexAccessor.count; ++k) indices.push_back(static_cast<const uint8_t*>(dataPtr)[k]); break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: for (size_t k = 0; k < indexAccessor.count; ++k) indices.push_back(static_cast<const uint16_t*>(dataPtr)[k]); break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   for (size_t k = 0; k < indexAccessor.count; ++k) indices.push_back(static_cast<const uint32_t*>(dataPtr)[k]); break;
                default: continue;
            }

            int materialID = (primitive.material < 0) ? 0 : primitive.material;
            std::string meshName = mesh.name.empty() ? (nameFromPath(filepath) + "_mesh" + std::to_string(i) + "_" + std::to_string(j)) : mesh.name;

            const uint32_t meshIndex = scene.add(MeshAsset(scene, meshName, vertices, indices,
                std::vector<Face>(indices.size() / 3), std::vector<Material>{globalMaterials[materialID]}));
            loadedMeshAssets[i].push_back(meshIndex);
        }
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
                            perInstanceTransforms.push_back(glm::make_mat4(&matrices[inst * 16]));
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
                            perInstanceTransforms.push_back(translate(mat4(1.0f), T) * toMat4(R) * scale(mat4(1.0f), S));
                        }
                    }
                }
            }

            if (!perInstanceTransforms.empty()) {
                // GPU instancing: one MeshInstance per entry
                for (size_t inst = 0; inst < perInstanceTransforms.size(); ++inst) {
                    const mat4 finalWorld = worldTransforms[i] * perInstanceTransforms[inst];
                    for (const uint32_t meshIndex : loadedMeshAssets[node.mesh]) {
                        const MeshAsset& asset = scene.getMeshAsset(meshIndex);
                        auto instance = std::make_unique<MeshInstance>(
                            scene, asset.getName() + "_inst_" + std::to_string(inst), meshIndex, Transform{finalWorld});
                        const uint64_t instId = scene.add(std::move(instance));
                        scene.reparentObject(instId, objPtr->getId());
                    }
                }
            } else {
                // Single instance with the node's world transform
                for (const uint32_t meshIndex : loadedMeshAssets[node.mesh]) {
                    const MeshAsset& asset = scene.getMeshAsset(meshIndex);
                    auto instance = std::make_unique<MeshInstance>(scene, asset.getName() + "_inst", meshIndex, Transform{worldTransforms[i]});
                    const uint64_t instId = scene.add(std::move(instance));
                    scene.reparentObject(instId, objPtr->getId());
                }
            }
        }
    }

    // STEP 3: Assemble the final scene hierarchy
    // With all nodes created, build the hierarchy by reparenting each node. The 'reparent'
    // operation should preserve the node's world transform by calculating the correct
    // new local transform relative to its parent.

    auto importRoot = std::make_unique<SceneObject>(scene, nameFromPath(filepath), Transform{});
    SceneObject* importRootPtr = importRoot.get();
    const uint64_t rootId = scene.add(std::move(importRoot));

    // Reparent nodes to build the glTF hierarchy
    for (size_t i = 0; i < model.nodes.size(); ++i) {
        const auto& node = model.nodes[i];
        if (nodeMap[i] == nullptr) continue;

        for (int childIndex : node.children) {
            if (nodeMap[childIndex])
                scene.reparentObject(nodeMap[childIndex]->getId(), nodeMap[i]->getId());
        }
    }

    // Attach the scene's original root nodes to our new import object
    if (model.defaultScene >= 0)
        for (int rootNodeIndex : model.scenes[model.defaultScene].nodes)
            if (nodeMap[rootNodeIndex])
                scene.reparentObject(nodeMap[rootNodeIndex]->getId(), rootId);

    scene.setActiveObjectId(rootId);
}

void SceneImporter::ImportObjScene(Scene& scene, const std::string& filepath, const Material* materialOverride)
{
    const std::filesystem::path filePath = resolveAssetPath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    // The directory containing the .obj file, used for finding .mtl and textures
    const std::filesystem::path objDir = filePath.has_parent_path() ? filePath.parent_path() : ".";
    const std::string resolvedFilepath = filePath.string();

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
    std::vector<Material> globalMaterials;
    for (const auto& mat : mats) {
        Material material{};
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
        auto addTexture = [&](const std::string& texname, int& index, vk::Format fmt) {
            if (!texname.empty()) {
                // Images named *_linear contain already-linear data; skip sRGB decode.
                if (fmt == vk::Format::eR8G8B8A8Srgb &&
                    texname.find("_linear") != std::string::npos)
                    fmt = vk::Format::eR8G8B8A8Unorm;
                const std::filesystem::path texturePath = objDir / texname;
                if (std::filesystem::exists(texturePath)) {
                    scene.add(Texture(scene.getContext(), texturePath.string(), fmt));
                    index = static_cast<int>(scene.getTextures().size() - 1);
                } else
                   LOG_ERROR("Warning: Texture file not found: " << texturePath.string());
            }
        };

        addTexture(mat.diffuse_texname, material.albedoIndex,
                   vk::Format::eR8G8B8A8Srgb);
        addTexture(mat.specular_texname, material.specularIndex,
                   vk::Format::eR8G8B8A8Srgb);
        addTexture(mat.roughness_texname, material.roughnessIndex,
                   vk::Format::eR8G8B8A8Unorm);
        addTexture(mat.normal_texname, material.normalIndex,
                   vk::Format::eR8G8B8A8Unorm);
        addTexture(mat.alpha_texname, material.opacityIndex,
                   vk::Format::eR8G8B8A8Unorm);
        addTexture(mat.emissive_texname, material.emissionIndex,
                   vk::Format::eR8G8B8A8Srgb);

        globalMaterials.push_back(material);
    }

    if (globalMaterials.empty())
        globalMaterials.emplace_back(); // Adds a default-constructed Material if the .mtl was missing

    if (materialOverride != nullptr)
        for (auto& mat : globalMaterials)
            mat = *materialOverride;

    // Create a parent object for this entire OBJ file to keep the scene organized
    std::string parentName = nameFromPath(filepath);
    auto parentObject = std::make_unique<SceneObject>(scene, parentName, Transform{});
    const uint64_t parentId = scene.add(std::move(parentObject));

    for (const auto& shape : shapes) {
        if (shape.mesh.indices.empty())
            continue; // Skip empty shapes that have no geometry data.

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<Face> faces;

        // Collect and remap materials used ONLY by this shape
        std::unordered_map<int, int> matRemap; // Global material index -> Local material index
        std::vector<Material> localMaterials;

        // **Robustness Step 2**: Build a local material list for this shape,
        // safely handling any invalid material IDs from the OBJ data.
        auto getLocalMaterialIndex = [&](int globalMatId) {
            // Sanitize: if ID is out of bounds or negative, map it to the first global material (our default).
            const int sanitizedGlobalId = (globalMatId < 0 || static_cast<size_t>(globalMatId) >= globalMaterials.size()) ? 0 : globalMatId;

            if (const auto it = matRemap.find(sanitizedGlobalId); it != matRemap.end())
                return it->second;

            int localIndex = static_cast<int>(localMaterials.size());
            localMaterials.push_back(globalMaterials[sanitizedGlobalId]);
            matRemap[sanitizedGlobalId] = localIndex;
            return localIndex;
        };

        // Load vertex and face data for the current shape
        size_t indexOffset = 0;
        for (size_t faceIndex = 0; faceIndex < shape.mesh.num_face_vertices.size(); ++faceIndex) {
            // Because we enabled triangulation, 'fv' will always be 3.
            const unsigned int fv = shape.mesh.num_face_vertices[faceIndex];

            Face face{};
            face.materialIndex = getLocalMaterialIndex(shape.mesh.material_ids[faceIndex]);

            uint32_t triIndices[3];
            for (unsigned int v = 0; v < fv; ++v) {
                const auto& idx = shape.mesh.indices[indexOffset + v];
                Vertex vertex{};

                // Position with Y/Z inversion (coordinate system conversion)
                vertex.position = vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    -attrib.vertices[3 * idx.vertex_index + 1],
                    -attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Normal, checking for existence
                if (idx.normal_index >= 0) {
                    vertex.normal = vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        -attrib.normals[3 * idx.normal_index + 1],
                        -attrib.normals[3 * idx.normal_index + 2]
                    );
                } else
                    vertex.normal = vec3(0.0f, 1.0f, 0.0f); // Default normal

                // UV, checking for existence
                if (idx.texcoord_index >= 0) {
                    vertex.uv = vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1] // Flip V coordinate for many formats
                    );
                } else
                    vertex.uv = vec2(0.0f); // Default UV

                vertices.push_back(vertex);
                triIndices[v] = static_cast<uint32_t>(vertices.size() - 1);
                indices.push_back(triIndices[v]);
            }

            // Accumulate tangent data for later averaging
            Vertex& v0 = vertices[triIndices[0]];
            Vertex& v1 = vertices[triIndices[1]];
            Vertex& v2 = vertices[triIndices[2]];

            vec3 edge1 = v1.position - v0.position;
            vec3 edge2 = v2.position - v0.position;
            vec2 deltaUV1 = v1.uv - v0.uv;
            vec2 deltaUV2 = v2.uv - v0.uv;

            float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
            vec3 tangent = (std::fabs(f) < 1e-8f)  ? vec3(1.0f, 0.0f, 0.0f)  : (1.0f / f) * (deltaUV2.y * edge1 - deltaUV1.y * edge2);

            v0.tangent += tangent;
            v1.tangent += tangent;
            v2.tangent += tangent;

            faces.push_back(face);
            indexOffset += fv;
        }

        // Finalize vertices by normalizing the accumulated tangents
        for (auto& v : vertices) {
            if (length2(v.tangent) > 1e-8f) {
                v.tangent = normalize(v.tangent - v.normal * dot(v.normal, v.tangent));
            } else {
                vec3 t1 = cross(v.normal, vec3(0.0, 0.0, 1.0));
                vec3 t2 = cross(v.normal, vec3(0.0, 1.0, 0.0));
                v.tangent = normalize((length2(t1) > length2(t2)) ? t1 : t2);
            }
            v.tangentSign = 1.0f;
        }

        // Compute mesh pivot and center the vertex positions
        vec3 minPos = vertices[0].position;
        vec3 maxPos = vertices[0].position;
        for (const auto& v : vertices) {
            minPos = min(minPos, v.position);
            maxPos = max(maxPos, v.position);
        }
        vec3 center = (minPos + maxPos) * 0.5f;

        for (auto& v : vertices)
            v.position -= center;

        // Create the final assets and instances for this shape
        std::string meshName = shape.name.empty() ? (parentName + "_shape") : shape.name;
        const uint32_t meshIndex = scene.add(MeshAsset(scene, meshName, vertices, indices, faces, localMaterials));

        Transform transform;
        transform.setPosition(center);
        auto instance = std::make_unique<MeshInstance>(scene, meshName, meshIndex, transform);
        const uint64_t instanceId = scene.add(std::move(instance));
        scene.reparentObject(instanceId, parentId);
    }

    scene.setActiveObjectId(parentId);
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
    glz::generic j{};
    std::string buf;
    if (const auto err = glz::read_file_json(j, filepath, buf))
        throw std::runtime_error("Failed to parse scene JSON: " + glz::format_error(err, buf));

    // Safe accessors
    static const glz::generic kNull{};
    auto at = [&](const glz::generic& obj, std::string_view key) -> const glz::generic& {
        return obj.contains(key) ? obj[key] : kNull;
    };
    auto jfloat = [](const glz::generic& v, float def = 0.f) {
        return v.is_number() ? v.as<float>() : def;
    };
    auto jint = [](const glz::generic& v, int def = 0) {
        return v.is_number() ? v.as<int>() : def;
    };
    auto jbool = [](const glz::generic& v, bool def = false) {
        return v.is_boolean() ? v.get<bool>() : def;
    };
    auto jstr = [](const glz::generic& v, std::string_view def = "") -> std::string {
        return v.is_string() ? v.get<std::string>() : std::string(def);
    };
    auto jvec3 = [&](const glz::generic& v, vec3 def = {}) -> vec3 {
        if (!v.is_array()) return def;
        const auto& a = v.get_array();
        return { a.size() > 0 ? jfloat(a[0], def.x) : def.x,
                 a.size() > 1 ? jfloat(a[1], def.y) : def.y,
                 a.size() > 2 ? jfloat(a[2], def.z) : def.z };
    };

    if (j.contains("environment")) {
        const auto& jenv = j["environment"];
        Environment& environment = scene.getEnvironment();
        environment.color = jvec3(at(jenv, "color"), {1.0f, 1.0f, 1.0f});
        environment.lightingExposure = jfloat(at(jenv, "lighting_exposure"), 1.0f);
        environment.visibleExposure = jfloat(at(jenv, "visible_exposure"), 0.0f);
        environment.visible = jbool(at(jenv, "visible"), true) ? 1 : 0;
        environment.updateDerivedSettings();
    }

    // Camera
    if (j.contains("camera")) {
        const auto& jcam = j["camera"];
        const std::string camType = jstr(at(jcam, "type"), "perspective");
        const Transform camTransform{
            jvec3(at(jcam, "position")),
            jvec3(at(jcam, "rotation_euler")),
            jvec3(at(jcam, "scale"), {1, 1, 1})
        };
        const float focalLength = jfloat(at(jcam, "focal_length"), 50.f);
        const float focusDist   = jfloat(at(jcam, "focus_distance"), 2.f);
        const float bokehBias   = jfloat(at(jcam, "bokeh_bias"), 1.f);
        const float aperture    = jfloat(at(jcam, "aperture_diameter"), 0.f);
        const bool disableAntiAliasing = jbool(
            at(jcam, "disable_anti_aliasing"), false);

        Camera cam;
        if (camType == "realistic") {
            nr::rstd::allocator<RealisticCamera> alc;
            RealisticCamera* rc = alc.allocate(1);
            alc.construct(rc);
            rc->apertureDiameterMm = aperture;
            cam = Camera(rc);
        } else if (camType == "thinlens") {
            nr::rstd::allocator<ThinLensCamera> alc;
            ThinLensCamera* tc = alc.allocate(1);
            alc.construct(tc);
            tc->fStop = aperture;
            cam = Camera(tc);
        } else if (camType == "fisheye") {
            nr::rstd::allocator<FisheyeCamera> alc;
            FisheyeCamera* fc = alc.allocate(1);
            alc.construct(fc);
            fc->fStop = aperture;
            cam = Camera(fc);
        } else if (camType == "orthographic") {
            nr::rstd::allocator<OrthographicCamera> alc;
            OrthographicCamera* oc = alc.allocate(1);
            alc.construct(oc);
            cam = Camera(oc);
        } else {
            nr::rstd::allocator<PerspectiveCamera> alc;
            PerspectiveCamera* pc = alc.allocate(1);
            alc.construct(pc);
            cam = Camera(pc);
        }

        cam.setFocalLength(focalLength);
        cam.setFocusDistance(focusDist);
        cam.setAntiAliasingDisabled(disableAntiAliasing);
        if (at(jcam, "resolution").is_array()) {
            const auto& resolution = at(jcam, "resolution").get_array();
            if (resolution.size() >= 2)
                cam.getSensor().setResolution(
                    static_cast<uint32_t>(std::max(jint(resolution[0]), 1)),
                    static_cast<uint32_t>(std::max(jint(resolution[1]), 1)));
        }
        if (auto* thinLens = cam.CastOrNullptr<ThinLensCamera>())
            thinLens->bokehBias = std::max(0.001f, bokehBias);
        else if (auto* fisheye = cam.CastOrNullptr<FisheyeCamera>())
            fisheye->bokehBias = std::max(0.001f, bokehBias);

        auto camInst = std::make_unique<CameraInstance>(scene, "Camera", camTransform, cam);
        if (camType == "realistic") {
            const std::string lens   = jstr(at(jcam, "lens"));
            const std::string sensor = jstr(at(jcam, "sensor"));
            const std::string glass  = jstr(at(jcam, "glass_catalogs"));
            if (!lens.empty() && !sensor.empty())
                camInst->loadRealisticLens(lens, sensor, glass);
            if (aperture > 0.f)
                camInst->setApertureDiameter(aperture);
        }
        scene.add(std::move(camInst));
    }

    // Objects
    if (!j.contains("objects") || !j["objects"].is_array()) return;
    for (const auto& obj : j["objects"].get_array()) {
        const std::string type = jstr(at(obj, "type"), "sphere");
        const std::string name = jstr(at(obj, "name"), "Object");
        Transform t{ jvec3(at(obj, "position")),
                     jvec3(at(obj, "rotation_euler")),
                     jvec3(at(obj, "scale"), {1, 1, 1}) };

        if (type == "gltf" || type == "glb") {
            ImportGltfScene(scene, jstr(at(obj, "path")));
            if (SceneObject* root = scene.getActiveObject())
                root->setLocalTransform(t);
        } else if (type == "obj") {
            Material matOverride{};
            const bool hasMaterial = obj.contains("material");
            if (hasMaterial) {
                const auto& md = at(obj, "material");
                matOverride.albedo           = jvec3(at(md, "albedo"),          {0.8f, 0.8f, 0.8f});
                matOverride.roughness        = jfloat(at(md, "roughness"),       0.5f);
                matOverride.metallic         = jfloat(at(md, "metallic"));
                matOverride.specular         = jfloat(at(md, "specular"),        0.5f);
                const float ior = jfloat(at(md, "ior"), 1.5f);
                const glm::vec3 iorRgb(
                    jfloat(at(md, "ior_r"), ior),
                    jfloat(at(md, "ior_g"), ior),
                    jfloat(at(md, "ior_b"), ior));
                matOverride.sellmeier        = fitSellmeierFromFraunhofer(iorRgb);
                matOverride.transmission     = jfloat(at(md, "transmission"));
                matOverride.opacity          = jfloat(at(md, "opacity"),         1.f);
                matOverride.emission         = jvec3(at(md, "emission"));
                matOverride.emissionStrength = jfloat(at(md, "emission_strength"));
            }
            ImportObjScene(scene, jstr(at(obj, "path")), hasMaterial ? &matOverride : nullptr);
            if (SceneObject* root = scene.getActiveObject())
                root->setLocalTransform(t);
        } else {
            const auto& md = at(obj, "material");
            Material mat{};
            mat.albedo           = jvec3(at(md, "albedo"),          {0.8f, 0.8f, 0.8f});
            mat.roughness        = jfloat(at(md, "roughness"),       0.5f);
            mat.metallic         = jfloat(at(md, "metallic"));
            mat.specular         = jfloat(at(md, "specular"),        0.5f);
            const float ior = jfloat(at(md, "ior"), 1.5f);
            const glm::vec3 iorRgb(
                jfloat(at(md, "ior_r"), ior),
                jfloat(at(md, "ior_g"), ior),
                jfloat(at(md, "ior_b"), ior));
            mat.sellmeier        = fitSellmeierFromFraunhofer(iorRgb);
            mat.transmission     = jfloat(at(md, "transmission"));
            mat.opacity          = jfloat(at(md, "opacity"),         1.f);
            mat.emission         = jvec3(at(md, "emission"));
            mat.emissionStrength = jfloat(at(md, "emission_strength"));

            uint32_t meshIndex;
            if      (type == "cube")  meshIndex = scene.add(MeshAsset::CreateCube(scene, name, mat));
            else if (type == "plane") meshIndex = scene.add(MeshAsset::CreatePlane(scene, name, mat));
            else if (type == "disk")  meshIndex = scene.add(MeshAsset::CreateDisk(scene, name, mat));
            else                      meshIndex = scene.add(MeshAsset::CreateSphere(scene, name, mat));
            scene.add(std::make_unique<MeshInstance>(scene, name, meshIndex, t));
        }
    }
}
