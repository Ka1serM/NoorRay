#include "SceneImporter.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "Shaders/Shared.h"
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

void SceneImporter::ImportGltfScene(Scene& scene, const std::string& filepath)
{
    // Boilerplate: Load GLTF file from disk 
    const std::filesystem::path filePath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);

    const std::filesystem::path gltfDir = filePath.parent_path();
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;

    const bool success = filePath.extension() == ".glb"
        ? loader.LoadBinaryFromFile(&model, &err, &warn, filepath)
        : loader.LoadASCIIFromFile(&model, &err, &warn, filepath);

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
        material.albedo = make_vec3(pbr.baseColorFactor.data());
        material.metallic = static_cast<float>(pbr.metallicFactor);
        material.roughness = static_cast<float>(pbr.roughnessFactor);
        material.emission = make_vec3(mat.emissiveFactor.data());
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
                 material.ior = static_cast<float>(ext.Get("ior").Get<double>());
        }

        auto addTexture = [&](int textureIndex, int& materialIndex) {
            if (textureIndex < 0) return;
            const tinygltf::Texture& tex = model.textures[textureIndex];
            if (tex.source < 0) return;
            const tinygltf::Image& image = model.images[tex.source];
            const std::filesystem::path texturePath = gltfDir / image.uri;
            if (std::filesystem::exists(texturePath)) {
                scene.add(Texture(scene.getContext(), texturePath.string()));
                materialIndex = static_cast<int>(scene.getTextures().size() - 1);
            } else {
               LOG_ERROR("Warning: Texture file not found: " << texturePath.string());
            }
        };

        addTexture(pbr.baseColorTexture.index, material.albedoIndex);
        addTexture(pbr.metallicRoughnessTexture.index, material.roughnessIndex);
        addTexture(mat.normalTexture.index, material.normalIndex);
        addTexture(mat.emissiveTexture.index, material.emissionIndex);
        
        globalMaterials.push_back(material);
    }
    if (globalMaterials.empty())
        globalMaterials.emplace_back(); // Default material

    // Load Meshes
    std::vector<std::vector<std::shared_ptr<MeshAsset>>> loadedMeshAssets(model.meshes.size());
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
                vertices[v].position    = make_vec3(&positions[v * 3]);
                vertices[v].normal      = normals ? normalize(make_vec3(&normals[v * 3])) : vec3(0, 1, 0);
                vertices[v].uv          = texcoords ? vec2(texcoords[v * 2], 1.0f - texcoords[v * 2 + 1]) : vec2(0);
                vertices[v].tangent     = tangents ? normalize(vec3(make_vec3(&tangents[v * 4]))) : vec3(1, 0, 0);
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
            
            auto meshAsset = std::make_shared<MeshAsset>(scene, meshName, std::move(vertices), std::move(indices), std::vector<Face>(indices.size() / 3), std::vector<Material>{globalMaterials[materialID]});
            scene.add(meshAsset);
            loadedMeshAssets[i].push_back(meshAsset);
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
            localTransform = make_mat4(node.matrix.data());
        } else {
            // Decompose TRS and combine. glTF spec is T * R * S.
            vec3 T = node.translation.size() == 3 ? vec3(make_vec3(node.translation.data())) : vec3(0.0f);
            // glTF quat is [x, y, z, w], while GLM is [w, x, y, z]
            quat R = node.rotation.size() == 4 ? quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]), static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2])) : quat(1.0f, 0.0f, 0.0f, 0.0f);
            vec3 S = node.scale.size() == 3 ? vec3(make_vec3(node.scale.data())) : vec3(1.0f);
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
            for (const auto& asset : loadedMeshAssets[node.mesh]) {
                // MeshInstances have an identity transform relative to their parent node
                auto instance = std::make_unique<MeshInstance>(scene, asset->getName() + "_inst", asset, Transform{});
                SceneObject* instPtr = instance.get();
                scene.registerObject(std::move(instance));
                instPtr->setParent(objPtr);
                objPtr->addChild(instPtr);
            }
        }
    }

    // STEP 3: Assemble the final scene hierarchy 
    // With all nodes created, build the hierarchy by reparenting each node. The 'reparent'
    // operation should preserve the node's world transform by calculating the correct
    // new local transform relative to its parent.

    auto importRoot = std::make_unique<SceneObject>(scene, nameFromPath(filepath), Transform{});
    SceneObject* importRootPtr = importRoot.get();
    const uint32_t rootIdx = scene.add(std::move(importRoot));

    // Reparent nodes to build the glTF hierarchy
    for (size_t i = 0; i < model.nodes.size(); ++i) {
        const auto& node = model.nodes[i];
        if (nodeMap[i] == nullptr) continue;

        for (int childIndex : node.children) {
            if (nodeMap[childIndex]) {
                scene.reparent(nodeMap[childIndex], nodeMap[i]);
            }
        }
    }
    
    // Attach the scene's original root nodes to our new import object
    if (model.defaultScene >= 0)
        for (int rootNodeIndex : model.scenes[model.defaultScene].nodes)
            if (nodeMap[rootNodeIndex])
                scene.reparent(nodeMap[rootNodeIndex], importRootPtr);
    
    scene.setActiveObjectIndex(rootIdx);

    const mat4 correctionMatrix = scale(mat4(1.0f), vec3(1.0f, -1.0f, -1.0f));
    scene.getActiveObject()->setLocalTransform(Transform(correctionMatrix));
}

void SceneImporter::ImportObjScene(Scene& scene, const std::string& filepath)
{
    std::filesystem::path filePath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);
    
    // The directory containing the .obj file, used for finding .mtl and textures
    std::filesystem::path objDir = filePath.has_parent_path() ? filePath.parent_path() : ".";

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, filepath.c_str(), objDir.string().c_str(), true))
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
        material.ior = mat.ior;
        material.transmissionColor = vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        material.transmission = (mat.illum == 4 || mat.illum == 7) ? 1.0f : 0.0f;
        material.opacity = mat.dissolve;
        material.emission = vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        material.emissionStrength = (length2(material.emission) > 1e-6f) ? 1.0f : 0.0f;

        // Lambda to safely find and load a texture
        auto addTexture = [&](const std::string& texname, int& index) {
            if (!texname.empty()) {
                const std::filesystem::path texturePath = objDir / texname;
                if (std::filesystem::exists(texturePath)) {
                    scene.add(Texture(scene.getContext(), texturePath.string()));
                    index = static_cast<int>(scene.getTextures().size() - 1);
                } else
                   LOG_ERROR("Warning: Texture file not found: " << texturePath.string());
            }
        };

        addTexture(mat.diffuse_texname, material.albedoIndex);
        addTexture(mat.specular_texname, material.specularIndex);
        addTexture(mat.roughness_texname, material.roughnessIndex);
        addTexture(mat.normal_texname, material.normalIndex);
        addTexture(mat.alpha_texname, material.opacityIndex);
        addTexture(mat.emissive_texname, material.emissionIndex);

        globalMaterials.push_back(material);
    }
    
    if (globalMaterials.empty())
        globalMaterials.emplace_back(); // Adds a default-constructed Material if the .mtl was missing

    // Create a parent object for this entire OBJ file to keep the scene organized
    std::string parentName = nameFromPath(filepath);
    auto parentObject = std::make_unique<SceneObject>(scene, parentName, Transform{});
    uint32_t parentIndex = scene.add(std::move(parentObject));

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
        auto meshAsset = std::make_shared<MeshAsset>(scene, meshName, std::move(vertices), std::move(indices), std::move(faces), std::move(localMaterials));
        scene.add(meshAsset);

        Transform transform;
        transform.setPosition(center);
        auto instance = std::make_unique<MeshInstance>(scene, meshName, meshAsset, transform);
        uint32_t instanceIndex = scene.add(std::move(instance));

        // Parent the new instance to the main object for this file
        scene.reparent(scene.getObject(instanceIndex), scene.getObject(parentIndex));
    }

    scene.setActiveObjectIndex(parentIndex);
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
