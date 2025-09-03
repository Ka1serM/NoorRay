#include "SceneImporter.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "Shaders/SharedStructs.h"
#include "Vulkan/Texture.h"
#include <nlohmann/json.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include "tiny_obj_loader.h"
#include "Scene/Scene.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "MeshInstance.h"
#include "glm/gtx/norm.hpp"
#include "Mesh/MeshAsset.h"
#include "Mesh/Transform.h"

void SceneImporter::ImportCrtScene(Scene& scene, const std::string& filepath)
{
    return;
}

void SceneImporter::ImportObjScene(Scene& scene, const std::string& filepath)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    std::string objDir;
    size_t lastSlash = filepath.find_last_of("/\\");
    objDir = (lastSlash != std::string::npos) ? filepath.substr(0, lastSlash) : ".";

    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, filepath.c_str(), objDir.c_str()))
        throw std::runtime_error("Failed to load OBJ: " + warn + err);

    // Load global materials
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
        material.emissionStrength = (material.emission != vec3(0.0f)) ? 1.0f : 0.0f;

        auto addTexture = [&](const std::string& texname, int& index) {
            if (!texname.empty()) {
                std::string texturePath = objDir + "/" + texname;
                if (std::filesystem::exists(texturePath)) {
                    scene.add(Texture(scene.getContext(), texturePath));
                    index = static_cast<int>(scene.getTextures().size() - 1);
                } else
                    std::cerr << "Warning: Texture file not found: " << texturePath << std::endl;
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
    if (globalMaterials.empty()) globalMaterials.emplace_back();

    // Parent object for this file
    std::string parentName = nameFromPath(filepath);
    auto parentObject = std::make_unique<SceneObject>(scene, parentName, Transform{});
    int parentIndex = scene.add(std::move(parentObject));

    // Process each shape
    for (const auto& shape : shapes) {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<Face> faces;

        // --- Collect used materials ---
        std::unordered_map<int, int> matRemap; // old index -> new local index
        std::vector<Material> localMaterials;
        for (int faceIdx = 0; faceIdx < shape.mesh.material_ids.size(); ++faceIdx) {
            int matId = shape.mesh.material_ids[faceIdx];
            if (matId < 0) matId = 0; // default to 0 if no material
            if (!matRemap.contains(matId)) {
                matRemap[matId] = static_cast<int>(localMaterials.size());
                localMaterials.push_back(globalMaterials[matId]);
            }
        }
        if (localMaterials.empty()) localMaterials.push_back(Material{}); // ensure at least one

        // --- Load vertices and faces ---
        size_t indexOffset = 0;
        for (size_t faceIndex = 0; faceIndex < shape.mesh.num_face_vertices.size(); ++faceIndex) {
            const unsigned int fv = shape.mesh.num_face_vertices[faceIndex];
            if (fv != 3) throw std::runtime_error("Only triangles supported");

            Face face{};
            int origMatIndex = shape.mesh.material_ids[faceIndex];
            if (origMatIndex < 0) origMatIndex = 0;
            face.materialIndex = matRemap[origMatIndex]; // remap to local index

            uint32_t triIndices[3];
            for (unsigned int v = 0; v < fv; ++v) {
                const auto& [vertex_index, normal_index, texcoord_index] = shape.mesh.indices[indexOffset + v];
                Vertex vertex{};
                vertex.position = vec3(
                    attrib.vertices[3 * vertex_index + 0],
                    -attrib.vertices[3 * vertex_index + 1],
                    -attrib.vertices[3 * vertex_index + 2]
                );

                if (!attrib.normals.empty() && normal_index >= 0) {
                    vertex.normal = vec3(
                        attrib.normals[3 * normal_index + 0],
                        -attrib.normals[3 * normal_index + 1],
                        -attrib.normals[3 * normal_index + 2]
                    );
                } else {
                    vertex.normal = vec3(0.0f, 1.0f, 0.0f);
                }

                if (!attrib.texcoords.empty() && texcoord_index >= 0) {
                    vertex.uv = vec2(
                        attrib.texcoords[2 * texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * texcoord_index + 1]
                    );
                } else {
                    vertex.uv = vec2(0.0f);
                }

                vertices.push_back(vertex);
                triIndices[v] = static_cast<uint32_t>(vertices.size() - 1);
                indices.push_back(triIndices[v]);
            }

            faces.push_back(face);
            indexOffset += fv;

            // --- Compute tangent ---
            Vertex& v0 = vertices[triIndices[0]];
            Vertex& v1 = vertices[triIndices[1]];
            Vertex& v2 = vertices[triIndices[2]];

            vec3 edge1 = v1.position - v0.position;
            vec3 edge2 = v2.position - v0.position;
            vec2 deltaUV1 = v1.uv - v0.uv;
            vec2 deltaUV2 = v2.uv - v0.uv;

            float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
            vec3 tangent = (std::fabs(f) < 1e-8f) ? vec3(1, 0, 0) : (1.0f / f) * (deltaUV2.y * edge1 - deltaUV1.y * edge2);

            v0.tangent += tangent;
            v1.tangent += tangent;
            v2.tangent += tangent;
        }

        // Normalize tangents
        for (auto& v : vertices)
            v.tangent = (length2(v.tangent) > 1e-8f) ? normalize(v.tangent) : vec3(1, 0, 0);

        // --- Compute pivot ---
        vec3 minPos = vertices[0].position;
        vec3 maxPos = vertices[0].position;
        for (const auto& v : vertices) {
            minPos = min(minPos, v.position);
            maxPos = max(maxPos, v.position);
        }
        vec3 center = (minPos + maxPos) * 0.5f;

        for (auto& v : vertices)
            v.position -= center;

        // --- Create MeshAsset ---
        std::string meshName = shape.name.empty() ? "Shape" : shape.name;
        auto meshAsset = std::make_shared<MeshAsset>(scene, meshName, std::move(vertices), std::move(indices), std::move(faces), std::move(localMaterials));
        scene.add(meshAsset);

        // --- Create MeshInstance ---
        Transform transform;
        transform.setPosition(center);
        auto instance = std::make_unique<MeshInstance>(scene, meshName, meshAsset, transform);
        int instanceIndex = scene.add(std::move(instance));

        scene.reparent(scene.getObject(instanceIndex), scene.getObject(parentIndex));
    }

    scene.setActiveObjectIndex(parentIndex);
}

std::string SceneImporter::nameFromPath(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string name = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;

    size_t lastDot = name.find_last_of('.');
    if (lastDot != std::string::npos)
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