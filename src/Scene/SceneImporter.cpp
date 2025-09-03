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
    std::filesystem::path filePath(filepath);
    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("File not found: " + filepath);
    
    // The directory containing the .obj file, used for finding .mtl and textures
    std::filesystem::path objDir = filePath.has_parent_path() ? filePath.parent_path() : ".";

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    // Load the OBJ file, with the crucial 'triangulate' parameter set to true.
    // This converts all polygons (quads, n-gons) into triangles.
    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, filepath.c_str(), objDir.string().c_str(), true))
        throw std::runtime_error("Failed to load OBJ file: " + warn + err);

    // Log any warnings or errors from the loader, but don't necessarily stop.
    // A missing material file is a common "error" we want to handle gracefully.
    if (!warn.empty())
        std::cerr << "TinyObjLoader Warning: " << warn << std::endl;
    if (!err.empty())
        std::cerr << "TinyObjLoader Info/Error: " << err << std::endl;

    // --- Load Global Materials from the MTL file (if it was found) ---
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
        material.emissionStrength = (glm::length2(material.emission) > 1e-6f) ? 1.0f : 0.0f;

        // Lambda to safely find and load a texture
        auto addTexture = [&](const std::string& texname, int& index) {
            if (!texname.empty()) {
                const std::filesystem::path texturePath = objDir / texname;
                if (std::filesystem::exists(texturePath)) {
                    scene.add(Texture(scene.getContext(), texturePath.string()));
                    index = static_cast<int>(scene.getTextures().size() - 1);
                } else
                    std::cerr << "Warning: Texture file not found: " << texturePath.string() << std::endl;
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

        // --- Collect and remap materials used ONLY by this shape ---
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
        
        // --- Load vertex and face data for the current shape ---
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

            // --- Accumulate tangent data for later averaging ---
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

        // --- Finalize vertices by normalizing the accumulated tangents ---
        for (auto& v : vertices) {
            if (length2(v.tangent) > 1e-8f) {
                // Gram-Schmidt orthogonalize tangent with respect to the normal
                v.tangent = normalize(v.tangent - v.normal * dot(v.normal, v.tangent));
            } else {
                // Create an arbitrary but valid tangent if calculation failed (e.g., degenerate UVs)
                vec3 t1 = cross(v.normal, vec3(0.0, 0.0, 1.0));
                vec3 t2 = cross(v.normal, vec3(0.0, 1.0, 0.0));
                v.tangent = normalize((length2(t1) > length2(t2)) ? t1 : t2);
            }
        }

        // --- Compute mesh pivot and center the vertex positions ---
        vec3 minPos = vertices[0].position;
        vec3 maxPos = vertices[0].position;
        for (const auto& v : vertices) {
            minPos = min(minPos, v.position);
            maxPos = max(maxPos, v.position);
        }
        vec3 center = (minPos + maxPos) * 0.5f;

        for (auto& v : vertices)
            v.position -= center;

        // --- Create the final assets and instances for this shape ---
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