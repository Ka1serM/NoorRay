#include <cmath>
#include <fstream>
#include <stdexcept>
#include "Utils.h"
#include "Shaders/SharedStructs.h"
#include "Vulkan/Texture.h"
#include <nlohmann/json.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include "tiny_obj_loader.h"
#include "Scene/Scene.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "glm/gtx/norm.hpp"

void Utils::loadCrtScene(Scene& scene, const std::string& filepath, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces, std::vector<Material>& materials)
{
    std::ifstream f(filepath);
    if (!f.is_open())
        throw std::runtime_error("Failed to open file: " + filepath);

    nlohmann::json data = nlohmann::json::parse(f);

    try
    {
        // // 1Parse Settings and Camera
        // // Parse Lights
        
        //. Parse Materials
        materials.clear();
        if (data.contains("materials")) {
            for (const auto& mat_data : data["materials"]) {
                Material material{};
                
                if (mat_data.contains("albedo"))
                    material.albedo = make_vec3(mat_data["albedo"].get<std::vector<float>>().data());
                else
                    material.albedo = vec3(0.7f, 0.7f, 0.7f); // Default gray
                
                material.specular = mat_data.value("specular", 0.0f);
                material.metallic = mat_data.value("metallic", 0.0f);
                material.roughness = mat_data.value("roughness", 1.0f);
                material.ior = mat_data.value("ior", 1.0f);
                
                if (mat_data.contains("transmission")) {
                    auto transmission_data = mat_data["transmission"].get<std::vector<float>>();
                    vec4 transmission(0.0f);
                    for (size_t i = 0; i < transmission_data.size() && i < 3; ++i)
                        transmission[i] = transmission_data[i];
                    material.transmissionColor = transmission;
                } else
                    material.transmissionColor = vec4(0.0f);

                if (mat_data.contains("emission")) {
                    auto emission_data = mat_data["emission"].get<std::vector<float>>();
                    vec4 emission(0.0f);
                    for (size_t i = 0; i < emission_data.size() && i < 3; ++i)
                        emission[i] = emission_data[i];
                    material.emission = emission;
                } else
                    material.emission = vec4(0.0f);
                
                materials.push_back(material);
            }
        }
        
        // Create default material if no materials found
        if (materials.empty())
            materials.emplace_back();

        // 4. Parse Objects (Meshes)
        if (data.contains("objects")) {
            for (const auto& obj_data : data["objects"]) {
                uint32_t baseVertexIndex = static_cast<uint32_t>(vertices.size());
                int materialIndex = obj_data.value("material_index", 0);
                
                // Ensure material index is valid, clamp to available materials
                if (materialIndex >= static_cast<int>(materials.size()) || materialIndex < 0) {
                    materialIndex = 0;
                }

                // Parse vertices
                if (obj_data.contains("vertices")) {
                    const auto& verts_array = obj_data["vertices"].get<std::vector<float>>();
                    for (size_t i = 0; i < verts_array.size(); i += 3) {
                        // Check bounds to prevent crashes
                        if (i + 2 >= verts_array.size()) break;
                        
                        Vertex v{};
                        // Flip Y-axis to be consistent with the OBJ loader
                        v.position = vec3(verts_array[i], -verts_array[i + 1], verts_array[i + 2]);
                        v.normal = vec3(0.0f, 1.0f, 0.0f); // Default normal
                        v.uv = vec2(0.0f); // Default UV
                        vertices.push_back(v);
                    }
                }

                // Parse normals if available
                std::vector<vec3> objNormals;
                if (obj_data.contains("normals")) {
                    const auto& normals_array = obj_data["normals"].get<std::vector<float>>();
                    for (size_t i = 0; i < normals_array.size(); i += 3) {
                        // Check bounds
                        if (i + 2 >= normals_array.size()) break;
                        
                        // Flip Y-axis for normals too
                        objNormals.push_back(vec3(normals_array[i], -normals_array[i + 1], normals_array[i + 2]));
                    }
                }

                // Parse UVs if available
                std::vector<vec2> objUVs;
                if (obj_data.contains("uvs")) {
                    const auto& uvs_array = obj_data["uvs"].get<std::vector<float>>();
                    for (size_t i = 0; i < uvs_array.size(); i += 2) {
                        // Check bounds
                        if (i + 1 >= uvs_array.size()) break;
                        
                        objUVs.push_back(vec2(uvs_array[i], 1.0f - uvs_array[i + 1])); // Flip V coordinate
                    }
                }

                // Parse triangles/faces
                if (obj_data.contains("triangles")) {
                    const auto& tris_array = obj_data["triangles"].get<std::vector<uint32_t>>();
                    for (size_t i = 0; i < tris_array.size(); i += 3) {
                        // Check bounds to prevent crashes
                        if (i + 2 >= tris_array.size())
                            break;
                        
                        uint32_t i0 = baseVertexIndex + tris_array[i];
                        uint32_t i1 = baseVertexIndex + tris_array[i + 1];
                        uint32_t i2 = baseVertexIndex + tris_array[i + 2];

                        // Validate vertex indices
                        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
                            continue; // Skip invalid triangles

                        indices.push_back(i0);
                        indices.push_back(i1);
                        indices.push_back(i2);

                        Face face{};
                        face.materialIndex = materialIndex;
                        faces.push_back(face);

                        // Get vertex references
                        Vertex& v0 = vertices[i0];
                        Vertex& v1 = vertices[i1];
                        Vertex& v2 = vertices[i2];

                        // Apply normals if available
                        if (!objNormals.empty()) {
                            uint32_t localIdx0 = tris_array[i];
                            uint32_t localIdx1 = tris_array[i + 1];
                            uint32_t localIdx2 = tris_array[i + 2];
                            
                            if (localIdx0 < objNormals.size())
                                v0.normal = objNormals[localIdx0];
                            if (localIdx1 < objNormals.size())
                                v1.normal = objNormals[localIdx1];
                            if (localIdx2 < objNormals.size())
                                v2.normal = objNormals[localIdx2];
                            
                        } else {
                            // Calculate face normal if no normals provided
                            vec3 edge1 = v1.position - v0.position;
                            vec3 edge2 = v2.position - v0.position;
                            vec3 normal = normalize(cross(edge1, edge2));
                            
                            v0.normal = normal;
                            v1.normal = normal;
                            v2.normal = normal;
                        }

                        // Apply UVs if available
                        if (!objUVs.empty()) {
                            uint32_t localIdx0 = tris_array[i];
                            uint32_t localIdx1 = tris_array[i + 1];
                            uint32_t localIdx2 = tris_array[i + 2];
                            
                            if (localIdx0 < objUVs.size())
                                v0.uv = objUVs[localIdx0];
                            if (localIdx1 < objUVs.size())
                                v1.uv = objUVs[localIdx1];
                            if (localIdx2 < objUVs.size())
                                v2.uv = objUVs[localIdx2];
                        }
                    }
                }
            }
        }
    }
    catch (nlohmann::json::exception& e)
    {
        throw std::runtime_error("Failed to parse JSON scene: " + std::string(e.what()));
    }
}
void Utils::loadObj(
    Scene& scene,
    const std::string& filepath,
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    std::vector<Face>& faces,
    std::vector<Material>& materials)
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

    // Load materials
    materials.clear();
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

        materials.push_back(material);
    }

    // Load geometry
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;
        for (size_t faceIndex = 0; faceIndex < shape.mesh.num_face_vertices.size(); ++faceIndex) {
            const unsigned int fv = shape.mesh.num_face_vertices[faceIndex];
            if (fv != 3)
                throw std::runtime_error("Only triangles are supported.");

            Face face{};
            int matIndex = shape.mesh.material_ids[faceIndex];
            face.materialIndex = (matIndex >= 0) ? matIndex : 0;

            uint32_t triIndices[3];
            for (unsigned int v = 0; v < fv; ++v) {
                const tinyobj::index_t& idx = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};
                vertex.position = vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    -attrib.vertices[3 * idx.vertex_index + 1],
                    -attrib.vertices[3 * idx.vertex_index + 2]
                );

                if (!attrib.normals.empty() && idx.normal_index >= 0) {
                    vertex.normal = vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        -attrib.normals[3 * idx.normal_index + 1],
                        -attrib.normals[3 * idx.normal_index + 2]
                    );
                } else {
                    vertex.normal = vec3(0.0f, 1.0f, 0.0f);
                }

                if (!attrib.texcoords.empty() && idx.texcoord_index >= 0) {
                    vertex.uv = vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
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

            // Tangent calculation
            Vertex& v0 = vertices[triIndices[0]];
            Vertex& v1 = vertices[triIndices[1]];
            Vertex& v2 = vertices[triIndices[2]];

            vec3 edge1 = v1.position - v0.position;
            vec3 edge2 = v2.position - v0.position;
            vec2 deltaUV1 = v1.uv - v0.uv;
            vec2 deltaUV2 = v2.uv - v0.uv;

            float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
            vec3 tangent;

            if (std::fabs(f) < 1e-8f) {
                // Degenerate UV, pick arbitrary tangent
                tangent = vec3(1.0f, 0.0f, 0.0f);
            } else {
                f = 1.0f / f;
                tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
            }

            v0.tangent += tangent;
            v1.tangent += tangent;
            v2.tangent += tangent;
        }
    }

    // Normalize per-vertex tangents safely
    for (auto& v : vertices) {
        if (length2(v.tangent) > 1e-8f)
            v.tangent = normalize(v.tangent);
        else
            v.tangent = vec3(1.0f, 0.0f, 0.0f);
    }
}


std::string Utils::nameFromPath(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string name = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;

    size_t lastDot = name.find_last_of('.');
    if (lastDot != std::string::npos)
        name = name.substr(0, lastDot);

    return name;
}

std::vector<char> Utils::readFile(const std::string& filename) {
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