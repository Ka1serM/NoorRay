#include <fstream>
#include <stdexcept>
#include "Utils.h"
#include "Shaders/SharedStructs.h"
#include "Vulkan/Texture.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "Vulkan/Renderer.h"

std::string Utils::nameFromPath(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string name = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;

    size_t lastDot = name.find_last_of('.');
    if (lastDot != std::string::npos)
        name = name.substr(0, lastDot);

    return name;
}

void Utils::loadObj(Renderer& renderer, const std::string& filepath, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces, std::vector<Material>& materials)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    // Extract directory from the filepath
    std::string objDir;
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos)
        objDir = filepath.substr(0, lastSlash);
    else
        objDir = "."; // fallback to current directory

    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, filepath.c_str(), objDir.c_str()))
        throw std::runtime_error("Failed to load OBJ: " + warn + err);


    // Append new materials from the OBJ file
    for (const auto& mat : mats) {
        Material material{};
        material.albedo       = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        material.specular     = mat.specular[0];
        material.metallic     = mat.metallic;
        material.roughness    = mat.roughness;
        material.ior          = mat.ior;
        material.transmission = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        material.emission     = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);

        // Textures indices default to -1
        material.albedoIndex = -1;
        material.specularIndex = -1;
        material.roughnessIndex = -1;
        material.normalIndex = -1;

        // Load textures relative to objDir
        if (!mat.diffuse_texname.empty()) {
            renderer.add(Texture(renderer.context, objDir + "/" + mat.diffuse_texname));
            material.albedoIndex = static_cast<int>(renderer.textures.size() - 1);
        }
        if (!mat.specular_texname.empty()) {
            renderer.add(Texture(renderer.context, objDir + "/" + mat.specular_texname));
            material.specularIndex = static_cast<int>(renderer.textures.size() - 1);
        }
        if (!mat.roughness_texname.empty()) {
            renderer.add(Texture(renderer.context, objDir + "/" + mat.roughness_texname));
            material.roughnessIndex = static_cast<int>(renderer.textures.size() - 1);
        }
        if (!mat.normal_texname.empty()) {
            renderer.add(Texture(renderer.context, objDir + "/" + mat.normal_texname));
            material.normalIndex = static_cast<int>(renderer.textures.size() - 1);
        }

        materials.emplace_back(material);
    }

    // Now process shapes and faces
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const unsigned int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
                throw std::runtime_error("Non-triangle face detected. Only triangles are supported.");

            Face face{};
            int originalMatIndex = shape.mesh.material_ids[f];
            if (originalMatIndex >= 0)
                face.materialIndex = originalMatIndex;
            else
                face.materialIndex = 0;

            // Accumulate positions for center calculation
            glm::vec3 faceCenter(0.0f);
            glm::vec3 faceNormal(0.0f);

            // Temporary storage for vertices of the face to compute normal
            glm::vec3 verts[3];

            // Process each vertex of the face
            for (int v = 0; v < fv; ++v) {
                auto [vertex_index, normal_index, texcoord_index] = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};
                vertex.position = glm::vec3(
                    attrib.vertices[3 * vertex_index + 0],
                    -attrib.vertices[3 * vertex_index + 1],
                    attrib.vertices[3 * vertex_index + 2]
                );

                verts[v] = vertex.position;
                faceCenter += vertex.position;

                if (!attrib.normals.empty() && normal_index >= 0) {
                    vertex.normal = glm::vec3(
                        attrib.normals[3 * normal_index + 0],
                        -attrib.normals[3 * normal_index + 1],
                        attrib.normals[3 * normal_index + 2]
                    );
                } else {
                    vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                }

                if (!attrib.texcoords.empty() && texcoord_index >= 0) {
                    vertex.uv = glm::vec2(
                        attrib.texcoords[2 * texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * texcoord_index + 1]
                    );
                } else {
                    vertex.uv = glm::vec2(0.0f);
                }

                vertices.push_back(vertex);
                indices.push_back(static_cast<uint32_t>(indices.size()));
            }

            faceCenter /= static_cast<float>(fv);

            // Compute face normal from cross product of edges
            glm::vec3 edge1 = verts[1] - verts[0];
            glm::vec3 edge2 = verts[2] - verts[0];
            faceNormal = glm::normalize(glm::cross(edge1, edge2));

            // Add point light if emissive material
            if (face.materialIndex < materials.size()) {
                const Material& mat = materials[face.materialIndex];
                if (glm::length(mat.emission) > 1e-6f) {
                    PointLight light{};
                    light.position = faceCenter + faceNormal * -0.01f;
                    light.color = mat.emission;
                    light.intensity = 0.01f;
                    renderer.add(light);
                }
            }

            faces.push_back(face);
            indexOffset += fv;
        }
    }
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
