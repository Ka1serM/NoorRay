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

void Utils::loadObj(
    const std::shared_ptr<Context>& context,
    const std::string& filepath,
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    std::vector<Face>& faces,
    std::vector<Material>& materials,   // global materials array
    std::vector<Texture>& textures      // global textures array
) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    if (!LoadObj(&attrib, &shapes, &mats, &warn, &err, filepath.c_str(), "../assets"))
        throw std::runtime_error("Failed to load OBJ: " + warn + err);

    // Store how many materials already exist globally
    size_t baseMaterialIndex = materials.size();

    // Append new materials from the OBJ file
    materials.reserve(baseMaterialIndex + mats.size());
    for (const auto& mat : mats) {
        Material material{};
        material.albedo       = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        material.specular     = mat.specular[0];
        material.metallic     = mat.metallic;
        material.roughness    = mat.roughness;
        material.ior          = mat.ior;
        material.transmission = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        material.emission     = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]) * 20.0f;

        // Textures indices default to -1
        material.albedoIndex = -1;
        material.specularIndex = -1;
        material.roughnessIndex = -1;
        material.normalIndex = -1;

        if (!mat.diffuse_texname.empty()) {
            textures.emplace_back(context, "../assets/" + mat.diffuse_texname);
            material.albedoIndex = static_cast<int>(textures.size() - 1);
        }
        if (!mat.specular_texname.empty()) {
            textures.emplace_back(context, "../assets/" + mat.specular_texname);
            material.specularIndex = static_cast<int>(textures.size() - 1);
        }
        if (!mat.roughness_texname.empty()) {
            textures.emplace_back(context, "../assets/" + mat.roughness_texname);
            material.roughnessIndex = static_cast<int>(textures.size() - 1);
        }
        if (!mat.normal_texname.empty()) {
            textures.emplace_back(context, "../assets/" + mat.normal_texname);
            material.normalIndex = static_cast<int>(textures.size() - 1);
        }

        materials.push_back(material);
    }

    // Now process shapes and faces
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const unsigned int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
                throw std::runtime_error("Non-triangle face detected. Only triangles are supported.");

            Face face{};
            // Update face.materialIndex to global index
            int originalMatIndex = shape.mesh.material_ids[f];
            face.materialIndex = (originalMatIndex >= 0) ? static_cast<uint32_t>(baseMaterialIndex + originalMatIndex) : 0;

            const Material& material = materials[face.materialIndex];

            // Spawn point lights
            for (int v = 0; v < fv; ++v) {
                auto [vertex_index, normal_index, texcoord_index] = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};
                vertex.position = glm::vec3(
                    attrib.vertices[3 * vertex_index + 0],
                    -attrib.vertices[3 * vertex_index + 1],
                    attrib.vertices[3 * vertex_index + 2]
                );

                if (!attrib.normals.empty() && normal_index >= 0) {
                    vertex.normal = glm::vec3(
                        attrib.normals[3 * normal_index + 0],
                        -attrib.normals[3 * normal_index + 1],
                        attrib.normals[3 * normal_index + 2]
                    );
                } else
                    vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);

                if (!attrib.texcoords.empty() && texcoord_index >= 0) {
                    vertex.uv = glm::vec2(
                        attrib.texcoords[2 * texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * texcoord_index + 1]
                    );
                } else
                    vertex.uv = glm::vec2(0.0f);

                vertices.push_back(vertex);
                indices.push_back(static_cast<uint32_t>(indices.size()));
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
