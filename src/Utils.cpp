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

    // Extract directory from filepath
    std::string objDir;
    size_t lastSlash = filepath.find_last_of("/\\");
    objDir = (lastSlash != std::string::npos) ? filepath.substr(0, lastSlash) : ".";

    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, filepath.c_str(), objDir.c_str()))
        throw std::runtime_error("Failed to load OBJ: " + warn + err);

    // Load materials
    materials.clear();
    for (const auto& mat : mats) {
        Material material{};
        material.albedo       = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        material.specular     = mat.specular[0];
        material.metallic     = mat.metallic;
        material.roughness    = mat.roughness;
        material.ior          = mat.ior;
        material.transmission = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        material.emission     = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);

        material.albedoIndex = -1;
        material.specularIndex = -1;
        material.roughnessIndex = -1;
        material.normalIndex = -1;

        // Load textures if present
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

        materials.push_back(material);
    }
    
    // Process shapes and faces
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const unsigned int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
                throw std::runtime_error("Only triangles are supported.");

            Face face{};
            int matIndex = shape.mesh.material_ids[f];
            face.materialIndex = (matIndex >= 0) ? matIndex : 0;
            
            // Process vertices of the face
            for (unsigned int v = 0; v < fv; ++v) {
                const tinyobj::index_t& idx = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};
                vertex.position = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    -attrib.vertices[3 * idx.vertex_index + 1],  // Flip Y-axis
                    attrib.vertices[3 * idx.vertex_index + 2]
                );
                
                if (!attrib.normals.empty() && idx.normal_index >= 0) {
                    vertex.normal = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        -attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                } else
                    vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);

                if (!attrib.texcoords.empty() && idx.texcoord_index >= 0) {
                    vertex.uv = glm::vec2(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
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
