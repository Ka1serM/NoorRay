#include <fstream>
#include <stdexcept>

#include "Utils.h"
#include "shaders/SharedStructs.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

void Utils::loadObj(std::string filename, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces, std::vector<Material>& materials, std::vector<PointLight>& pointLights) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;

    //Load OBJ
    if (!LoadObj(&attrib, &shapes, &mats, &warn, &err, filename.c_str(), "../assets"))
        throw std::runtime_error("Failed to load OBJ: " + warn + err);

    // Preload materials
    materials.reserve(mats.size());
    for (const auto& mat : mats) {
        Material material{};
        material.albedo       = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        material.specular     = mat.specular[0];
        material.metallic     = mat.metallic;
        material.roughness    = mat.roughness;
        material.ior          = mat.ior;
        material.transmission = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        material.emission     = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        materials.push_back(material);
    }

    // Load vertices, faces, indices
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                throw std::runtime_error("Non-triangle face detected. Only triangles are supported.");
            }

            Face face{};
            face.materialIndex = shape.mesh.material_ids[f];

            // Retrieve the material for this face
            const Material& material = materials[face.materialIndex];

            // Check if the material is emissive
            if (material.emission.x > 1 || material.emission.y > 1 || material.emission.z > 1) {
                // Calculate the center of the triangle to spawn a point light
                glm::vec3 p0 = glm::vec3(
                    attrib.vertices[3 * shape.mesh.indices[indexOffset + 0].vertex_index + 0],
                    -attrib.vertices[3 * shape.mesh.indices[indexOffset + 0].vertex_index + 1], // Flip Y axis
                    attrib.vertices[3 * shape.mesh.indices[indexOffset + 0].vertex_index + 2]
                );
                glm::vec3 p1 = glm::vec3(
                    attrib.vertices[3 * shape.mesh.indices[indexOffset + 1].vertex_index + 0],
                    -attrib.vertices[3 * shape.mesh.indices[indexOffset + 1].vertex_index + 1], // Flip Y axis
                    attrib.vertices[3 * shape.mesh.indices[indexOffset + 1].vertex_index + 2]
                );
                glm::vec3 p2 = glm::vec3(
                    attrib.vertices[3 * shape.mesh.indices[indexOffset + 2].vertex_index + 0],
                    -attrib.vertices[3 * shape.mesh.indices[indexOffset + 2].vertex_index + 1], // Flip Y axis
                    attrib.vertices[3 * shape.mesh.indices[indexOffset + 2].vertex_index + 2]
                );

                glm::vec3 center = (p0 + p1 + p2) / 3.0f; // Average of the 3 vertices to get the center

                // Create a point light at this position with the emission color as the light color
                PointLight light{};
                light.position = center - glm::vec3(0.0f, -0.1f, 0.0f);
                light.color = material.emission;
                light.intensity = 0.025f;

                pointLights.push_back(light);
            }

            for (int v = 0; v < fv; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};

                // Position
                vertex.position = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    -attrib.vertices[3 * idx.vertex_index + 1], // Flip Y axis
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // Normal (handle missing normals gracefully)
                if (!attrib.normals.empty() && idx.normal_index >= 0) {
                    vertex.normal = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        -attrib.normals[3 * idx.normal_index + 1], // Flip Y axis
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                } else {
                    vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f); // Default normal if missing
                }

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

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}
