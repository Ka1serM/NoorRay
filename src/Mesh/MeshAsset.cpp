#include "MeshAsset.h"
#include "Utils.h"
#include "Vulkan/Renderer.h"

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>

#include "imgui.h"
#include "glm/gtc/type_ptr.inl"
#include "UI/ImGuiManager.h"

std::shared_ptr<MeshAsset> MeshAsset::CreateCube(Renderer& renderer, const std::string& name, const Material& material) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    float h = 0.5f;
    materials.push_back(material);

    const vec3 faceNormals[6] = {
        { 0,  0,  1}, { 0,  0, -1},
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0}
    };

    const vec3 tangents[6] = {
        {1, 0, 0}, {-1, 0, 0},
        {0, 0, -1}, {0, 0, 1},
        {1, 0, 0}, {1, 0, 0}
    };

    const vec3 bitangents[6] = {
        {0, 1, 0}, {0, 1, 0},
        {0, 1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    };

    uint32_t vertexStart = 0;

    for (int faceIdx = 0; faceIdx < 6; ++faceIdx) {
        vec3 normal = faceNormals[faceIdx];
        vec3 tangent = glm::normalize(tangents[faceIdx]);
        vec3 bitangent = glm::normalize(bitangents[faceIdx]);

        vec3 corners[4] = {
            normal * h + (-tangent - bitangent) * h,
            normal * h + ( tangent - bitangent) * h,
            normal * h + ( tangent + bitangent) * h,
            normal * h + (-tangent + bitangent) * h
        };

        vec2 uvs[4] = {{0,0}, {1,0}, {1,1}, {0,1}};

        for (int i = 0; i < 4; ++i) {
            vertices.push_back(Vertex{
                corners[i], 0,
                normal, 0,
                tangent, 0,
                uvs[i], 0, 0
            });
        }

        indices.insert(indices.end(), {
            vertexStart + 0, vertexStart + 1, vertexStart + 2,
            vertexStart + 0, vertexStart + 2, vertexStart + 3
        });

        faces.push_back({0});
        faces.push_back({0});
        vertexStart += 4;
    }

    return std::make_shared<MeshAsset>(renderer, name,
        std::move(vertices),
        std::move(indices),
        std::move(faces),
        std::move(materials));
}

std::shared_ptr<MeshAsset> MeshAsset::CreatePlane(Renderer& renderer, const std::string& name, const Material& material) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
    std::vector<Face> faces;
    std::vector<Material> materials;

    float halfSize = 0.5f;
    vec3 normal = {0.0f, 1.0f, 0.0f};
    vec3 tangent = {1.0f, 0.0f, 0.0f};

    vec3 positions[4] = {
        {-halfSize, 0.0f, -halfSize},
        { halfSize, 0.0f, -halfSize},
        { halfSize, 0.0f,  halfSize},
        {-halfSize, 0.0f,  halfSize}
    };

    vec2 uvs[4] = {{0,0}, {1,0}, {1,1}, {0,1}};

    for (int i = 0; i < 4; ++i) {
        vertices.push_back(Vertex{
            positions[i], 0,
            normal, 0,
            tangent, 0,
            uvs[i], 0, 0
        });
    }

    materials.push_back(material);

    for (size_t i = 0; i < indices.size(); i += 3) {
        faces.push_back({0});
    }

    return std::make_shared<MeshAsset>(renderer, name,
        std::move(vertices),
        std::move(indices),
        std::move(faces),
        std::move(materials));
}

std::shared_ptr<MeshAsset> MeshAsset::CreateSphere(Renderer& renderer, const std::string& name, const Material& material, uint32_t latSeg, uint32_t lonSeg) {
    if (latSeg < 2 || lonSeg < 3)
        throw std::runtime_error("Sphere segments too low.");

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    for (uint32_t lat = 0; lat <= latSeg; ++lat) {
        float theta = float(M_PI) * lat / latSeg;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        for (uint32_t lon = 0; lon <= lonSeg; ++lon) {
            float phi = 2.0f * float(M_PI) * lon / lonSeg;
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;

            vec3 pos = {x * 0.5f, y * 0.5f, z * 0.5f};
            vec3 normal = glm::normalize(pos);

            vec3 tangent = glm::normalize(vec3{-sinPhi, 0.0f, cosPhi});

            vec2 uv = {lon / float(lonSeg), lat / float(latSeg)};

            vertices.push_back(Vertex{
                pos, 0,
                normal, 0,
                tangent, 0,
                uv, 0, 0
            });
        }
    }

    for (uint32_t lat = 0; lat < latSeg; ++lat) {
        for (uint32_t lon = 0; lon < lonSeg; ++lon) {
            uint32_t i0 = lat * (lonSeg + 1) + lon;
            uint32_t i1 = (lat + 1) * (lonSeg + 1) + lon;
            uint32_t i2 = i0 + 1;
            uint32_t i3 = i1 + 1;

            indices.insert(indices.end(), {i0, i1, i2, i2, i1, i3});
        }
    }

    materials.push_back(material);

    for (size_t i = 0; i < indices.size(); i += 3) {
        faces.push_back({0});
    }

    return std::make_shared<MeshAsset>(renderer, name,
        std::move(vertices),
        std::move(indices),
        std::move(faces),
        std::move(materials));
}

std::shared_ptr<MeshAsset> MeshAsset::CreateDisk(Renderer& renderer, const std::string& name, const Material& material, uint32_t segments) {
    if (segments < 3)
        throw std::runtime_error("Disk requires at least 3 segments");

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    float radius = 0.5f;
    vec3 normal = {0.0f, 1.0f, 0.0f};

    vertices.push_back(Vertex{
        {0.0f, 0.0f, 0.0f}, 0,
        normal, 0,
        {1.0f, 0.0f, 0.0f}, 0,
        {0.5f, 0.5f}, 0, 0
    }); // center vertex with tangent along +X

    for (uint32_t i = 0; i <= segments; ++i) {
        float angle = float(i) / segments * 2.0f * float(M_PI);
        float x = std::cos(angle) * radius;
        float z = std::sin(angle) * radius;

        vec3 pos = {x, 0.0f, z};
        vec3 tangent = {-std::sin(angle), 0.0f, std::cos(angle)};

        vec2 uv = {0.5f + x, 0.5f + z};

        vertices.push_back(Vertex{
            pos, 0,
            normal, 0,
            tangent, 0,
            uv, 0, 0
        });
    }

    for (uint32_t i = 1; i <= segments; ++i) {
        indices.insert(indices.end(), {0, i, i + 1});
        faces.push_back({0});
    }

    materials.push_back(material);

    return std::make_shared<MeshAsset>(renderer, name,
        std::move(vertices),
        std::move(indices),
        std::move(faces),
        std::move(materials));
}

std::shared_ptr<MeshAsset> MeshAsset::CreateFromObj(Renderer& renderer, const std::string& objFilePath)
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    Utils::loadObj(renderer, objFilePath, vertices, indices, faces, materials);

    return std::make_shared<MeshAsset>(renderer, objFilePath, std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
}

MeshAsset::MeshAsset(Renderer& renderer, const std::string& name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials)
: renderer(renderer), path(name), materials(materials) {
    
    // Upload mesh data to GPU
    vertexBuffer = Buffer{renderer.context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data()};
    indexBuffer = Buffer{renderer.context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    faceBuffer = Buffer{renderer.context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};
    materialBuffer = Buffer{renderer.context, Buffer::Type::AccelInput, sizeof(Material) * materials.size(), materials.data()};

    vk::AccelerationStructureGeometryTrianglesDataKHR triangleData{};
    triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triangleData.setVertexData(vertexBuffer.getDeviceAddress());
    triangleData.setVertexStride(sizeof(Vertex));
    triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
    triangleData.setIndexType(vk::IndexType::eUint32);
    triangleData.setIndexData(indexBuffer.getDeviceAddress());

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    geometry.setGeometry({triangleData});
    geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    // Create bottom-level acceleration structure (BLAS)
    blas.build(renderer.context, geometry, static_cast<uint32_t>(indices.size() / 3), vk::AccelerationStructureTypeKHR::eBottomLevel);
}

uint64_t MeshAsset::getBlasAddress() const {
    return blas.buffer.getDeviceAddress();
}

MeshAddresses MeshAsset::getBufferAddresses() const {
    return MeshAddresses{
        vertexBuffer.getDeviceAddress(),
        indexBuffer.getDeviceAddress(),
        faceBuffer.getDeviceAddress(),
        materialBuffer.getDeviceAddress(),
    };
}

uint32_t MeshAsset::getMeshIndex() const {
    return index;
}

void MeshAsset::setMeshIndex(uint32_t newIndex) {
    index = newIndex;
}

void MeshAsset::renderUi() {
    ImGuiManager::tableRowLabel("Source");

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    ImGui::PushItemWidth(-1);
    ImGui::InputText("##meshPath", path.data(), path.size() + 1, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopItemWidth();
    ImGui::PopStyleColor();

    if (!materials.empty()) {
        ImGuiManager::tableRowLabel("Materials");

        bool anyMaterialChanged = false;

        const auto& textureNames = renderer.getTextureNames();

        auto drawTextureCombo = [&](const char* label, int& texIndex, int materialIndex) {
            int currentComboIndex = texIndex == -1 ? 0 : texIndex + 1;

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(label);

            ImGui::TableNextColumn();
            if (ImGui::BeginCombo((std::string("##") + label + std::to_string(materialIndex)).c_str(),
                currentComboIndex == 0 ? "No texture selected" : textureNames[currentComboIndex - 1].c_str()))
            {
                // No texture option
                bool isSelected = (currentComboIndex == 0);
                if (ImGui::Selectable("No texture selected", isSelected)) {
                    currentComboIndex = 0;
                    texIndex = -1;
                    anyMaterialChanged = true;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();

                // Texture options
                for (int t = 0; t < static_cast<int>(textureNames.size()); ++t) {
                    isSelected = currentComboIndex == t + 1;
                    if (ImGui::Selectable(textureNames[t].c_str(), isSelected)) {
                        currentComboIndex = t + 1;
                        texIndex = t;
                        anyMaterialChanged = true;
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }

                ImGui::EndCombo();
            }
        };

        for (int i = 0; i < (int)materials.size(); ++i) {
            Material& mat = materials[i];
            std::string label = "Material " + std::to_string(i);

            if (ImGui::TreeNode(label.c_str())) {
                ImGui::BeginTable("MaterialTable", 2, ImGuiTableFlags_SizingStretchProp);

                // Albedo texture + color
                drawTextureCombo("Albedo Texture", mat.albedoIndex, i);
                ImGui::TableNextRow();
                ImGuiManager::colorEdit3Row("Albedo Color", mat.albedo, [&](glm::vec3 v) {
                    mat.albedo = v; anyMaterialChanged = true;
                });
                ImGui::TableNextRow();

                // Specular texture + scalar
                drawTextureCombo("Specular Texture", mat.specularIndex, i);
                ImGui::TableNextRow();
                ImGuiManager::dragFloatRow("Specular", mat.specular, 0.01f, 0.0f, 1.0f, [&](float v) {
                    mat.specular = v; anyMaterialChanged = true;
                });
                ImGui::TableNextRow();

                // Metallic texture + scalar
                drawTextureCombo("Metallic Texture", mat.metallicIndex, i);
                ImGui::TableNextRow();
                ImGuiManager::dragFloatRow("Metallic", mat.metallic, 0.01f, 0.0f, 1.0f, [&](float v) {
                    mat.metallic = v; anyMaterialChanged = true;
                });
                ImGui::TableNextRow();

                // Roughness texture + scalar
                drawTextureCombo("Roughness Texture", mat.roughnessIndex, i);
                ImGui::TableNextRow();
                ImGuiManager::dragFloatRow("Roughness", mat.roughness, 0.01f, 0.0f, 1.0f, [&](float v) {
                    mat.roughness = v; anyMaterialChanged = true;
                });
                ImGui::TableNextRow();

                drawTextureCombo("Normal Texture", mat.normalIndex, i);
                ImGui::TableNextRow();

                ImGuiManager::dragFloatRow("IOR", mat.ior, 0.01f, 1.0f, 3.0f, [&](float v) {
                    mat.ior = v; anyMaterialChanged = true;
                });
                ImGui::TableNextRow();

                ImGuiManager::colorEdit3Row("Transmission", mat.transmission, [&](glm::vec3 v) {
                    mat.transmission = v; anyMaterialChanged = true;
                });
                ImGui::TableNextRow();

                ImGuiManager::colorEdit3Row("Emission", mat.emission, [&](glm::vec3 v) {
                    mat.emission = v; anyMaterialChanged = true;
                });

                ImGui::EndTable();
                ImGui::TreePop();
            }
        }

        if (anyMaterialChanged)
            updateMaterials();
    }
}

void MeshAsset::updateMaterials() {
    //TODO
    // Recreate materialBuffer with updated materials
    materialBuffer = Buffer{renderer.context, Buffer::Type::AccelInput, sizeof(Material) * materials.size(), materials.data()};
    renderer.rebuildMeshBuffer(); //to update the stored buffer address on the mesh
}