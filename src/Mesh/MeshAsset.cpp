#include "MeshAsset.h"
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <numbers>
#include "imgui.h"
#include "glm/gtc/type_ptr.inl"
#include "../UI/ImGui/ImGuiManager.h"

std::shared_ptr<MeshAsset> MeshAsset::CreateCube(Scene& scene, const std::string& name, const Material& material) {
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
        vec3 tangent = normalize(tangents[faceIdx]);
        vec3 bitangent = normalize(bitangents[faceIdx]);

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

    return std::make_shared<MeshAsset>(scene, name, vertices, indices, faces, materials);
}

std::shared_ptr<MeshAsset> MeshAsset::CreatePlane(Scene& scene, const std::string& name, const Material& material) {
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

    return std::make_shared<MeshAsset>(scene, name, vertices, indices, faces, materials);
}

std::shared_ptr<MeshAsset> MeshAsset::CreateSphere(Scene& scene, const std::string& name, const Material& material, uint32_t latSeg, uint32_t lonSeg) {
    // Ensure the sphere has enough segments to be properly formed.
    if (latSeg < 2 || lonSeg < 3)
        throw std::runtime_error("Sphere segments too low. Use at least 2 latitude and 3 longitude segments.");

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    constexpr float radius = 0.5f;

    // Generate vertices in a grid pattern based on spherical coordinates.
    // The grid is (lonSeg + 1) vertices wide and (latSeg + 1) vertices tall.
    for (uint32_t lat = 0; lat <= latSeg; ++lat) {
        float theta = std::numbers::pi_v<float> * lat / latSeg; // Polar angle from 0 to pi
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        for (uint32_t lon = 0; lon <= lonSeg; ++lon) {
            float phi = 2.0f * std::numbers::pi_v<float> * lon / lonSeg; // Azimuthal angle from 0 to 2*pi
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);

            // Calculate vertex attributes
            vec3 normal = {cosPhi * sinTheta, cosTheta, sinPhi * sinTheta};
            vec3 pos = normal * radius;
            vec2 uv = {static_cast<float>(lon) / lonSeg, static_cast<float>(lat) / latSeg};
            
            vec3 tangent;
            // At the poles, the derivative of position with respect to the azimuthal angle 'phi'
            // is zero, making the tangent undefined. All vertices on the top/bottom rows share
            // a single position but would get different tangents, causing lighting artifacts.
            // We assign a fixed, consistent tangent to all vertices at each pole.
            if (lat == 0) { // North Pole
                tangent = vec3{1.0f, 0.0f, 0.0f};
            } else if (lat == latSeg) { // South Pole
                tangent = vec3{-1.0f, 0.0f, 0.0f};
            } else {
                // For all other vertices, the tangent runs along lines of latitude.
                tangent = normalize(vec3{-sinPhi, 0.0f, cosPhi});
            }

            vertices.push_back(Vertex{
                pos, 0,
                normal, 0,
                tangent, 0,
                uv, 0, 0
            });
        }
    }

    // Generate indices to form triangles for each quad in the grid
    for (uint32_t lat = 0; lat < latSeg; ++lat) {
        for (uint32_t lon = 0; lon < lonSeg; ++lon) {
            uint32_t i0 = lat * (lonSeg + 1) + lon;      // Top-left
            uint32_t i1 = (lat + 1) * (lonSeg + 1) + lon; // Bottom-left
            uint32_t i2 = i0 + 1;                         // Top-right
            uint32_t i3 = i1 + 1;                         // Bottom-right

            // Create two triangles for the quad. The winding order is CCW (Counter-Clockwise).
            indices.insert(indices.end(), {i0, i2, i1, i2, i3, i1});
        }
    }

    materials.push_back(material);

    // Create a Face object for each generated triangle
    for (size_t i = 0; i < indices.size(); i += 3) {
        faces.push_back({0});
    }

    return std::make_shared<MeshAsset>(scene, name, vertices, indices, faces, materials);
}

std::shared_ptr<MeshAsset> MeshAsset::CreateDisk(Scene& scene, const std::string& name, const Material& material, uint32_t segments) {
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
        float angle = static_cast<float>(i) / segments * 2.0f * std::numbers::pi_v<float>;
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

    return std::make_shared<MeshAsset>(scene, name, vertices, indices, faces, materials);
}

MeshAsset::MeshAsset(Scene& scene, std::string  name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials)
    : scene(scene), path(std::move(name)), vertices(vertices), indices(indices), faces(faces), materials(materials)
{
    // Upload mesh data to GPU from the new member variable copies
    vertexBuffer = Buffer{scene.getContext(), Buffer::Type::AccelInput , sizeof(Vertex) * this->vertices.size(), this->vertices.data()};
    indexBuffer = Buffer{scene.getContext(), Buffer::Type::AccelInput , sizeof(uint32_t) * this->indices.size(), this->indices.data()};
    faceBuffer = Buffer{scene.getContext(), Buffer::Type::AccelInput , sizeof(Face) * this->faces.size(), this->faces.data()};
    materialBuffer = Buffer{scene.getContext(), Buffer::Type::AccelInput , sizeof(Material) * this->materials.size(), this->materials.data()};

    // Create bottom-level acceleration structure (BLAS)
    if (scene.getContext().isRtxSupported()) {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData{};
        triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangleData.setVertexData(vertexBuffer.getDeviceAddress());
        triangleData.setVertexStride(sizeof(Vertex));
        triangleData.setMaxVertex(static_cast<uint32_t>(this->vertices.size()));
        triangleData.setIndexType(vk::IndexType::eUint32);
        triangleData.setIndexData(indexBuffer.getDeviceAddress());

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometry.setGeometry({triangleData});
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
        blasRtx.build(scene.getContext(), geometry, this->faces.size(), vk::AccelerationStructureTypeKHR::eBottomLevel);
    }
    else
        blasCompute.build(scene.getContext(), this->vertices, this->indices);
}

uint64_t MeshAsset::getBlasAddress() const {
    // Only return a valid GPU BLAS address if RTX is supported
    if (scene.getContext().isRtxSupported())
        return blasRtx.getBuffer().getDeviceAddress();
    return 0;
}

MeshAddresses MeshAsset::getBufferAddresses() const {
    return MeshAddresses{
        vertexBuffer.getDeviceAddress(),
        indexBuffer.getDeviceAddress(),
        faceBuffer.getDeviceAddress(),
        materialBuffer.getDeviceAddress(),
        //only used for Compute RT
        blasCompute.getNodesBufferAddress(),
        blasCompute.getIndicesBufferAddress()
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
    ImGui::InputText("##meshPath", const_cast<char*>(path.c_str()), path.size() + 1, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopItemWidth();
    ImGui::PopStyleColor();

    if (materials.empty())
        return;
    
    ImGuiManager::tableRowLabel("Materials");

    bool anyMaterialChanged = false;
    
    const auto textureNames = scene.getTextureNames();

    // Helper lambda for drawing texture selection combos.
    auto drawTextureCombo = [&](const char* label, int& texIndex, const int materialIndex) {
        ImGui::TableNextRow();
        // The current index for the combo box. We add 1 because index 0 is our "No Texture" option.
        const int currentComboIndex = texIndex == -1 ? 0 : texIndex + 1;

        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label);

        ImGui::TableNextColumn();
        // Use a unique ID for each combo box.
        const std::string comboId = "##" + std::string(label) + std::to_string(materialIndex);
        
        const char* previewValue = (currentComboIndex == 0) ? "No Texture" : textureNames[currentComboIndex - 1].c_str();
        if (ImGui::BeginCombo(comboId.c_str(), previewValue))
        {
            if (ImGui::Selectable("No Texture", currentComboIndex == 0)) {
                if (texIndex != -1) { // Only mark as changed if it was different.
                    texIndex = -1;
                    anyMaterialChanged = true;
                }
            }

            // Loop through available textures.
            for (int t = 0; t < static_cast<int>(textureNames.size()); ++t) {
                if (ImGui::Selectable(textureNames[t].c_str(), texIndex == t)) {
                    if (texIndex != t) { // Only mark as changed if it was different.
                        texIndex = t;
                        anyMaterialChanged = true;
                    }
                }
            }
            ImGui::EndCombo();
        }
    };

       for (auto i = 0; i < materials.size(); ++i) {
        Material& mat = materials[i];
        std::string label = "Material " + std::to_string(i);
        if (ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_Framed)) {
            if (ImGui::BeginTable("MaterialTable", 2, ImGuiTableFlags_SizingStretchProp))
            {
                drawTextureCombo("Albedo Texture", mat.albedoIndex, i);
                ImGuiManager::colorEdit3Row("Albedo Color", mat.albedo, [&](const vec3 v) { mat.albedo = v; anyMaterialChanged = true; });


                drawTextureCombo("Specular Texture", mat.specularIndex, i);
                ImGuiManager::dragFloatRow("Specular", mat.specular, 0.01f, 0.0f, 1.0f, [&](const float v) { mat.specular = v; anyMaterialChanged = true; });


                drawTextureCombo("Metallic Texture", mat.metallicIndex, i);
                ImGuiManager::dragFloatRow("Metallic", mat.metallic, 0.01f, 0.0f, 1.0f, [&](const float v) { mat.metallic = v; anyMaterialChanged = true; });


                drawTextureCombo("Roughness Texture", mat.roughnessIndex, i);
                ImGuiManager::dragFloatRow("Roughness", mat.roughness, 0.01f, 0.0f, 1.0f, [&](const float v) { mat.roughness = v; anyMaterialChanged = true; });

                
                drawTextureCombo("Normal Texture", mat.normalIndex, i);

                
                ImGuiManager::dragFloatRow("IOR", mat.ior, 0.01f, 1.0f, 3.0f, [&](const float v) { mat.ior = v; anyMaterialChanged = true; });
                ImGuiManager::dragFloatRow("Transmission Strength", mat.transmission, 0.01f, 0.0f, 1.0f, [&](const float v) { mat.transmission = v; anyMaterialChanged = true; });
                drawTextureCombo("Transmission Texture", mat.transmissionIndex, i);
                ImGuiManager::colorEdit3Row("Transmission Color", mat.transmissionColor, [&](const vec3 v) { mat.transmissionColor = v; anyMaterialChanged = true; });

                
                ImGuiManager::dragFloatRow("Emission Strength", mat.emissionStrength, 0.1f, 0.0f, 100000.0f, [&](const float v) { mat.emissionStrength = v; anyMaterialChanged = true; });
                drawTextureCombo("Emission Texture", mat.emissionIndex, i);
                ImGuiManager::colorEdit3Row("Emission Color", mat.emission, [&](const vec3 v) { mat.emission = v; anyMaterialChanged = true; });


                ImGuiManager::dragFloatRow("Opacity", mat.opacity, 0.01f, 0.0f, 1.0f, [&](const float v) { mat.opacity = v; anyMaterialChanged = true; });
                drawTextureCombo("Opacity Texture", mat.opacityIndex, i);
                
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
    }

    if (anyMaterialChanged) {
        dirty = true;
        scene.setDirtyFlag(Meshes);
    }
}

//this gets called from the renderer when the scene material flag is dirty
void MeshAsset::updateMaterials() {
    materialBuffer = Buffer{scene.getContext(), Buffer::Type::Storage, sizeof(Material) * materials.size(), materials.data()};
    dirty = false; // Reset dirty flag after updating
    scene.setDirtyFlag(Accumulation);
}