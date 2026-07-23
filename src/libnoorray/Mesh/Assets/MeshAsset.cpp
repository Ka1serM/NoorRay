#include "MeshAsset.h"
#include <utility>
#include "Scene/Scene.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <numbers>

#include "glm/gtc/type_ptr.inl"

MeshAsset MeshAsset::CreateCube(Scene& scene, const std::string& name, const Material& material) {
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
                corners[i],
                normal,
                tangent,
                1.0f, uvs[i]
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

    return MeshAsset(scene, name, vertices, indices, faces, materials);
}

MeshAsset MeshAsset::CreatePlane(Scene& scene, const std::string& name, const Material& material) {
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
            positions[i],
            normal,
            tangent,
            1.0f, uvs[i]
        });
    }

    materials.push_back(material);

    for (size_t i = 0; i < indices.size(); i += 3) {
        faces.push_back({0});
    }

    return MeshAsset(scene, name, vertices, indices, faces, materials);
}

MeshAsset MeshAsset::CreateSphere(Scene& scene, const std::string& name, const Material& material, uint32_t latSeg, uint32_t lonSeg) {
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
                pos,
                normal,
                tangent,
                1.0f, uv
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

    return MeshAsset(scene, name, vertices, indices, faces, materials);
}

MeshAsset MeshAsset::CreateDisk(Scene& scene, const std::string& name, const Material& material, uint32_t segments) {
    if (segments < 3)
        throw std::runtime_error("Disk requires at least 3 segments");

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    float radius = 0.5f;
    vec3 normal = {0.0f, 1.0f, 0.0f};

    vertices.push_back(Vertex{
        {0.0f, 0.0f, 0.0f},
        normal,
        {1.0f, 0.0f, 0.0f},
        1.0f, {0.5f, 0.5f}
    }); // center vertex with tangent along +X

    for (uint32_t i = 0; i <= segments; ++i) {
        float angle = static_cast<float>(i) / segments * 2.0f * std::numbers::pi_v<float>;
        float x = std::cos(angle) * radius;
        float z = std::sin(angle) * radius;

        vec3 pos = {x, 0.0f, z};
        vec3 tangent = {-std::sin(angle), 0.0f, std::cos(angle)};

        vec2 uv = {0.5f + x, 0.5f + z};

        vertices.push_back(Vertex{
            pos,
            normal,
            tangent,
            1.0f, uv
        });
    }

    for (uint32_t i = 1; i <= segments; ++i) {
        indices.insert(indices.end(), {0, i, i + 1});
        faces.push_back({0});
    }

    materials.push_back(material);

    return MeshAsset(scene, name, vertices, indices, faces, materials);
}

MeshAsset::MeshAsset(Scene& scene, std::string  name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials)
    : scene(scene), path(std::move(name)),
      vertices(vertices.begin(), vertices.end()), indices(indices.begin(), indices.end()),
      faces(faces.begin(), faces.end())
{
    for (const Material& material : materials)
        materialIds.push_back(scene.add(material));
    scene.synchronizeBeforeMutation();
    const auto& ctx = scene.getContext();
    if (ctx.getOptixContext() != nullptr && ctx.getCudaStream() != nullptr)
        blas.build(
            ctx.getOptixContext(), ctx.getCudaStream(),
            this->vertices.data(), static_cast<uint32_t>(this->vertices.size()), sizeof(Vertex),
            this->indices.data(), static_cast<uint32_t>(this->indices.size() / 3));
}

MeshAsset::MeshAsset(MeshAsset&& other) noexcept
    : scene(other.scene), path(std::move(other.path)), index(other.index),
      vertices(std::move(other.vertices)), indices(std::move(other.indices)),
      faces(std::move(other.faces)), materialIds(std::move(other.materialIds)),
      blas(std::move(other.blas))
{
}

uint32_t MeshAsset::getMeshIndex() const {
    return index;
}

void MeshAsset::setMeshIndex(uint32_t newIndex) {
    index = newIndex;
}

void MeshAsset::replaceGeometry(const std::vector<Vertex>& newVertices,
    const std::vector<uint32_t>& newIndices, const std::vector<Face>& newFaces)
{
    scene.synchronizeBeforeMutation();
    vertices = nr::rstd::vector<Vertex>(newVertices.begin(), newVertices.end());
    indices = nr::rstd::vector<uint32_t>(newIndices.begin(), newIndices.end());
    faces = nr::rstd::vector<Face>(newFaces.begin(), newFaces.end());
    const auto& context = scene.getContext();
    if (context.getOptixContext() != nullptr && context.getCudaStream() != nullptr)
        blas.build(context.getOptixContext(), context.getCudaStream(),
            vertices.data(), static_cast<uint32_t>(vertices.size()), sizeof(Vertex),
            indices.data(), static_cast<uint32_t>(indices.size() / 3));
    scene.setDirtyFlag(Meshes);
    scene.setDirtyFlag(TLAS);
    scene.setDirtyFlag(Accumulation);
}

void MeshAsset::setMaterial(const uint32_t materialIndex, const Material& material)
{
    if (materialIndex < materialIds.size())
        scene.updateMaterial(materialIds[materialIndex], material);
}

void MeshAsset::setMaterialId(
    const uint32_t materialSlot, const uint32_t materialId)
{
    if (materialSlot >= materialIds.size()
        || materialId >= scene.getMaterials().size())
        return;
    scene.synchronizeBeforeMutation();
    materialIds[materialSlot] = materialId;
    notifyMaterialsChanged();
}

const Material& MeshAsset::getMaterial(const uint32_t slot) const
{
    return scene.getMaterial(materialIds[slot]);
}

void MeshAsset::notifyMaterialsChanged()
{
    scene.setDirtyFlag(Meshes);
    scene.setDirtyFlag(Accumulation);
}
