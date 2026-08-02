#include "MeshAsset.h"
#include <cstring>
#include <type_traits>
#include <utility>
#include "Scene/Scene.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <numbers>

#include "Backend/CUDA/Checks.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "glm/gtc/type_ptr.inl"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Types.h>

namespace {
MeshGeometry copyGeometry(const std::vector<Vertex>& vertices,
    const std::vector<uint32_t>& indices, const std::vector<Face>& faces)
{
    MeshGeometry geometry;
    geometry.vertices =
        nr::rstd::vector<Vertex>(vertices.begin(), vertices.end());
    geometry.indices =
        nr::rstd::vector<uint32_t>(indices.begin(), indices.end());
    geometry.faces = nr::rstd::vector<Face>(faces.begin(), faces.end());
    return geometry;
}

MaterialX::DocumentPtr greyMaterial()
{
    return nr::materialx::defaultMaterial();
}
}

MeshAsset MeshAsset::CreateCube(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<MaterialX::DocumentPtr> materials;

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

MeshAsset MeshAsset::CreatePlane(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
    std::vector<Face> faces;
    std::vector<MaterialX::DocumentPtr> materials;

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

MeshAsset MeshAsset::CreateSphere(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material, uint32_t latSeg, uint32_t lonSeg) {
    // Ensure the sphere has enough segments to be properly formed.
    if (latSeg < 2 || lonSeg < 3)
        throw std::runtime_error("Sphere segments too low. Use at least 2 latitude and 3 longitude segments.");

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<MaterialX::DocumentPtr> materials;

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

MeshAsset MeshAsset::CreateDisk(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material, uint32_t segments) {
    if (segments < 3)
        throw std::runtime_error("Disk requires at least 3 segments");

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<MaterialX::DocumentPtr> materials;

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

MeshAsset::MeshAsset(Scene& scene, std::string name,
    const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
    const std::vector<Face>& faces, const std::vector<MaterialX::DocumentPtr>& materials)
    : MeshAsset(scene, std::move(name),
        copyGeometry(vertices, indices, faces), materials)
{
}

MeshAsset::MeshAsset(Scene& scene, std::string name,
    const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices,
    const std::vector<Face>& faces, std::vector<MaterialRef> materials)
    : MeshAsset(scene, std::move(name),
        copyGeometry(vertices, indices, faces), std::move(materials))
{
}

MeshAsset::MeshAsset(Scene& scene, std::string name, MeshGeometry&& geometry,
    const std::vector<MaterialX::DocumentPtr>& materials)
    : scene(&scene), path(std::move(name)),
      vertices(std::move(geometry.vertices)),
      indices(std::move(geometry.indices)),
      faces(std::move(geometry.faces))
{
    materialRefs.reserve(materials.size());
    for (const MaterialX::DocumentPtr& material : materials)
        materialRefs.push_back(scene.addMaterial(material));
    initializeMaterialIds();
    buildBlas();
}

MeshAsset::MeshAsset(Scene& scene, std::string name, MeshGeometry&& geometry,
    std::vector<MaterialRef> materials)
    : scene(&scene), path(std::move(name)),
      vertices(std::move(geometry.vertices)),
      indices(std::move(geometry.indices)),
      faces(std::move(geometry.faces)),
      materialRefs(std::move(materials))
{
    initializeMaterialIds();
    buildBlas();
}

void MeshAsset::initializeMaterialIds()
{
    materialIds.reserve(materialRefs.size());
    for (const MaterialRef& material : materialRefs) {
        if (!material.isValid())
            throw std::invalid_argument(
                "MeshAsset cannot reference a released material");
        materialIds.push_back(material.index());
    }
}

void MeshAsset::buildBlas()
{
    scene->synchronizeBeforeMutation();
    const auto& ctx = scene->getContext();
    if (ctx.getOptixContext() != nullptr && ctx.getCudaStream() != nullptr)
        blas.build(
            ctx.getOptixContext(), ctx.getCudaStream(),
            this->vertices.data(), static_cast<uint32_t>(this->vertices.size()), sizeof(Vertex),
            this->indices.data(), static_cast<uint32_t>(this->indices.size() / 3),
            this->faces.empty() ? nullptr : &this->faces.data()->materialIndex,
            sizeof(Face), static_cast<uint32_t>(materialIds.size()));
}

MeshAsset::MeshAsset(MeshAsset&& other) noexcept
    : scene(other.scene), path(std::move(other.path)), index(other.index),
      vertices(std::move(other.vertices)), indices(std::move(other.indices)),
      faces(std::move(other.faces)), materialIds(std::move(other.materialIds)),
      materialRefs(std::move(other.materialRefs)), blas(std::move(other.blas))
{
}

MeshAsset& MeshAsset::operator=(MeshAsset&& other) noexcept
{
    if (this != &other) {
        scene = other.scene;
        path = std::move(other.path);
        index = other.index;
        vertices = std::move(other.vertices);
        indices = std::move(other.indices);
        faces = std::move(other.faces);
        materialIds = std::move(other.materialIds);
        materialRefs = std::move(other.materialRefs);
        blas = std::move(other.blas);
    }
    return *this;
}

void MeshAsset::releaseResources()
{
    path.clear();
    index = ~0u;
    // Move-assign from an empty vector: clear() would keep the managed
    // allocation alive, and freeing it is the entire point here.
    vertices = nr::rstd::vector<Vertex>{};
    indices = nr::rstd::vector<uint32_t>{};
    faces = nr::rstd::vector<Face>{};
    materialIds = nr::rstd::vector<uint32_t>{};
    materialRefs.clear();
    blas.reset();
}

uint32_t MeshAsset::getMeshIndex() const {
    return index;
}

void MeshAsset::setMeshIndex(uint32_t newIndex) {
    index = newIndex;
}

void MeshAsset::updatePositions(const std::vector<glm::vec3>& positions)
{
    if (positions.size() != vertices.size())
        return;

    scene->synchronizeBeforeMutation();
    for (size_t i = 0; i < positions.size(); ++i)
        vertices[i].position = positions[i];

    const auto& context = scene->getContext();
    if (context.getOptixContext() != nullptr && context.getCudaStream() != nullptr)
        blas.refit(context.getOptixContext(), context.getCudaStream(),
            vertices.data(), static_cast<uint32_t>(vertices.size()), sizeof(Vertex),
            indices.data(), static_cast<uint32_t>(indices.size() / 3),
            faces.empty() ? nullptr : &faces.data()->materialIndex,
            sizeof(Face), static_cast<uint32_t>(materialIds.size()));

    scene->setDirtyFlag(Meshes);
    scene->setDirtyFlag(TLAS);
    scene->setDirtyFlag(Accumulation);
}

void MeshAsset::updateVertexData(const std::vector<Vertex>& newVertices)
{
    if (newVertices.size() != vertices.size())
        return;

    scene->synchronizeBeforeMutation();
    static_assert(std::is_trivially_copyable_v<Vertex>);
    std::memcpy(vertices.data(), newVertices.data(),
        newVertices.size() * sizeof(Vertex));

    const auto& context = scene->getContext();
    if (context.getOptixContext() != nullptr && context.getCudaStream() != nullptr)
        blas.refit(context.getOptixContext(), context.getCudaStream(),
            vertices.data(), static_cast<uint32_t>(vertices.size()), sizeof(Vertex),
            indices.data(), static_cast<uint32_t>(indices.size() / 3),
            faces.empty() ? nullptr : &faces.data()->materialIndex,
            sizeof(Face), static_cast<uint32_t>(materialIds.size()));

    scene->setDirtyFlag(Meshes);
    scene->setDirtyFlag(TLAS);
    scene->setDirtyFlag(Accumulation);
}

void MeshAsset::updatePositionsDevice(CUdeviceptr devicePositions,
    const uint32_t count, cudaStream_t stream)
{
    if (count != vertices.size())
        return;

    scene->synchronizeBeforeMutation();

    const size_t bytes = count * sizeof(glm::vec3);
    NR_GPU_CHECK(cudaMemcpyAsync(vertices.data(), reinterpret_cast<const void*>(devicePositions),
        bytes, cudaMemcpyDeviceToHost, stream));

    const auto& context = scene->getContext();
    if (context.getOptixContext() != nullptr && context.getCudaStream() != nullptr)
        blas.refit(context.getOptixContext(), stream,
            vertices.data(), static_cast<uint32_t>(vertices.size()), sizeof(Vertex),
            indices.data(), static_cast<uint32_t>(indices.size() / 3),
            faces.empty() ? nullptr : &faces.data()->materialIndex,
            sizeof(Face), static_cast<uint32_t>(materialIds.size()));

    scene->setDirtyFlag(Meshes);
    scene->setDirtyFlag(TLAS);
    scene->setDirtyFlag(Accumulation);
}

void MeshAsset::replaceGeometry(const std::vector<Vertex>& newVertices,
    const std::vector<uint32_t>& newIndices, const std::vector<Face>& newFaces,
    const uint32_t desiredMaterialSlotCount)
{
    replaceGeometry(copyGeometry(newVertices, newIndices, newFaces), desiredMaterialSlotCount);
}

void MeshAsset::replaceGeometry(MeshGeometry&& geometry, const uint32_t desiredMaterialSlotCount)
{
    scene->synchronizeBeforeMutation();
    vertices = std::move(geometry.vertices);
    indices = std::move(geometry.indices);
    faces = std::move(geometry.faces);
    if (desiredMaterialSlotCount > materialIds.size())
    {
        // Same native grey fallback the first-construction path uses --
        // BindMaterial (hdnoorray/renderParam.cpp) replaces it once the new
        // slot's real material Sprim has synced and published.
        const MaterialX::DocumentPtr fallbackDocument = greyMaterial();
        while (materialIds.size() < desiredMaterialSlotCount)
        {
            MaterialRef ref = scene->addMaterial(fallbackDocument);
            materialIds.push_back(ref.index());
            materialRefs.push_back(std::move(ref));
        }
    }
    const auto& context = scene->getContext();
    if (context.getOptixContext() != nullptr && context.getCudaStream() != nullptr)
        blas.build(context.getOptixContext(), context.getCudaStream(),
            vertices.data(), static_cast<uint32_t>(vertices.size()), sizeof(Vertex),
            indices.data(), static_cast<uint32_t>(indices.size() / 3),
            faces.empty() ? nullptr : &faces.data()->materialIndex,
            sizeof(Face), static_cast<uint32_t>(materialIds.size()));
    scene->setDirtyFlag(Meshes);
    scene->setDirtyFlag(TLAS);
    scene->setDirtyFlag(Accumulation);
}

void MeshAsset::setMaterial(const uint32_t materialIndex, const Material& material)
{
    if (materialIndex < materialRefs.size())
        scene->updateMaterial(materialRefs[materialIndex].handle(), material);
}

void MeshAsset::setMaterial(
    const uint32_t materialSlot, const MaterialRef& material)
{
    if (materialSlot >= materialIds.size() || !material.isValid())
        return;
    scene->synchronizeBeforeMutation();
    materialRefs[materialSlot] = material;
    materialIds[materialSlot] = material.index();
    notifyMaterialsChanged();
}

const Material& MeshAsset::getMaterial(const uint32_t slot) const
{
    return scene->getMaterial(materialRefs[slot].handle());
}

MaterialHandle MeshAsset::getMaterialHandle(const uint32_t slot) const
{
    return slot < materialRefs.size() ? materialRefs[slot].handle() : MaterialHandle();
}

void MeshAsset::notifyMaterialsChanged()
{
    scene->setDirtyFlag(Meshes);
    scene->setDirtyFlag(Accumulation);
}
