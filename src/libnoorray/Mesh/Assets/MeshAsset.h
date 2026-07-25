#pragma once

#include <string>
#include <vector>

#include <cuda.h>

#include "CUDA/rstd/Vector.h"
#include "Raytracing/Acceleration/Blas.h"
#include "Scene/Inspectable.h"
#include "Scene/MaterialRegistry.h"
#include "Shading/Material.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

using glm::vec2;
using glm::vec3;

class Scene;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    float tangentSign;
    glm::vec2 uv;
};

struct Face
{
    int materialIndex;
};

class MeshAsset : public Inspectable
{
public:
    static MeshAsset CreateCube(Scene& scene, const std::string& name, const Material& material);
    static MeshAsset CreatePlane(Scene& scene, const std::string& name, const Material& material);
    static MeshAsset CreateSphere(Scene& scene, const std::string& name,  const Material& material, uint32_t latitudeSegments = 64, uint32_t longitudeSegments = 64);
    static MeshAsset CreateDisk(Scene& scene, const std::string& name, const Material& material, uint32_t segments = 64);
    
    MeshAsset(Scene& context, std::string  name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials);
    MeshAsset(MeshAsset&& other) noexcept;
    MeshAsset(const MeshAsset&) = delete;
    MeshAsset& operator=(const MeshAsset&) = delete;
    MeshAsset& operator=(MeshAsset&& other) noexcept;
    ~MeshAsset() = default;

    // Frees the geometry and its acceleration structure and gives up the
    // material references, leaving an empty asset behind. Called by the
    // registry once nothing refers to this mesh any more.
    void releaseResources();

    const Blas& getBlas() const { return blas; }

    const std::string& getName() const override { return path; }
    std::string getType() const override { return "Mesh Asset"; }

    // Getters & Setters-
    const std::string& getPath() const { return path; }
    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);
    
    NR_CPU_GPU const nr::rstd::vector<Vertex>& getVertices() const { return vertices; }
    NR_CPU_GPU const nr::rstd::vector<uint32_t>& getIndices() const { return indices; }
    NR_CPU_GPU const nr::rstd::vector<Face>& getFaces() const { return faces; }
    NR_CPU_GPU const nr::rstd::vector<uint32_t>& getMaterialIds() const {
        return materialIds;
    }
    size_t getMaterialCount() const { return materialIds.size(); }
    const Material& getMaterial(uint32_t slot) const;
    MaterialHandle getMaterialHandle(uint32_t slot) const;
    Scene& getScene() const { return *scene; }
    // Incremental updates for real-time editing. These update vertex data
    // in-place and refit the BLAS (much cheaper than replaceGeometry).
    // Topology (indices + faces) and material count must stay the same.
    void updatePositions(const std::vector<glm::vec3>& positions);
    void updateVertexData(const std::vector<Vertex>& newVertices);

    // GPU-direct position upload. Copies positions from a device pointer into
    // the vertex buffer and refits the BLAS — no CPU staging required.
    void updatePositionsDevice(CUdeviceptr devicePositions, uint32_t count,
        cudaStream_t stream);

    void replaceGeometry(const std::vector<Vertex>& newVertices,
        const std::vector<uint32_t>& newIndices, const std::vector<Face>& newFaces);
    void setMaterial(uint32_t materialIndex, const Material& material);
    void setMaterial(uint32_t materialSlot, const MaterialRef& material);
    void notifyMaterialsChanged();

private:
    Scene* scene;
    std::string path;
    uint32_t index = ~0u;
    nr::rstd::vector<Vertex> vertices;
    nr::rstd::vector<uint32_t> indices;
    nr::rstd::vector<Face> faces;
    // Slot indices of the materials this mesh uses, read directly by the
    // shading kernels. materialRefs holds the matching ownership so a material
    // outlives every mesh that points at it.
    nr::rstd::vector<uint32_t> materialIds;
    std::vector<MaterialRef> materialRefs;

    Blas blas;

};
