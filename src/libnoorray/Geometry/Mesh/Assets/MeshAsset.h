#pragma once

#include <string>
#include <vector>

#include <cuda.h>

#include "Backend/CUDA/rstd/Vector.h"
#include "Backend/OptiX/Acceleration/Blas.h"
#include "Scene/Inspectable.h"
#include "Scene/Resources/MaterialRegistry.h"
#include "Materials/Shading/Material.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Geometry/Mesh/VertexColor.h"

using glm::vec2;
using glm::vec3;

class Scene;

namespace MaterialX_v1_39_4
{
class Document;
using DocumentPtr = std::shared_ptr<Document>;
}
namespace MaterialX = MaterialX_v1_39_4;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    float tangentSign;
    glm::vec2 uv;
    uint32_t color{nr::vertex_color::White};
};

struct Face
{
    int materialIndex{};
};

// Move-owned final geometry storage. Importers can fill these managed buffers
// directly on worker threads, then hand them to MeshAsset without a second
// std::vector -> managed-vector allocation and element copy.
struct MeshGeometry
{
    MeshGeometry() = default;
    MeshGeometry(const MeshGeometry&) = delete;
    MeshGeometry& operator=(const MeshGeometry&) = delete;
    MeshGeometry(MeshGeometry&&) noexcept = default;
    MeshGeometry& operator=(MeshGeometry&&) noexcept = default;

    nr::rstd::vector<Vertex> vertices;
    nr::rstd::vector<uint32_t> indices;
    nr::rstd::vector<Face> faces;
};

class MeshAsset : public Inspectable
{
public:
    // The material argument is a MaterialX document (the conversion from an
    // importer's simple authoring record happens in the caller). A null
    // document means the default MaterialX material.
    static MeshAsset CreateCube(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material);
    static MeshAsset CreatePlane(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material);
    static MeshAsset CreateSphere(Scene& scene, const std::string& name,  const MaterialX::DocumentPtr& material, uint32_t latitudeSegments = 64, uint32_t longitudeSegments = 64);
    static MeshAsset CreateDisk(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material, uint32_t segments = 64);
    
    MeshAsset(Scene& context, std::string  name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<MaterialX::DocumentPtr>& materials);
    MeshAsset(Scene& context, std::string name, const std::vector<Vertex>& vertices,
        const std::vector<uint32_t>& indices, const std::vector<Face>& faces,
        std::vector<MaterialRef> materials);
    MeshAsset(Scene& context, std::string name, MeshGeometry&& geometry,
        const std::vector<MaterialX::DocumentPtr>& materials);
    MeshAsset(Scene& context, std::string name, MeshGeometry&& geometry,
        std::vector<MaterialRef> materials);
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

    // desiredMaterialSlotCount: grows materialIds/materialRefs (each new slot
    // gets the same native grey fallback material construction uses) when the
    // new topology's Face.materialIndex values reference more slots than
    // this mesh currently has -- e.g. a live-edited mesh gaining an
    // HdGeomSubset. 0 (the default) means "no change", the common case where
    // topology changes but material count does not. Never shrinks: unused
    // trailing slots are harmless, and shrinking could invalidate a
    // Face.materialIndex the caller forgot to remap.
    void replaceGeometry(const std::vector<Vertex>& newVertices,
        const std::vector<uint32_t>& newIndices, const std::vector<Face>& newFaces,
        uint32_t desiredMaterialSlotCount = 0);
    // Adopts already-managed geometry without allocating or copying it. This
    // is the preferred integration point for loaders and Hydra adapters that
    // can prepare final buffers before serial Scene publication.
    void replaceGeometry(MeshGeometry&& geometry, uint32_t desiredMaterialSlotCount = 0);
    void setMaterial(uint32_t materialIndex, const Material& material);
    void setMaterial(uint32_t materialSlot, const MaterialRef& material);
    void notifyMaterialsChanged();

private:
    void initializeMaterialIds();
    void buildBlas();

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
