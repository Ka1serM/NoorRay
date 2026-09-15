#pragma once

#include <string>
#include <vector>
#include <memory>
#include <noorrhi/noorrhi.hpp>

#include "Materials/Material.h"
#include "Materials/MaterialX/MaterialXFwd.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Mesh/VertexColor.h"
#include "Shared/Mesh.h"

class Scene;

// The GPU layout is the only layout: `Mesh::vertices` is uploaded verbatim.
using Vertex = nr::graphics::Vertex;

// Vertex colour is a linear multiplier on albedo, so geometry without authored
// colours must be explicitly white -- a zero-initialised Vertex renders black.
inline constexpr glm::vec4 DefaultVertexColor{1.0f, 1.0f, 1.0f, 1.0f};

// A vertex with every field zeroed except the colour, which defaults to white.
inline Vertex defaultVertex()
{
    Vertex vertex{};
    vertex.color = DefaultVertexColor;
    return vertex;
}

using Face = nr::graphics::Face;

// Move-owned final geometry storage. Importers can fill these managed buffers
// directly on worker threads, then hand them to Mesh without a second
// std::vector -> managed-vector allocation and element copy.
struct MeshGeometry
{
    MeshGeometry() = default;
    MeshGeometry(const MeshGeometry&) = delete;
    MeshGeometry& operator=(const MeshGeometry&) = delete;
    MeshGeometry(MeshGeometry&&) noexcept = default;
    MeshGeometry& operator=(MeshGeometry&&) noexcept = default;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
};

// Each mesh owns its shader record, reached through the inherited `data`
// member. The Raytracer publishes a table of pointers to these records rather
// than copying the structs into one array, so a mesh that re-uploads does not
// force the whole table to be rebuilt - the same arrangement Material uses.
class Mesh : public noorrhi::Shared<nr::graphics::Mesh>
{
public:
    // The material argument is a MaterialX document (the conversion from an
    // importer's simple authoring record happens in the caller). A null
    // document means the default MaterialX material.
    static Mesh CreateCube(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material);
    static Mesh CreatePlane(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material);
    static Mesh CreateSphere(Scene& scene, const std::string& name,  const MaterialX::DocumentPtr& material, uint32_t latitudeSegments = 64, uint32_t longitudeSegments = 64);
    static Mesh CreateDisk(Scene& scene, const std::string& name, const MaterialX::DocumentPtr& material, uint32_t segments = 64);
    
    Mesh(Scene& context, std::string  name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<MaterialX::DocumentPtr>& materials);
    Mesh(Scene& context, std::string name, const std::vector<Vertex>& vertices,
        const std::vector<uint32_t>& indices, const std::vector<Face>& faces,
        std::vector<Material*> materials);
    Mesh(Scene& context, std::string name, MeshGeometry&& geometry,
        const std::vector<MaterialX::DocumentPtr>& materials);
    Mesh(Scene& context, std::string name, MeshGeometry&& geometry,
        std::vector<Material*> materials);
    Mesh(Mesh&& other) noexcept;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&& other) = delete;
    ~Mesh() = default;

    const std::string& getName() const { return path; }
    std::string getType() const { return "Mesh Asset"; }

    // Getters & Setters-
    const std::string& getPath() const { return path; }
    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);
    
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<uint32_t>& getIndices() const { return indices; }
    const std::vector<Face>& getFaces() const { return faces; }
    const std::vector<uint32_t>& getMaterialIds() const {
        return materialIds;
    }
    size_t getMaterialCount() const { return materialIds.size(); }
    const Material& getMaterial(uint32_t slot) const;
    Material* getMaterialPtr(uint32_t slot) const;
    Scene& getScene() const { return scene; }
    // Replaces the vertex data and refits the BLAS in place, which is much
    // cheaper than replaceGeometry. Topology (indices + faces) and material
    // count stay the same, so the caller must pass one vertex per existing
    // vertex. updatePositions/updateVertexData are conveniences over this.
    void setVertices(std::vector<Vertex> value);
    void updatePositions(const std::vector<glm::vec3>& positions);
    void updateVertexData(const std::vector<Vertex>& newVertices);

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
    void setMaterial(uint32_t materialSlot, Material* material);
    void notifyMaterialsChanged();

    void upload(noorrhi::Device& device);
    void releaseGpu();
    noorrhi::Buffer<std::uint32_t> indexBuffer;
    noorrhi::Buffer<nr::graphics::Vertex> vertexBuffer;
    noorrhi::Buffer<nr::graphics::Face> faceBuffer;
    noorrhi::Buffer<std::uint32_t> materialBuffer;
    noorrhi::AccelerationStructure blas;

private:
    void initializeMaterialIds(const std::vector<Material*>& materials);

    Scene& scene;
    std::string path;
    uint32_t index = ~0u;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<uint32_t> materialIds;

};
