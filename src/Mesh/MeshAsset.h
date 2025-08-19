#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Scene/Scene.h"
#include "../Shaders/SharedStructs.h"
#include "UI/ImGuiComponent.h"
#include "Vulkan/Accel.h"
#include "BVH/BVH.h"

class Scene;

class MeshAsset : public ImGuiComponent
{
public:
    static std::shared_ptr<MeshAsset> CreateCube(Scene& scene, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreatePlane(Scene& scene, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreateSphere(Scene& scene, const std::string& name,  const Material& material, uint32_t latitudeSegments = 16, uint32_t longitudeSegments = 16);
    static std::shared_ptr<MeshAsset> CreateDisk(Scene& scene, const std::string& name, const Material& material, uint32_t segments = 16);
    
    MeshAsset(Scene& context, const std::string& name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials);

    void renderUi() override;
    void updateMaterials();

    // Getters & Setters-
    const std::string& getPath() const { return path; }
    uint64_t getBlasAddress() const;
    MeshAddresses getBufferAddresses() const;
    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);
    
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<uint32_t>& getIndices() const { return indices; }
    const std::vector<Face>& getFaces() const { return faces; }
    const std::vector<Material>& getMaterials() const { return materials; }

    const Buffer& getVertexBuffer() const { return vertexBuffer; }
    const Buffer& getIndexBuffer() const { return indexBuffer; }
    const Buffer& getFaceBuffer() const { return faceBuffer; }
    const Buffer& getMaterialBuffer() const { return materialBuffer; }
    
    const Accel& getBlasGpu() const { return blasGpu; }
    const BVH& getBlasCpu() const { return blasCpu; }
    
    // Dirty Flag
    bool isDirty() const { return dirty; }
    void clearDirtyFlag() { dirty = false; }

private:
    Scene& scene;
    std::string path;
    uint32_t index = -1;
    bool dirty = false;

    // CPU-side data
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    // GPU-side data
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer faceBuffer;
    Buffer materialBuffer;

    // Acceleration structures
    Accel blasGpu;
    BVH blasCpu;
};