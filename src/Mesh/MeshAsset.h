#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Scene/Scene.h"
#include "Shaders/SharedStructs.h"
#include "UI/ImGuiComponent.h"
#include "Vulkan/Accel.h"
#include "Cpu/BVH.h"

class Scene;

class MeshAsset : public ImGuiComponent
{
public:
    std::string path;

    static std::shared_ptr<MeshAsset> CreateCube(Scene& scene, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreatePlane(Scene& scene, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreateSphere(Scene& scene, const std::string& name,  const Material& material, uint32_t latitudeSegments = 16, uint32_t longitudeSegments = 16);
    static std::shared_ptr<MeshAsset> CreateDisk(Scene& scene, const std::string& name, const Material& material, uint32_t segments = 16);
    
    // Constructor accepts const references to copy the data
    MeshAsset(Scene& context, const std::string& name,
              const std::vector<Vertex>& vertices,
              const std::vector<uint32_t>& indices,
              const std::vector<Face>& faces,
              const std::vector<Material>& materials);

    uint64_t getBlasAddress() const;
    MeshAddresses getBufferAddresses() const;

    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);
    void updateMaterials();
    void renderUi() override;

    Scene& scene;
    uint32_t index = -1;

    // CPU-side mesh data (copies)
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;

    // GPU buffers
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer faceBuffer;
    Buffer materialBuffer;

    // Acceleration structures
    Accel blasGpu;
    BVH blasCpu;
};