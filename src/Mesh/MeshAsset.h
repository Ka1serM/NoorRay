#pragma once

#include <string>

#include "Shaders/SharedStructs.h"
#include "UI/ImGuiComponent.h"
#include "Vulkan/Accel.h"

class Renderer;

class MeshAsset : public ImGuiComponent {

public:
    std::string path;

    static std::shared_ptr<MeshAsset> CreateCube(Renderer& renderer, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreatePlane(Renderer& renderer, const std::string& name, const Material& material);
    static std::shared_ptr<MeshAsset> CreateSphere(Renderer& renderer, const std::string& name,  const Material& material, uint32_t latitudeSegments = 16, uint32_t longitudeSegments = 16);
    static std::shared_ptr<MeshAsset> CreateDisk(Renderer& renderer, const std::string& name, const Material& material, uint32_t segments = 16);
    static std::shared_ptr<MeshAsset> CreateFromObj(Renderer& renderer, const std::string& objFilePath);
    
    MeshAsset(Renderer& context,  const std::string& name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces, const std::vector<Material>& materials);
    uint64_t getBlasAddress() const;
    MeshAddresses getBufferAddresses() const;

    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);
    void updateMaterials();
    void renderUi() override;

private:

    Renderer& renderer;

    uint32_t index = -1;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer faceBuffer;

    std::vector<Material> materials;
    Buffer materialBuffer;

    Accel blas;
};
