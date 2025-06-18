#pragma once

#include <string>

#include "Shaders/SharedStructs.h"
#include "Vulkan/Accel.h"

class Renderer;

class MeshAsset {

public:

    std::string path;

    static std::shared_ptr<MeshAsset> CreateCube(Context& context, const std::string& name);
    static std::shared_ptr<MeshAsset> CreatePlane(Context& context, const std::string& name);
    static std::shared_ptr<MeshAsset> CreateSphere(Context& context, const std::string& name, uint32_t latitudeSegments = 16, uint32_t longitudeSegments = 16);
    static std::shared_ptr<MeshAsset> CreateFromObj(Context& context, Renderer& renderer, const std::string& objFilePath);

    
    MeshAsset(Context& context,  const std::string& name, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::vector<Face>& faces);
    uint64_t getBlasAddress() const;
    MeshAddresses getBufferAddresses() const;

    uint32_t getMeshIndex() const;
    void setMeshIndex(uint32_t newIndex);

private:

    uint32_t index = -1;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer faceBuffer;

    Accel blas;
};
