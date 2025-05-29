#pragma once

#include <string>
#include "Vulkan/Accel.h"
#include "Vulkan/Renderer.h"

class MeshAsset {

public:

    std::string name;

    MeshAsset(Context& context, Renderer& renderer, const std::string& objFilePath);
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
