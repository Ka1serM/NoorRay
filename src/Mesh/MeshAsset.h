#pragma once

#include <memory>
#include "Scene/Scene.h"
#include "Vulkan/Accel.h"

class MeshAsset {

public:

    std::string name;

    MeshAsset(const std::shared_ptr<Context> &context, const std::shared_ptr<Scene> &scene, const std::string& objFilePath);

    uint64_t getBlasAddress() const;

    MeshAddresses getBufferAddresses() const;

    uint32_t getMeshIndex() const;

    void setMeshIndex(uint32_t newIndex);

private:
    std::shared_ptr<Context> context;
    std::shared_ptr<Scene> scene;

    uint32_t index = -1;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer faceBuffer;

    std::unique_ptr<Accel> blas;
};
