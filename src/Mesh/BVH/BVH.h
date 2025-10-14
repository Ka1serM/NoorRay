#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Shaders/SharedStructs.h"
#include "Vulkan/Buffer.h"

class BVH
{
public:
    void build(const Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices);
    void build(const Context& context, const AABB& aabb);
    uint64_t getNodesBufferAddress() const { return nodesBuffer.getDeviceAddress(); }
    uint64_t getIndicesBufferAddress() const { return indicesBuffer.getDeviceAddress(); }

private:
    struct PrimitiveInfo {
        uint32_t faceIndex;
        AABB bounds;
        vec3 centroid;
    };

    uint32_t recursiveBuild(std::vector<PrimitiveInfo>& primitiveInfo, uint32_t start, uint32_t end);
    
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> orderedFaceIndices;

    // GPU buffers
    Buffer nodesBuffer;
    Buffer indicesBuffer;

    // Non-owning pointers
    const std::vector<Vertex>* pVertices = nullptr;
    const std::vector<uint32_t>* pIndices = nullptr;
};