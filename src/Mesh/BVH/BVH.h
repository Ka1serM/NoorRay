#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Shaders/SharedStructs.h"
#include "Vulkan/Buffer.h"

class BVH
{
    // Temporary struct used only during the build process.
    struct PrimitiveInfo {
        int faceIndex;
        vec3 centroid;
        AABB bbox;
    };

    std::vector<BVHNode> nodes;
    Buffer nodesBuffer;

    // Non-owning pointers to the original mesh data
    const std::vector<Vertex>* pVertices = nullptr;
    const std::vector<uint32_t>* pIndices = nullptr;
    
    void buildIterative(std::vector<PrimitiveInfo>& primitiveInfo, const AABB& sceneBounds);
    bool findBestSplit(std::vector<PrimitiveInfo>& primitiveInfo, int start, int end, const AABB& bounds, int& bestAxis, int& bestSplitIndex);
    
public:
    void build(const Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices);
    uint64_t getBufferAddress() const { return nodesBuffer.getDeviceAddress(); }
    const std::vector<BVHNode>& getNodes() const { return nodes; }
};