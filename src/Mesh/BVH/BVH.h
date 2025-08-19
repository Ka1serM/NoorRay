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
    
    bool intersectTriangle(int faceIdx, const vec3& origin, const vec3& direction, HitInfo& hitInfo) const;
    void buildIterative(std::vector<PrimitiveInfo>& primitiveInfo, const AABB& sceneBounds);
    bool findBestSplit(std::vector<PrimitiveInfo>& primitiveInfo, int start, int end, const AABB& bounds, int& bestAxis, int& bestSplitIndex);
    bool intersectIterative(const vec3& origin, const vec3& direction, HitInfo& hitInfo, float tMin) const;
    
public:
    void build(const Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices);
    bool intersect(const vec3& origin, const vec3& direction, HitInfo& hitInfo, float tMin = 0.001f, float tMax = 10000.0f) const;
    uint64_t getBufferAddress() const;
};