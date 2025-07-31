#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "Shaders/SharedStructs.h"
#include "Vulkan/Buffer.h"

class BVH {
private:
    // Ttemporary struct used only during the build process.
    struct PrimitiveInfo {
        int faceIndex;
        glm::vec3 centroid;
        AABB bbox;
    };

    std::vector<BVHNode> nodes;
    Buffer nodesBuffer;

    // Non-owning pointers to the original mesh data
    const std::vector<Vertex>* pVertices = nullptr;
    const std::vector<uint32_t>* pIndices = nullptr;
    
    int buildRecursive(std::vector<PrimitiveInfo>& primitiveInfo, int start, int end, int depth);
    void intersectNode(int nodeIndex, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin, bool& hit) const;
    bool intersectTriangle(int faceIdx, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo) const;
    
public:
    void build(Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices);
    bool intersect(const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin = 0.001f, float tMax = 10000.0f) const;
    uint64_t getBufferAddress() const;
};
