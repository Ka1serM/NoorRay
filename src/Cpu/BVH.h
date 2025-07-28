#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "AABB.h"
#include "Shaders/SharedStructs.h" // For HitInfo and Vertex

// Represents a single triangle with pre-calculated data (internal to BVH)
struct Triangle {
    int indices[3];
    AABB bbox;
    glm::vec3 centroid;

    Triangle(int i0, int i1, int i2, const std::vector<Vertex>& vertices);
};

// A node in the BVH tree (internal to BVH)
struct BVHNode {
    AABB bbox;
    int leftChild = -1;
    int rightChild = -1;
    int firstPrimitive = -1;
    int primitiveCount = 0;

    bool isLeaf() const { return primitiveCount > 0; }
};

class BVH {
public:
    void build(const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices);
    bool intersect(const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin = 0.001f, float tMax = 10000.0f) const;

private:
    // Non-owning pointers to the mesh data
    const std::vector<Vertex>* pVertices = nullptr;
    const std::vector<uint32_t>* pIndices = nullptr;

    // Internal data structures for the BVH
    std::vector<Triangle> triangles;
    std::vector<BVHNode> nodes;
    std::vector<int> triangleIndices; // Indices into the 'triangles' vector

    static const int MAX_LEAF_SIZE = 4;
    static const int MAX_DEPTH = 28;

    int buildRecursive(int start, int end, int depth);
    void intersectNode(int nodeIndex, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin, bool& hit) const;
    bool intersectTriangle(int triIdx, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo) const;
};