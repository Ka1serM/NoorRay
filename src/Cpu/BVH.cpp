#include "BVH.h"
#include <algorithm>
#include <limits>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

// --- Triangle Constructor Implementation ---
Triangle::Triangle(int i0, int i1, int i2, const std::vector<Vertex>& vertices)
    : indices{ i0, i1, i2 } {
    const glm::vec3& v0 = vertices[i0].position;
    const glm::vec3& v1 = vertices[i1].position;
    const glm::vec3& v2 = vertices[i2].position;

    bbox.expand(v0);
    bbox.expand(v1);
    bbox.expand(v2);

    centroid = (v0 + v1 + v2) / 3.0f;
}


// --- BVH Method Implementations ---

void BVH::build(const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices) {
    // Store pointers to the original data
    pVertices = &inputVertices;
    pIndices = &inputIndices;

    // Use the  pointers to create triangles
    triangles.clear();
    triangles.reserve(pIndices->size() / 3);
    for (size_t i = 0; i < pIndices->size(); i += 3)
        triangles.emplace_back((*pIndices)[i], (*pIndices)[i + 1], (*pIndices)[i + 2], *pVertices);

    triangleIndices.clear();
    triangleIndices.resize(triangles.size());
    for (int i = 0; i < triangles.size(); ++i)
        triangleIndices[i] = i;

    nodes.clear();
    nodes.reserve(triangles.size() * 2);

    buildRecursive(0, triangles.size(), 0);
}

bool BVH::intersect(const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin, float tMax) const {
    if (nodes.empty())
        return false;

    hitInfo.t = tMax;
    hitInfo.primitiveIndex = -1;
    bool was_hit = false;

    intersectNode(0, origin, direction, hitInfo, tMin, was_hit);
    return was_hit;
}

int BVH::buildRecursive(int start, int end, int depth) {
    int nodeIndex = nodes.size();
    nodes.emplace_back();
    BVHNode& node = nodes[nodeIndex];

    for (int i = start; i < end; ++i)
        node.bbox.expand(triangles[triangleIndices[i]].bbox);

    int count = end - start;
    if (count <= MAX_LEAF_SIZE || depth >= MAX_DEPTH) {
        node.firstPrimitive = start;
        node.primitiveCount = count;
        return nodeIndex;
    }

    int bestAxis = -1;
    int bestSplitIndex = -1;
    float bestCost = std::numeric_limits<float>::max();

    for (int axis = 0; axis < 3; ++axis) {
        std::sort(triangleIndices.begin() + start, triangleIndices.begin() + end,
            [this, axis](int a, int b) {
                return triangles[a].centroid[axis] < triangles[b].centroid[axis];
            });

        std::vector<AABB> rightBoxes(count);
        AABB currentBox;
        for (int i = count - 1; i > 0; --i) {
            currentBox.expand(triangles[triangleIndices[start + i]].bbox);
            rightBoxes[i - 1] = currentBox;
        }

        AABB leftBox;
        for (int i = 1; i < count; ++i) {
            leftBox.expand(triangles[triangleIndices[start + i - 1]].bbox);
            float cost = i * leftBox.surfaceArea() + (count - i) * rightBoxes[i - 1].surfaceArea();
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplitIndex = start + i;
            }
        }
    }

    if (bestAxis == -1) {
        bestSplitIndex = start + count / 2;
    } else {
        std::sort(triangleIndices.begin() + start, triangleIndices.begin() + end,
            [this, bestAxis](int a, int b) {
                return triangles[a].centroid[bestAxis] < triangles[b].centroid[bestAxis];
            });
    }

    int leftChildIndex = buildRecursive(start, bestSplitIndex, depth + 1);
    int rightChildIndex = buildRecursive(bestSplitIndex, end, depth + 1);

    node.leftChild = leftChildIndex;
    node.rightChild = rightChildIndex;

    return nodeIndex;
}

void BVH::intersectNode(int nodeIndex, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin, bool& hit) const {
    const BVHNode& node = nodes[nodeIndex];

    if (!node.bbox.intersect(origin, direction, tMin, hitInfo.t))
        return;

    if (node.isLeaf()) {
        for (int i = 0; i < node.primitiveCount; ++i) {
            int triIdx = triangleIndices[node.firstPrimitive + i];
            HitInfo tempHit;
            if (intersectTriangle(triIdx, origin, direction, tempHit) && tempHit.t >= tMin && tempHit.t < hitInfo.t) {
                hitInfo = tempHit;
                hitInfo.primitiveIndex = triIdx;
                hit = true;
            }
        }
    } else {
        const BVHNode& leftChild = nodes[node.leftChild];
        const BVHNode& rightChild = nodes[node.rightChild];

        float dist1 = glm::distance2(leftChild.bbox.min, origin);
        float dist2 = glm::distance2(rightChild.bbox.min, origin);

        if (dist1 < dist2) {
            intersectNode(node.leftChild, origin, direction, hitInfo, tMin, hit);
            intersectNode(node.rightChild, origin, direction, hitInfo, tMin, hit);
        } else {
            intersectNode(node.rightChild, origin, direction, hitInfo, tMin, hit);
            intersectNode(node.leftChild, origin, direction, hitInfo, tMin, hit);
        }
    }
}

// Möller–Trumbore intersection algorithm
bool BVH::intersectTriangle(int triIdx, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo) const {
    const Triangle& tri = triangles[triIdx];

    // Dereference the pVertices pointer and access the .position member
    const glm::vec3& v0 = (*pVertices)[tri.indices[0]].position;
    const glm::vec3& v1 = (*pVertices)[tri.indices[1]].position;
    const glm::vec3& v2 = (*pVertices)[tri.indices[2]].position;

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -1e-7f && a < 1e-7f) {
        return false;
    }

    float f = 1.0f / a;
    glm::vec3 s = origin - v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(direction, q);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = f * glm::dot(edge2, q);

    if (t > 1e-7f) {
        hitInfo.t = t;
        hitInfo.barycentrics = glm::vec3(1.0f - u - v, u, v);
        return true;
    }

    return false;
}