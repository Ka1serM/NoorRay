#include "BVH.h"
#include <algorithm>
#include <limits>
#include <numeric>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

#define BVH_MAX_DEPTH 28

// Definition of BVHNode is in Shaders/SharedStructs.h

void BVH::build(Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices) {
    pVertices = &inputVertices;
    pIndices = &inputIndices;

    size_t faceCount = pIndices->size() / 3;
    if (faceCount == 0) {
        nodes.clear();
        return;
    }
    
    std::vector<PrimitiveInfo> primitiveInfo(faceCount);
    for (size_t i = 0; i < faceCount; ++i) {
        primitiveInfo[i].faceIndex = static_cast<int>(i);
        const glm::vec3& v0 = (*pVertices)[(*pIndices)[i * 3 + 0]].position;
        const glm::vec3& v1 = (*pVertices)[(*pIndices)[i * 3 + 1]].position;
        const glm::vec3& v2 = (*pVertices)[(*pIndices)[i * 3 + 2]].position;
        primitiveInfo[i].centroid = (v0 + v1 + v2) / 3.0f;
        primitiveInfo[i].bbox.expand(v0);
        primitiveInfo[i].bbox.expand(v1);
        primitiveInfo[i].bbox.expand(v2);
    }

    nodes.clear();
    nodes.reserve(faceCount * 2);

    buildRecursive(primitiveInfo, 0, faceCount, 0);
    if (nodes.empty())
        throw std::runtime_error("BVH build resulted in no nodes.");

    nodesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(BVHNode) * nodes.size(), nodes.data()};
}

uint64_t BVH::getBufferAddress() const
{
    return nodesBuffer.getDeviceAddress();
}

int BVH::buildRecursive(std::vector<PrimitiveInfo>& primitiveInfo, int start, int end, int depth) {
    int nodeIndex = static_cast<int>(nodes.size());
    nodes.emplace_back();
    BVHNode& node = nodes[nodeIndex];

    AABB bbox;
    for (int i = start; i < end; ++i)
        bbox.expand(primitiveInfo[i].bbox);
    node.bbox = bbox;

    int count = end - start;
    if (count <= BVH_MAX_LEAF_SIZE || depth >= BVH_MAX_DEPTH) {
        node.faceCount = count;
        node.leftChild = -1;
        node.rightChild = -1;
        for (int i = 0; i < count; ++i) {
            node.faceIndices[i] = primitiveInfo[start + i].faceIndex;
        }
        return nodeIndex;
    }

    int bestAxis = -1;
    int bestSplitIndex = -1;
    float bestCost = std::numeric_limits<float>::max();

    for (int axis = 0; axis < 3; ++axis) {
        std::sort(primitiveInfo.begin() + start, primitiveInfo.begin() + end,
            [axis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                return a.centroid[axis] < b.centroid[axis];
            });
        std::vector<AABB> rightBoxes(count);
        AABB currentBox;
        for (int i = count - 1; i > 0; --i) {
            currentBox.expand(primitiveInfo[start + i].bbox);
            rightBoxes[i - 1] = currentBox;
        }
        AABB leftBox;
        for (int i = 1; i < count; ++i) {
            leftBox.expand(primitiveInfo[start + i - 1].bbox);
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
        std::sort(primitiveInfo.begin() + start, primitiveInfo.begin() + end,
            [bestAxis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                return a.centroid[bestAxis] < b.centroid[bestAxis];
            });
    }

    int leftChildIndex = buildRecursive(primitiveInfo, start, bestSplitIndex, depth + 1);
    int rightChildIndex = buildRecursive(primitiveInfo, bestSplitIndex, end, depth + 1);

    node.leftChild = leftChildIndex;
    node.rightChild = rightChildIndex;
    node.faceCount = 0; 

    return nodeIndex;
}

bool BVH::intersect(const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin, float tMax) const {
    if (nodes.empty()) return false;
    hitInfo.t = tMax;
    hitInfo.primitiveIndex = -1;
    bool was_hit = false;
    intersectNode(0, origin, direction, hitInfo, tMin, was_hit);
    return was_hit;
}

void BVH::intersectNode(int nodeIndex, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo, float tMin, bool& hit) const {
    const BVHNode& node = nodes[nodeIndex];
    
    // FIX: The AABB intersection test MUST use the closest hit found so far (hitInfo.t)
    // to correctly prune branches.
    if (!node.bbox.intersect(origin, direction, tMin, hitInfo.t)) {
        return;
    }

    if (node.isLeaf()) {
        for (int i = 0; i < node.faceCount; ++i) {
            int faceIdx = node.faceIndices[i];
            HitInfo tempHit;
            if (intersectTriangle(faceIdx, origin, direction, tempHit) && tempHit.t >= tMin && tempHit.t < hitInfo.t) {
                // When a closer triangle is hit, hitInfo is updated.
                // This new, smaller hitInfo.t will be used for all subsequent AABB tests.
                hitInfo = tempHit;
                hitInfo.primitiveIndex = faceIdx;
                hit = true;
            }
        }
    } else {
        // FIX: Reverted to the simple and correct distance-based traversal order.
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

bool BVH::intersectTriangle(int faceIdx, const glm::vec3& origin, const glm::vec3& direction, HitInfo& hitInfo) const {
    uint32_t i0 = (*pIndices)[faceIdx * 3 + 0];
    uint32_t i1 = (*pIndices)[faceIdx * 3 + 1];
    uint32_t i2 = (*pIndices)[faceIdx * 3 + 2];
    const glm::vec3& v0 = (*pVertices)[i0].position;
    const glm::vec3& v1 = (*pVertices)[i1].position;
    const glm::vec3& v2 = (*pVertices)[i2].position;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(direction, edge2);
    float a = glm::dot(edge1, h);
    if (a > -1e-7f && a < 1e-7f) return false;
    float f = 1.0f / a;
    glm::vec3 s = origin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * glm::dot(edge2, q);
    if (t > 1e-7f) {
        hitInfo.t = t;
        hitInfo.barycentrics = glm::vec3(1.0f - u - v, u, v);
        return true;
    }
    return false;
}
