#include "BVH.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stack>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

#define BVH_MAX_DEPTH 128

// Surface Area Heuristic constants
#define SAH_TRAVERSAL_COST 1.0f
#define SAH_INTERSECTION_COST 1.0f

void BVH::build(const Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices) {
    pVertices = &inputVertices;
    pIndices = &inputIndices;

    size_t faceCount = pIndices->size() / 3;
    if (faceCount == 0) {
        nodes.clear();
        return;
    }
    
    // Pre-allocate to avoid reallocations
    nodes.clear();
    nodes.reserve(faceCount * 2);
    
    // Build primitive info with better memory layout
    std::vector<PrimitiveInfo> primitiveInfo;
    primitiveInfo.reserve(faceCount);
    
    AABB sceneBounds;
    for (size_t i = 0; i < faceCount; ++i) {
        PrimitiveInfo info;
        info.faceIndex = static_cast<int>(i);
        
        const vec3& v0 = (*pVertices)[(*pIndices)[i * 3 + 0]].position;
        const vec3& v1 = (*pVertices)[(*pIndices)[i * 3 + 1]].position;
        const vec3& v2 = (*pVertices)[(*pIndices)[i * 3 + 2]].position;
        
        info.centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
        info.bbox.expand(v0);
        info.bbox.expand(v1);
        info.bbox.expand(v2);
        
        sceneBounds.expand(info.bbox);
        primitiveInfo.push_back(info);
    }

    // Use iterative build to avoid stack overflow and improve cache locality
    buildIterative(primitiveInfo, sceneBounds);
    
    if (nodes.empty())
        throw std::runtime_error("BVH build resulted in no nodes.");

    nodesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(BVHNode) * nodes.size(), nodes.data()};
}

void BVH::buildIterative(std::vector<PrimitiveInfo>& primitiveInfo, const AABB& sceneBounds) {
    struct BuildTask {
        int start, end, nodeIndex, depth;
        AABB bounds;
    };
    
    std::stack<BuildTask> buildStack;
    
    // Create root node
    int rootIndex = static_cast<int>(nodes.size());
    nodes.emplace_back();
    
    buildStack.push({0, static_cast<int>(primitiveInfo.size()), rootIndex, 0, sceneBounds});
    
    while (!buildStack.empty()) {
        BuildTask task = buildStack.top();
        buildStack.pop();
        
        BVHNode& node = nodes[task.nodeIndex];
        node.bbox = task.bounds;
        
        int count = task.end - task.start;
        
        // Create leaf node if we hit termination criteria
        if (count <= BVH_MAX_LEAF_SIZE || task.depth >= BVH_MAX_DEPTH) {
            node.faceCount = count;
            node.leftChild = -1;
            node.rightChild = -1;
            
            for (int i = 0; i < count && i < BVH_MAX_LEAF_SIZE; ++i) {
                node.faceIndices[i] = primitiveInfo[task.start + i].faceIndex;
            }
            continue;
        }
        
        // Find best split using optimized SAH
        int bestAxis, splitIndex;
        if (!findBestSplit(primitiveInfo, task.start, task.end, task.bounds, bestAxis, splitIndex)) {
            // Fallback to middle split if SAH fails
            splitIndex = task.start + count / 2;
        }
        
        // Partition primitives
        if (bestAxis != -1) {
            std::nth_element(primitiveInfo.begin() + task.start, 
                           primitiveInfo.begin() + splitIndex,
                           primitiveInfo.begin() + task.end,
                           [bestAxis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                               return a.centroid[bestAxis] < b.centroid[bestAxis];
                           });
        }
        
        // Calculate bounds for children
        AABB leftBounds, rightBounds;
        for (int i = task.start; i < splitIndex; ++i) {
            leftBounds.expand(primitiveInfo[i].bbox);
        }
        for (int i = splitIndex; i < task.end; ++i) {
            rightBounds.expand(primitiveInfo[i].bbox);
        }
        
        // Create child nodes
        int leftChildIndex = static_cast<int>(nodes.size());
        nodes.emplace_back();
        int rightChildIndex = static_cast<int>(nodes.size());
        nodes.emplace_back();
        
        node.leftChild = leftChildIndex;
        node.rightChild = rightChildIndex;
        node.faceCount = 0;
        
        // Add child tasks to stack (right first for depth-first order)
        buildStack.push({splitIndex, task.end, rightChildIndex, task.depth + 1, rightBounds});
        buildStack.push({task.start, splitIndex, leftChildIndex, task.depth + 1, leftBounds});
    }
}

bool BVH::findBestSplit(std::vector<PrimitiveInfo>& primitiveInfo, int start, int end, 
                       const AABB& bounds, int& bestAxis, int& bestSplitIndex) {
    int count = end - start;
    if (count <= 2) return false;
    
    bestAxis = -1;
    bestSplitIndex = start + count / 2;
    float bestCost = std::numeric_limits<float>::max();
    
    // Try each axis
    for (int axis = 0; axis < 3; ++axis) {
        // Skip degenerate axes
        if (bounds.max[axis] - bounds.min[axis] < 1e-6f) continue;
        
        // Sort by centroid along this axis
        std::sort(primitiveInfo.begin() + start, primitiveInfo.begin() + end,
                  [axis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                      return a.centroid[axis] < b.centroid[axis];
                  });
        
        // Pre-compute right bounds
        std::vector<AABB> rightBounds(count - 1);
        AABB currentBox;
        for (int i = count - 1; i > 0; --i) {
            currentBox.expand(primitiveInfo[start + i].bbox);
            rightBounds[i - 1] = currentBox;
        }
        
        // Evaluate splits
        AABB leftBox;
        for (int i = 1; i < count; ++i) {
            leftBox.expand(primitiveInfo[start + i - 1].bbox);
            
            float leftArea = leftBox.surfaceArea();
            float rightArea = rightBounds[i - 1].surfaceArea();
            float cost = SAH_TRAVERSAL_COST + SAH_INTERSECTION_COST * 
                        (i * leftArea + (count - i) * rightArea) / bounds.surfaceArea();
            
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplitIndex = start + i;
            }
        }
    }
    
    // Check if split is beneficial
    float leafCost = SAH_INTERSECTION_COST * count;
    return bestCost < leafCost;
}

uint64_t BVH::getBufferAddress() const {
    return nodesBuffer.getDeviceAddress();
}

bool BVH::intersect(const vec3& origin, const vec3& direction, HitInfo& hitInfo, float tMin, float tMax) const {
    if (nodes.empty()) return false;
    
    hitInfo.t = tMax;
    hitInfo.primitiveIndex = -1;
    
    // Use iterative traversal to avoid recursion overhead and stack overflow
    return intersectIterative(origin, direction, hitInfo, tMin);
}

bool BVH::intersectIterative(const vec3& origin, const vec3& direction, HitInfo& hitInfo, float tMin) const {
    struct TraversalNode {
        int nodeIndex;
        float tNear;
    };
    
    // Pre-compute direction reciprocals for faster AABB tests
    vec3 invDir = 1.0f / direction;
    ivec3 dirIsNeg = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    
    std::stack<TraversalNode> nodeStack;
    nodeStack.push({0, tMin});
    
    bool hit = false;
    
    while (!nodeStack.empty()) {
        TraversalNode current = nodeStack.top();
        nodeStack.pop();
        
        // Skip if we've found a closer hit
        if (current.tNear >= hitInfo.t) continue;
        
        const BVHNode& node = nodes[current.nodeIndex];
        
        if (node.isLeaf()) {
            // Test triangles in leaf
            for (int i = 0; i < node.faceCount; ++i) {
                int faceIdx = node.faceIndices[i];
                HitInfo tempHit;
                if (intersectTriangle(faceIdx, origin, direction, tempHit) && 
                    tempHit.t >= tMin && tempHit.t < hitInfo.t) {
                    hitInfo = tempHit;
                    hitInfo.primitiveIndex = faceIdx;
                    hit = true;
                }
            }
        } else {
            // Internal node - test children
            const BVHNode& leftChild = nodes[node.leftChild];
            const BVHNode& rightChild = nodes[node.rightChild];
            
            float tNearLeft, tFarLeft, tNearRight, tFarRight;
            bool hitLeft = leftChild.bbox.intersect(origin, invDir, dirIsNeg, tNearLeft, tFarLeft);
            bool hitRight = rightChild.bbox.intersect(origin, invDir, dirIsNeg, tNearRight, tFarRight);
            
            // Clamp to valid range
            if (hitLeft) {
                tNearLeft = std::max(tNearLeft, tMin);
                hitLeft = tNearLeft < std::min(tFarLeft, hitInfo.t);
            }
            if (hitRight) {
                tNearRight = std::max(tNearRight, tMin);
                hitRight = tNearRight < std::min(tFarRight, hitInfo.t);
            }
            
            // Traverse closer child first
            if (hitLeft && hitRight) {
                if (tNearLeft > tNearRight) {
                    nodeStack.push({node.leftChild, tNearLeft});
                    nodeStack.push({node.rightChild, tNearRight});
                } else {
                    nodeStack.push({node.rightChild, tNearRight});
                    nodeStack.push({node.leftChild, tNearLeft});
                }
            } else if (hitLeft) {
                nodeStack.push({node.leftChild, tNearLeft});
            } else if (hitRight) {
                nodeStack.push({node.rightChild, tNearRight});
            }
        }
    }
    
    return hit;
}

bool BVH::intersectTriangle(int faceIdx, const vec3& origin, const vec3& direction, HitInfo& hitInfo) const {
    uint32_t i0 = (*pIndices)[faceIdx * 3 + 0];
    uint32_t i1 = (*pIndices)[faceIdx * 3 + 1];
    uint32_t i2 = (*pIndices)[faceIdx * 3 + 2];
    
    const vec3& v0 = (*pVertices)[i0].position;
    const vec3& v1 = (*pVertices)[i1].position;
    const vec3& v2 = (*pVertices)[i2].position;
    
    // Möller-Trumbore intersection algorithm
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(direction, edge2);
    float a = dot(edge1, h);
    
    if (a > -1e-7f && a < 1e-7f) return false; // Ray parallel to triangle
    
    float f = 1.0f / a;
    vec3 s = origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return false;
    
    vec3 q = cross(s, edge1);
    float v = f * dot(direction, q);
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    float t = f * dot(edge2, q);
    
    if (t > 1e-7f) {
        hitInfo.t = t;
        hitInfo.barycentrics = vec3(1.0f - u - v, u, v);
        return true;
    }
    
    return false;
}