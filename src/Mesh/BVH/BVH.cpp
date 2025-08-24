#include "BVH.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stack>

#define GLM_ENABLE_EXPERIMENTAL

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