#include "BVH.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

void BVH::build(const Context& context, const std::vector<Vertex>& inputVertices, const std::vector<uint32_t>& inputIndices) {
    pVertices = &inputVertices;
    pIndices = &inputIndices;
    const size_t faceCount = pIndices->size() / 3;

    if (faceCount == 0) {
        nodes.clear();
        orderedFaceIndices.clear();
        return;
    }

    // 1. Create primitive info for each face
    std::vector<PrimitiveInfo> primitiveInfo(faceCount);
    for (size_t i = 0; i < faceCount; ++i) {
        primitiveInfo[i].faceIndex = static_cast<uint32_t>(i);
        const auto& v0 = (*pVertices)[(*pIndices)[i * 3 + 0]].position;
        const auto& v1 = (*pVertices)[(*pIndices)[i * 3 + 1]].position;
        const auto& v2 = (*pVertices)[(*pIndices)[i * 3 + 2]].position;
        primitiveInfo[i].bounds.expand(v0);
        primitiveInfo[i].bounds.expand(v1);
        primitiveInfo[i].bounds.expand(v2);
        primitiveInfo[i].centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
    }

    // 2. Pre-allocate memory to avoid reallocations during build
    nodes.clear();
    nodes.reserve(faceCount * 2);
    orderedFaceIndices.clear();
    orderedFaceIndices.reserve(faceCount);
    
    // 3. Start the recursive build process
    recursiveBuild(primitiveInfo, 0, static_cast<uint32_t>(faceCount));

    // 4. Create GPU buffers
    if (nodes.empty()) throw std::runtime_error("BVH build resulted in no nodes.");
        nodesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(BVHNode) * nodes.size(), nodes.data()};
    
    if (orderedFaceIndices.empty()) throw std::runtime_error("BVH build resulted in no indices.");
        indicesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * orderedFaceIndices.size(), orderedFaceIndices.data()};
}
void BVH::build(const Context& context, const AABB& aabb) {
    nodes.clear();
    orderedFaceIndices.clear();

    // Only one primitive
    orderedFaceIndices.push_back(0);

    // Create a single leaf node
    BVHNode leaf{};
    leaf.leftBounds = aabb;
    leaf.rightBounds = {};          // Not used
    leaf.rightChildOrPrimIndex = 0; // Index into orderedFaceIndices
    leaf.primCount = 1;             // One primitive
    leaf.splitAxis = 3;             // 3 = leaf

    nodes.push_back(leaf);

    // Upload to GPU
    nodesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(BVHNode) * nodes.size(), nodes.data()};
    indicesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * orderedFaceIndices.size(), orderedFaceIndices.data()};
}

uint32_t BVH::recursiveBuild(std::vector<PrimitiveInfo>& primitiveInfo, uint32_t start, uint32_t end) {
    uint32_t nodeIndex = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back(); // Add a placeholder for the current node

    AABB nodeBounds;
    for (uint32_t i = start; i < end; ++i) {
        nodeBounds.expand(primitiveInfo[i].bounds);
    }
    
    uint32_t count = end - start;

    // Create a leaf node if termination criteria are met 
    if (count <= MAX_LEAF_SIZE) {
        uint32_t firstPrimIndex = static_cast<uint32_t>(orderedFaceIndices.size());
        for (uint32_t i = start; i < end; ++i) {
            orderedFaceIndices.push_back(primitiveInfo[i].faceIndex);
        }
        BVHNode& node = nodes[nodeIndex];
        node.leftBounds = nodeBounds;
        node.rightBounds = {}; // Not used for leaves
        node.rightChildOrPrimIndex = firstPrimIndex;
        node.primCount = count;
        node.splitAxis = 3; // Indicate leaf
        return nodeIndex;
    }

    // Find the best split using Binned SAH for interior nodes 
    AABB centroidBounds;
    for (uint32_t i = start; i < end; ++i) {
        centroidBounds.expand(primitiveInfo[i].centroid);
    }
    
    int bestAxis = -1;
    uint32_t splitIndex = start;
    float bestCost = std::numeric_limits<float>::max();

    // Try splitting on each axis
    for (int axis = 0; axis < 3; ++axis) {
        float axisRange = centroidBounds.maxBounds[axis] - centroidBounds.minBounds[axis];
        if (axisRange < 1e-6f) continue;

        // Binning setup 
        struct Bin { AABB bounds; uint32_t count = 0; };
        Bin bins[SAH_BINS];
        float invAxisRange = 1.0f / axisRange;
        
        // Populate bins
        for (uint32_t i = start; i < end; ++i) {
            float relativePos = (primitiveInfo[i].centroid[axis] - centroidBounds.minBounds[axis]) * invAxisRange;
            int binIdx = std::min(SAH_BINS - 1, static_cast<int>(relativePos * SAH_BINS));
            bins[binIdx].count++;
            bins[binIdx].bounds.expand(primitiveInfo[i].bounds);
        }
        
        // Evaluate splits 
        float rightArea[SAH_BINS - 1];
        AABB rightBox;
        for (int i = SAH_BINS - 1; i > 0; --i) {
            rightBox.expand(bins[i].bounds);
            rightArea[i - 1] = rightBox.surfaceArea();
        }

        AABB leftBox;
        uint32_t leftCount = 0;
        for (int i = 0; i < SAH_BINS - 1; ++i) {
            leftBox.expand(bins[i].bounds);
            leftCount += bins[i].count;
            if (leftCount == 0)
                continue;

            uint32_t rightCount = count - leftCount;
            if (rightCount == 0)
                continue;

            float cost = (leftCount * leftBox.surfaceArea() + rightCount * rightArea[i]) / nodeBounds.surfaceArea();
            
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                // We don't know the exact index yet, just the bin number. We'll partition later.
                splitIndex = i + 1; 
            }
        }
    }
    
    // If no good split found, fall back to making a leaf
    if (bestAxis == -1) {
        uint32_t firstPrimIndex = static_cast<uint32_t>(orderedFaceIndices.size());
        for (uint32_t i = start; i < end; ++i) {
            orderedFaceIndices.push_back(primitiveInfo[i].faceIndex);
        }
        BVHNode& node = nodes[nodeIndex];
        node.leftBounds = nodeBounds;
        node.rightBounds = {};
        node.rightChildOrPrimIndex = firstPrimIndex;
        node.primCount = count;
        node.splitAxis = 3;
        return nodeIndex;
    }

    // Partition the primitives based on the best split found 
    float axisRange = centroidBounds.maxBounds[bestAxis] - centroidBounds.minBounds[bestAxis];
    float invAxisRange = (axisRange > 1e-8f) ? 1.0f / axisRange : 0.0f;

    // Safe partitioning with clamped relativePos
    auto midIt = std::partition(primitiveInfo.begin() + start, primitiveInfo.begin() + end, 
        [&](const PrimitiveInfo& pi) {
            float relativePos = (pi.centroid[bestAxis] - centroidBounds.minBounds[bestAxis]) * invAxisRange;
            relativePos = std::min(0.999999f, std::max(0.0f, relativePos)); // clamp to [0,1)
            int binIdx = static_cast<int>(relativePos * SAH_BINS);
            return binIdx < splitIndex;
        });

    uint32_t midIndex = static_cast<uint32_t>(std::distance(primitiveInfo.begin(), midIt));

    // Fallback: if partition failed (all centroids in same bin), do a middle split
    if (midIndex == start || midIndex == end) {
        midIndex = start + count / 2;
        std::nth_element(
            primitiveInfo.begin() + start,
            primitiveInfo.begin() + midIndex,
            primitiveInfo.begin() + end,
            [bestAxis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                return a.centroid[bestAxis] < b.centroid[bestAxis];
            }
        );
    }
    
    // Recursively build children 
    uint32_t leftChildIndex = recursiveBuild(primitiveInfo, start, midIndex);
    uint32_t rightChildIndex = recursiveBuild(primitiveInfo, midIndex, end);

    // Populate the current (interior) node 
    BVHNode& node = nodes[nodeIndex];
    node.leftBounds = nodes[leftChildIndex].leftBounds; // For leaves, this is the only bound
    if (nodes[leftChildIndex].primCount == 0) // If child is interior, get its full bounds
        node.leftBounds.expand(nodes[leftChildIndex].rightBounds);

    node.rightBounds = nodes[rightChildIndex].leftBounds;
    if (nodes[rightChildIndex].primCount == 0)
        node.rightBounds.expand(nodes[rightChildIndex].rightBounds);
    
    node.rightChildOrPrimIndex = rightChildIndex;
    node.primCount = 0; // 0 indicates an interior node
    node.splitAxis = static_cast<uint8_t>(bestAxis);

    return nodeIndex;
}