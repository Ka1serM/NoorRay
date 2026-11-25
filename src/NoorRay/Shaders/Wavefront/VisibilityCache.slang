import NoorRHI;
import Wavefront.Records;
import Common.Math;
import Common.Random;
import SceneRecords;

//! A cell is about this many pixels wide at its distance from the camera, rounded up to a power of two, as the levels of SHaRC.
static const float visibilityCellPixels = 16.0;
//! The smallest cell as a share of the scene radius, for an orthographic camera and points near the eye.
static const float visibilityMinCell = 1.0 / 1024.0;
//! A key that is not in its first slots is not stored; the receiver then has no cell.
static const uint visibilityProbes = 8u;
//! Each frame keeps this share of the counts of the earlier frames.
static const float visibilityDecay = 0.9;
//! The traced rays of a choice saturate here, so that the half floats stay exact enough and the cache keeps up with change.
static const float visibilityConfidence = 1024.0;
//! A cell whose decayed traced rays sum to less is evicted.
static const float visibilityEvict = 1.0 / 64.0;

//! A cell splits a node of its cut after this many decayed shadow rays through it.
static const float visibilitySplitRays = 32.0;
//! A cell merges two sibling nodes when the variance that their split explains is below this share of the variance of the best split, and their parent would not split again.
static const float visibilityMergeShare = 0.25;

//! The cell of a surface point of a light set plus one, inserted when it is new; zero when the cache is off or its probes find no free slot.
uint visibilityCell(WavefrontFrame frame, float3 position, float3 normal, uint set) {
    const VisibilityCacheRecord cache = frame.visibility;
    if (isNull(cache.cells))
        return 0;
    const CameraRecord camera = frame.camera[0];
    const float3 eye = float3(camera.cameraToWorld[0].w, camera.cameraToWorld[1].w, camera.cameraToWorld[2].w);
    const float footprint = max(length(position - eye) * camera.projection.w * visibilityCellPixels, frame.scene[0].environmentParams.w * visibilityMinCell);
    const int level = int(ceil(log2(footprint)));
    const int3 grid = int3(floor(position * exp2(float(-level))));
    const float3 extent = abs(normal);
    const uint axis = extent.x >= extent.y && extent.x >= extent.z ? 0u : extent.y >= extent.z ? 1u : 2u;
    const uint face = axis * 2u + (normal[axis] < 0.0 ? 1u : 0u);
    const uint hash = hashCombine(hashCombine(hashCombine(hashPcg(uint(level + 128) | face << 8 | set << 16), uint(grid.x)), uint(grid.y)), uint(grid.z));
    const uint key = hashPcg(hash ^ 0x5bd1e995u) | 1u;
    for (uint probe = 0; probe < visibilityProbes; ++probe) {
        const uint index = (hash + probe) & cache.mask;
        const uint stored = atomicLoad(cache.keys + index);
        if (stored == key)
            return index + 1;
        if (stored != 0)
            continue;
        const uint previous = atomicCompareExchange(cache.keys + index, 0u, key);
        if (previous == 0)
            cache.cells[index * visibilityCellWords + visibilityCellSet] = set;
        if (previous == 0 || previous == key)
            return index + 1;
    }
    return 0;
}

//! The shadow ray entry of a receiver cell and a first choice of the light tree.
uint visibilityEntry(uint cell, uint choice) {
    return cell == 0 || choice >= visibilityChoices ? visibilityNoEntry : (cell - 1) * visibilityChoices + choice;
}

//! Counts a shadow ray of an entry, and whether it arrived.
void recordVisibility(VisibilityCacheRecord cache, uint entry, bool arrived) {
    if (entry == visibilityNoEntry)
        return;
    if (arrived)
        atomicAdd(cache.counts + 2 * entry, 1u);
    atomicAdd(cache.counts + 2 * entry + 1, 1u);
}

float2 cellStats(Ptr<uint> cell, uint choice) { return unpackHalf2(cell[choice]); }

//! The variance of the visibility of the shadow rays of a choice, summed over its rays.
float visibilityVariance(float2 stats) { return stats.y > 0.0 ? stats.x * (1.0 - stats.x / stats.y) : 0.0; }

uint cutSlotCount(Ptr<uint> cell) {
    uint count = 0;
    while (count < visibilityCutSize && cell[visibilityCellCut + count] != invalidIndex)
        ++count;
    return count;
}

uint findCutSlot(Ptr<uint> cell, uint count, uint node) {
    for (uint slot = 0; slot < count; ++slot)
        if (cell[visibilityCellCut + slot] == node)
            return slot;
    return invalidIndex;
}

//! Replaces two sibling nodes of the cut by their parent and moves the last node into the freed slot; returns that slot.
uint mergeSiblings(Ptr<uint> cell, uint count, uint first, uint second, uint parent) {
    const uint kept = min(first, second);
    const uint freed = max(first, second);
    cell[visibilityCellCut + kept] = parent;
    cell[visibilityChoiceCut + kept] = packHalf2(cellStats(cell, visibilityChoiceCut + first) + cellStats(cell, visibilityChoiceCut + second));
    const uint last = count - 1;
    cell[visibilityCellCut + freed] = cell[visibilityCellCut + last];
    cell[visibilityChoiceCut + freed] = cell[visibilityChoiceCut + last];
    cell[visibilityCellCut + last] = invalidIndex;
    cell[visibilityChoiceCut + last] = 0u;
    return freed;
}

//! Refines the cut of a cell where its visibility varies most, so that the light tree learns occlusion below the cut of its set, as Wang et al. 2021.
void refineCut(SceneRecord scene, Ptr<uint> cell) {
    const LightTreeSet tree = scene.lightSets[cell[visibilityCellSet]];
    if (cell[visibilityCellCut] - tree.root >= tree.nodeCount) {
        for (uint slot = 0; slot < visibilityCutSize; ++slot)
            cell[visibilityCellCut + slot] = slot < tree.cutCount ? tree.cut[slot] : invalidIndex;
        return;
    }
    uint count = cutSlotCount(cell);
    uint split = invalidIndex;
    float splitVariance = 0.0;
    for (uint slot = 0; slot < count; ++slot) {
        const float2 stats = cellStats(cell, visibilityChoiceCut + slot);
        const float variance = visibilityVariance(stats);
        if (stats.y < visibilitySplitRays || variance <= splitVariance || (scene.lightNodes[cell[visibilityCellCut + slot]].meta.y & lightNodeLeaf) != 0)
            continue;
        split = slot;
        splitVariance = variance;
    }
    if (split == invalidIndex)
        return;
    if (count == visibilityCutSize) {
        uint first = invalidIndex;
        uint second = invalidIndex;
        float least = visibilityMergeShare * splitVariance;
        for (uint slot = 0; slot < count; ++slot) {
            const uint node = cell[visibilityCellCut + slot];
            if (slot == split || node <= tree.root || (scene.lightNodes[node - 1].meta.y & lightNodeLeaf) != 0)
                continue;
            const uint sibling = findCutSlot(cell, count, scene.lightNodes[node - 1].meta.x);
            if (sibling == invalidIndex || sibling == split)
                continue;
            const float2 a = cellStats(cell, visibilityChoiceCut + slot);
            const float2 b = cellStats(cell, visibilityChoiceCut + sibling);
            const float merged = visibilityVariance(a + b);
            const float explained = merged - visibilityVariance(a) - visibilityVariance(b);
            if (explained >= least || merged >= splitVariance)
                continue;
            first = slot;
            second = sibling;
            least = explained;
        }
        if (first == invalidIndex)
            return;
        const uint freed = mergeSiblings(cell, count, first, second, cell[visibilityCellCut + first] - 1);
        if (split == count - 1)
            split = freed;
        --count;
    }
    const uint node = cell[visibilityCellCut + split];
    const uint half = packHalf2(0.5 * cellStats(cell, visibilityChoiceCut + split));
    cell[visibilityCellCut + split] = node + 1;
    cell[visibilityChoiceCut + split] = half;
    cell[visibilityCellCut + count] = scene.lightNodes[node].meta.x;
    cell[visibilityChoiceCut + count] = half;
}

void clearCell(Ptr<uint> cell) {
    for (uint choice = 0; choice < visibilityChoices; ++choice)
        cell[choice] = 0u;
    for (uint slot = 0; slot < visibilityCutSize; ++slot)
        cell[visibilityCellCut + slot] = invalidIndex;
}

//! Between frames, folds the counts of a cell into its decayed stats, refines its cut, and evicts the cell when it saw too few rays; clear empties the cache.
void resolveVisibility(VisibilityCacheRecord cache, SceneRecord scene, uint index, bool clear) {
    const Ptr<uint> cell = cache.cells + index * visibilityCellWords;
    if (!clear && atomicLoad(cache.keys + index) == 0)
        return;
    float traced = 0.0;
    for (uint choice = 0; choice < visibilityChoices; ++choice) {
        const uint entry = index * visibilityChoices + choice;
        const float arrived = float(atomicExchange(cache.counts + 2 * entry, 0u));
        const float rays = float(atomicExchange(cache.counts + 2 * entry + 1, 0u));
        float2 stats = clear ? float2(0.0, 0.0) : cellStats(cell, choice) * visibilityDecay + float2(arrived, rays);
        if (stats.y > visibilityConfidence)
            stats *= visibilityConfidence / stats.y;
        cell[choice] = packHalf2(stats);
        traced += stats.y;
    }
    if (!clear && traced >= visibilityEvict) {
        refineCut(scene, cell);
        return;
    }
    atomicStore(cache.keys + index, 0u);
    clearCell(cell);
}
