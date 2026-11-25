import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Scene.SceneSurface;

//! The work counts of the grid stages. Every stage dispatches the same fixed grid, whose threads stride over its count.
uint activeCount(WavefrontFrame frame) { return atomicLoad(frame.queues.counters + counterActive); }
uint shadowCount(WavefrontFrame frame) { return atomicLoad(frame.queues.counters + counterShadow); }
uint regenerateCount(WavefrontFrame frame) { return frame.options.w != 0 ? atomicLoad(frame.queues.counters + counterFree) : 0u; }
uint binSourceCount(WavefrontFrame frame, uint source) {
    return atomicLoad(frame.queues.counters + ((source & advanceSourceShadow) != 0 ? counterNextShadow : counterActive));
}
uint binnedCount(WavefrontFrame frame, uint bin) { return atomicLoad(frame.queues.binCounts + bin); }

//! The surface bins lie one after another in shade, so the shade kernels read them as one range from the first surface bin on.
uint surfaceStart(WavefrontFrame frame) { return atomicLoad(frame.queues.binCursors + fixedMaterialBin) - binnedCount(frame, fixedMaterialBin); }
uint surfaceCount(WavefrontFrame frame) { return atomicLoad(frame.queues.binCursors + (frame.queues.binCount - 1)) - surfaceStart(frame); }
uint surfaceSlot(WavefrontFrame frame, uint start, uint index) { return frame.queues.shade[start + index]; }

//! The bin of a surface hit: the bin of the material program of its triangle.
uint surfaceBin(SceneRecord scene, uint instance, uint primitive) {
    const MaterialRecord material = scene.materials[triangleMaterial(scene, instance, primitive)];
    return (material.flags & materialFlagGraph) != 0 ? firstGraphBin + material.graph.x : fixedMaterialBin;
}

//! The slot of the index-th path of a bin, after sortIntoBin; index is below binnedCount.
uint binnedSlot(WavefrontFrame frame, uint bin, uint index) {
    const QueueRecord queues = frame.queues;
    return queues.shade[atomicLoad(queues.binCursors + bin) - atomicLoad(queues.binCounts + bin) + index];
}

//! The groups of a grid stage over count; a full grid strides over the rest.
void writeGridDispatch(QueueRecord queues, uint slot, uint count) {
    queues.gridDispatch[slot * 3] = min((count + wavefrontGroupSize - 1) / wavefrontGroupSize, queues.gridThreads / wavefrontGroupSize);
    queues.gridDispatch[slot * 3 + 1] = 1u;
    queues.gridDispatch[slot * 3 + 2] = 1u;
}

void advanceShadowQueue(WavefrontFrame frame) {
    const QueueRecord queues = frame.queues;
    for (uint bin = 0; bin < queues.binCount; ++bin)
        atomicStore(queues.binCounts + bin, 0u);
    atomicStore(queues.counters + counterNextShadow, 0u);
    writeGridDispatch(queues, gridShadow, shadowCount(frame));
}

void swapActiveQueue(WavefrontFrame frame) {
    const QueueRecord queues = frame.queues;
    const AtomicPtr<uint> counters = queues.counters;
    const uint active = atomicLoad(counters + counterNextActive);
    atomicStore(counters + counterActive, active);
    atomicStore(counters + counterNextActive, 0u);
    atomicStore(counters + counterFree, 0u);
    writeGridDispatch(queues, gridActive, active);
}

//! One thread: the bookkeeping between two wavefront stages, see advance*.
void advanceQueues(WavefrontFrame frame, uint parity, uint phaseAndSource) {
    const QueueRecord queues = frame.queues;
    const AtomicPtr<uint> counters = queues.counters;
    const uint phase = phaseAndSource & 0xffu;
    switch (phase) {
    case advanceReset:
        for (uint index = 0; index < counterCount; ++index)
            atomicStore(counters + index, 0u);
        for (uint bin = 0; bin < queues.binCount; ++bin)
            atomicStore(queues.binCounts + bin, 0u);
        break;
    case advanceGenerate: {
        const uint count = min(atomicLoad(counters + counterSamples), frame.pool.capacity);
        atomicStore(counters + counterActive, count);
        atomicStore(counters + counterSampleCursor, count);
        writeGridDispatch(queues, gridActive, count);
        break;
    }
    case advanceBins: {
        uint offset = 0;
        for (uint bin = 0; bin < queues.binCount; ++bin) {
            const uint count = atomicLoad(queues.binCounts + bin);
            atomicStore(queues.binCursors + bin, offset);
            offset += count;
            queues.binDispatch[bin * 3] = (count + wavefrontGroupSize - 1) / wavefrontGroupSize;
            queues.binDispatch[bin * 3 + 1] = 1u;
            queues.binDispatch[bin * 3 + 2] = 1u;
        }
        const uint surfaceStart = atomicLoad(queues.binCursors + fixedMaterialBin);
        writeGridDispatch(queues, gridSurface, offset - surfaceStart);
        writeGridDispatch(queues, gridMedium, atomicLoad(queues.binCounts + mediumBin));
        writeGridDispatch(queues, gridBinned, offset);
        writeGridDispatch(queues, gridBinSource, binSourceCount(frame, phaseAndSource));
        if ((phaseAndSource & advanceSourceShadow) != 0)
            atomicStore(counters + counterShadow, 0u);
        break;
    }
    case advanceShadow:
        advanceShadowQueue(frame);
        break;
    case advanceRegenerate:
        atomicStore(counters + counterShadow, 0u);
        writeGridDispatch(queues, gridFree, regenerateCount(frame));
        break;
    default:
        swapActiveQueue(frame);
        break;
    }
    if ((phaseAndSource & advanceThenSwap) != 0)
        swapActiveQueue(frame);
}

//! Copies each binned path into the range of its bin, so that one shade kernel reads one contiguous range.
void sortIntoBin(WavefrontFrame frame, uint parity, uint source, uint index) {
    const QueueRecord queues = frame.queues;
    const bool shadow = (source & advanceSourceShadow) != 0;
    const uint slot = shadow ? shadowQueue(queues, parity ^ 1)[index] : activeQueue(queues, parity)[index];
    const uint bin = frame.pool.pathBin[slot];
    if (bin >= queues.binCount)
        return;
    const uint position = atomicAdd(queues.binCursors + bin, 1u);
    queues.shade[position] = slot;
}

void pushBin(WavefrontFrame frame, uint slot, uint bin) {
    frame.pool.pathBin[slot] = bin;
    if (bin < frame.queues.binCount)
        atomicAdd(frame.queues.binCounts + bin, 1u);
}

void pushFree(WavefrontFrame frame, uint slot) {
    const uint index = atomicAdd(frame.queues.counters + counterFree, 1u);
    frame.queues.free[index] = slot;
}
