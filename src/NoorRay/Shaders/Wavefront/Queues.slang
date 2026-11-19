import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;

//! The work counts of the grid stages. Every stage dispatches the same fixed grid, whose threads stride over its count.
uint activeCount(WavefrontFrame frame) { return atomicLoad(frame.queues.counters + counterActive); }
uint shadowCount(WavefrontFrame frame) { return atomicLoad(frame.queues.counters + counterShadow); }
uint regenerateCount(WavefrontFrame frame) { return frame.options.w != 0 ? atomicLoad(frame.queues.counters + counterFree) : 0u; }
uint binSourceCount(WavefrontFrame frame, uint source) {
    return atomicLoad(frame.queues.counters + ((source & advanceSourceShadow) != 0 ? counterNextShadow : counterActive));
}
uint binnedCount(WavefrontFrame frame, uint bin) { return atomicLoad(frame.queues.binCounts + bin); }

//! The slot of the index-th path of a bin, after sortIntoBin; index is below binnedCount.
uint binnedSlot(WavefrontFrame frame, uint bin, uint index) {
    const QueueRecord queues = frame.queues;
    return queues.shade[atomicLoad(queues.binCursors + bin) - atomicLoad(queues.binCounts + bin) + index];
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
        break;
    }
    case advanceBins: {
        uint offset = 0;
        for (uint bin = 0; bin < queues.binCount; ++bin) {
            const uint count = atomicLoad(queues.binCounts + bin);
            atomicStore(queues.binCursors + bin, offset);
            offset += count;
        }
        if ((phaseAndSource & advanceSourceShadow) != 0)
            atomicStore(counters + counterShadow, 0u);
        break;
    }
    case advanceShadow:
        for (uint bin = 0; bin < queues.binCount; ++bin)
            atomicStore(queues.binCounts + bin, 0u);
        atomicStore(counters + counterNextShadow, 0u);
        break;
    case advanceRegenerate:
        atomicStore(counters + counterShadow, 0u);
        break;
    default: {
        const uint active = atomicLoad(counters + counterNextActive);
        atomicStore(counters + counterActive, active);
        atomicStore(counters + counterNextActive, 0u);
        atomicStore(counters + counterFree, 0u);
        break;
    }
    }
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
