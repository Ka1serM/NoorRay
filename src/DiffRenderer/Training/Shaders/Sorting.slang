#include "Sorting.h"

typealias ScanArguments = Sorting::ScanArguments;
typealias SortArguments = Sorting::SortArguments;

static const uint groupSize = 256;
static const uint scanItems = 8;
static const uint scanBlock = groupSize * scanItems;
static const uint sortRounds = 16;
static const uint digitCount = 256;
static const uint maxWaves = groupSize / 16;

groupshared uint scanValues[scanBlock];
groupshared uint waveTotals[maxWaves];
groupshared uint digitOffsets[digitCount];
groupshared uint waveDigits[maxWaves * digitCount];

uint getCount(Ptr<uint> count, uint maxCount) {
    return isNull(count) ? maxCount : min(count[0], maxCount);
}

uint getWaveCount() {
    return groupSize / WaveGetLaneCount();
}

uint getGroupSum(uint value, uint thread) {
    const uint waveSum = WaveActiveSum(value);
    if (WaveIsFirstLane())
        waveTotals[thread / WaveGetLaneCount()] = waveSum;
    GroupMemoryBarrierWithGroupSync();
    uint sum = 0;
    for (uint wave = 0; wave < getWaveCount(); ++wave)
        sum += waveTotals[wave];
    return sum;
}

uint getGroupPrefix(uint value, uint thread) {
    const uint wavePrefix = WavePrefixSum(value);
    if (WaveGetLaneIndex() == WaveGetLaneCount() - 1)
        waveTotals[thread / WaveGetLaneCount()] = wavePrefix + value;
    GroupMemoryBarrierWithGroupSync();
    uint prefix = wavePrefix;
    for (uint wave = 0; wave < thread / WaveGetLaneCount(); ++wave)
        prefix += waveTotals[wave];
    return prefix;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduceBlocks(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const ScanArguments arguments = rootArguments<ScanArguments>();
    const uint count = getCount(arguments.count, arguments.maxCount);
    uint sum = 0;
    for (uint item = 0; item < scanItems; ++item) {
        const uint index = group.x * scanBlock + item * groupSize + thread;
        if (index < count)
            sum += arguments.input[index];
    }
    const uint total = getGroupSum(sum, thread);
    if (thread == 0)
        arguments.blockSums[group.x] = total;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scanBlocks(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const ScanArguments arguments = rootArguments<ScanArguments>();
    const uint count = getCount(arguments.count, arguments.maxCount);
    const uint start = group.x * scanBlock;
    for (uint item = 0; item < scanItems; ++item) {
        const uint index = start + item * groupSize + thread;
        scanValues[item * groupSize + thread] = index < count ? arguments.input[index] : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint sum = 0;
    for (uint item = 0; item < scanItems; ++item)
        sum += scanValues[thread * scanItems + item];
    uint prefix = getGroupPrefix(sum, thread) + (isNull(arguments.blockSums) ? 0 : arguments.blockSums[group.x]);
    for (uint item = 0; item < scanItems; ++item) {
        const uint value = scanValues[thread * scanItems + item];
        scanValues[thread * scanItems + item] = prefix;
        prefix += value;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint item = 0; item < scanItems; ++item) {
        const uint index = start + item * groupSize + thread;
        if (index < count)
            arguments.output[index] = scanValues[item * groupSize + thread];
    }
}

uint2 loadKey(SortArguments arguments, Ptr<uint> keys, uint index) {
    if (arguments.wideKeys != 0)
        return uint2(keys[index * 2], keys[index * 2 + 1]);
    return uint2(keys[index], 0);
}

void storeKey(SortArguments arguments, Ptr<uint> keys, uint index, uint2 key) {
    if (arguments.wideKeys != 0) {
        keys[index * 2] = key.x;
        keys[index * 2 + 1] = key.y;
        return;
    }
    keys[index] = key.x;
}

uint getDigit(SortArguments arguments, uint2 key) {
    const uint word = arguments.shift < 32 ? key.x >> arguments.shift : key.y >> (arguments.shift - 32);
    return word & arguments.digitMask;
}

uint countLanes(uint4 mask) {
    return countbits(mask.x) + countbits(mask.y) + countbits(mask.z) + countbits(mask.w);
}

uint4 getLowerLanes(uint lane) {
    uint4 mask;
    for (uint word = 0; word < 4; ++word) {
        const uint begin = word * 32;
        mask[word] = lane >= begin + 32 ? 0xffffffff : (lane <= begin ? 0 : (1u << (lane - begin)) - 1);
    }
    return mask;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void countDigits(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const SortArguments arguments = rootArguments<SortArguments>();
    const uint count = getCount(arguments.count, arguments.maxCount);
    digitOffsets[thread] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint round = 0; round < sortRounds; ++round) {
        const uint index = (group.x * sortRounds + round) * groupSize + thread;
        if (index < count)
            InterlockedAdd(digitOffsets[getDigit(arguments, loadKey(arguments, arguments.keys, index))], 1);
    }
    GroupMemoryBarrierWithGroupSync();
    arguments.histograms[thread * arguments.blockCount + group.x] = digitOffsets[thread];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scatterDigits(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    const SortArguments arguments = rootArguments<SortArguments>();
    const uint count = getCount(arguments.count, arguments.maxCount);
    const uint waves = getWaveCount();
    const uint wave = thread / WaveGetLaneCount();
    const uint4 lowerLanes = getLowerLanes(WaveGetLaneIndex());
    digitOffsets[thread] = arguments.histograms[thread * arguments.blockCount + group.x];
    for (uint round = 0; round < sortRounds; ++round) {
        for (uint other = 0; other < waves; ++other)
            waveDigits[other * digitCount + thread] = 0;
        GroupMemoryBarrierWithGroupSync();
        const uint index = (group.x * sortRounds + round) * groupSize + thread;
        const bool valid = index < count;
        const uint2 key = valid ? loadKey(arguments, arguments.keys, index) : uint2(0);
        const uint digit = getDigit(arguments, key);
        uint4 peers = WaveActiveBallot(valid);
        for (uint bit = 0; bit < 8; ++bit) {
            const uint4 set = WaveActiveBallot(((digit >> bit) & 1) != 0);
            peers &= ((digit >> bit) & 1) != 0 ? set : ~set;
        }
        const uint rank = countLanes(peers & lowerLanes);
        const uint peerCount = countLanes(peers);
        if (valid && rank + 1 == peerCount)
            waveDigits[wave * digitCount + digit] = peerCount;
        GroupMemoryBarrierWithGroupSync();
        uint running = digitOffsets[thread];
        for (uint other = 0; other < waves; ++other) {
            const uint digitCountInWave = waveDigits[other * digitCount + thread];
            waveDigits[other * digitCount + thread] = running;
            running += digitCountInWave;
        }
        digitOffsets[thread] = running;
        GroupMemoryBarrierWithGroupSync();
        if (valid) {
            const uint target = waveDigits[wave * digitCount + digit] + rank;
            storeKey(arguments, arguments.sortedKeys, target, key);
            arguments.sortedValues[target] = arguments.values[index];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void copyPairs(uint3 id : SV_DispatchThreadID) {
    const SortArguments arguments = rootArguments<SortArguments>();
    const uint count = getCount(arguments.count, arguments.maxCount);
    for (uint index = id.x; index < count; index += arguments.threads) {
        storeKey(arguments, arguments.keys, index, loadKey(arguments, arguments.sortedKeys, index));
        arguments.values[index] = arguments.sortedValues[index];
    }
}
