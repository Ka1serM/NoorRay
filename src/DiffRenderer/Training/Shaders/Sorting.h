#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Sorting {

using namespace NoorRHI;

struct ScanArguments {
    Ptr<uint> input;
    Ptr<uint> output;
    //! The sum of each block, written by the reduction and read back as the offset of each block after their scan.
    Ptr<uint> blockSums;
    //! The count on the GPU, or null for maxCount.
    Ptr<uint> count;
    uint maxCount;
};

struct SortArguments {
    Ptr<uint> keys;
    Ptr<uint> values;
    Ptr<uint> sortedKeys;
    Ptr<uint> sortedValues;
    //! Digit-major: the count of digit d in block b is at d x blockCount + b.
    Ptr<uint> histograms;
    Ptr<uint> count;
    uint maxCount;
    uint blockCount;
    uint shift;
    uint digitMask;
    //! 1 for keys of two words, low word first.
    uint wideKeys;
    uint threads;
};

}
