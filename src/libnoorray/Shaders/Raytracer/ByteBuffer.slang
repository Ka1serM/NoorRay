#pragma once

#include "../../Shared/Types.h"

// A shader-only view over a GpuPtr(uint). It provides byte-offset loads for
// packed data without becoming part of the shared CPU/GPU record ABI.
//
// This lives in its own header because leaf shading code (Spectrum, EnergyLut,
// the BSDF lobes, Gaussian) takes these as parameters and is compiled both
// into the renderer's entry points and standalone by the tests.
struct ByteBuffer
{
    uint* words;

    bool valid() { return words != nullptr; }
    uint Load(uint byteOffset) { return words[byteOffset >> 2]; }
    // Two-word load for 64-bit values (device addresses) stored inside packed
    // records; the host writes them little-endian, low word first.
    uint64_t Load64(uint byteOffset)
    {
        const uint index = byteOffset >> 2;
        return (uint64_t(words[index + 1u]) << 32) | uint64_t(words[index]);
    }
};

ByteBuffer byteBuffer(GpuPtr(uint) words)
{
    ByteBuffer result;
    result.words = words;
    return result;
}
