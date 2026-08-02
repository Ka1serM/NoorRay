#pragma once

#include <cstdint>

#include "Backend/CUDA/Annotations.h"

// Joe-Kuo direction numbers for Sobol dimensions 2 through 6. Keeping the
// initializer shared prevents the CPU and CUDA lookup tables from drifting.
#define NR_OWEN_SOBOL_DIRECTION_NUMBERS \
    { \
        { \
            0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u, \
            0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u, \
            0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u, \
            0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu, \
        }, \
        { \
            0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u, \
            0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u, 0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u, \
            0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u, 0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u, \
            0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u, 0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u, \
        }, \
        { \
            0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u, \
            0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u, 0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u, \
            0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u, 0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u, \
            0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u, 0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u, \
        }, \
        { \
            0x80000000u, 0x40000000u, 0x20000000u, 0xb0000000u, 0xf8000000u, 0xdc000000u, 0x7a000000u, 0x9d000000u, \
            0x5a800000u, 0x2fc00000u, 0xa1600000u, 0xf0b00000u, 0xda880000u, 0x6fc40000u, 0x81620000u, 0x40bb0000u, \
            0x22878000u, 0xb3c9c000u, 0xfb65a000u, 0xddb2d000u, 0x78022800u, 0x9c0b3c00u, 0x5a0fb600u, 0x2d0ddb00u, \
            0xa2878080u, 0xf3c9c040u, 0xdb65a020u, 0x6db2d0b0u, 0x800228f8u, 0x400b3cdcu, 0x200fb67au, 0xb00ddb9du, \
        }, \
        { \
            0x80000000u, 0x40000000u, 0x60000000u, 0x30000000u, 0xc8000000u, 0x24000000u, 0x56000000u, 0xfb000000u, \
            0xe0800000u, 0x70400000u, 0xa8600000u, 0x14300000u, 0x9ec80000u, 0xdf240000u, 0xb6d60000u, 0x8bbb0000u, \
            0x48008000u, 0x64004000u, 0x36006000u, 0xcb003000u, 0x2880c800u, 0x54402400u, 0xfe605600u, 0xef30fb00u, \
            0x7e48e080u, 0xaf647040u, 0x1eb6a860u, 0x9f8b1430u, 0xd6c81ec8u, 0xbb249f24u, 0x80d6d6d6u, 0x40bbbbbbu, \
        }, \
    }

namespace owen_sobol_detail
{
inline constexpr uint32_t DirectionNumbers[5][32] =
    NR_OWEN_SOBOL_DIRECTION_NUMBERS;

#if defined(__CUDACC__)
// Rendering uses a frame-wide sample index, allowing warp-wide broadcasts
// from CUDA constant memory instead of one table per thread.
static __device__ __constant__ uint32_t DeviceDirectionNumbers[5][32] =
    NR_OWEN_SOBOL_DIRECTION_NUMBERS;
#endif
}

#undef NR_OWEN_SOBOL_DIRECTION_NUMBERS
