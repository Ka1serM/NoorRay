#pragma once


#include <cstdint>

#include <cuda_runtime_api.h>

struct OutputSurfaces
{
    cudaSurfaceObject_t color{};
    cudaSurfaceObject_t albedo{};
    cudaSurfaceObject_t normal{};
    cudaSurfaceObject_t cryptomatte{};
    cudaSurfaceObject_t position{};
    cudaSurfaceObject_t material{};
    uint32_t width{};
    uint32_t height{};
};
